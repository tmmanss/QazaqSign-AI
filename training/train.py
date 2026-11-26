"""Обучение классификатора жестов на MediaPipe-точках из Slovo.

Фаза A: пул кандидатов + класс _other → точность по словам на официальном тесте
        (5 видео/слово, справочно) и 5-fold кросс-валидация по носителям внутри train
        (~15 видео/слово). Автоотбор 10–12 слов по F1 из CV; слова FORCE включаются всегда.
Фаза B: переобучение только на отобранных словах + _other → экспорт ONNX и labels.json
        в web/public/model/.

Запуск: python training/prepare_data.py   (один раз, создаёт data/cache/landmarks.npz)
        python training/train.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import FEAT_DIM, SEQ_LEN, assign_slots, raw_to_features, resample_idx, seq_features, to_pixels

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "landmarks.npz"
ANN = ROOT / "data" / "annotations.csv"
OUT_MODEL = ROOT / "web" / "public" / "model"
REPORTS = ROOT / "training" / "reports"
OTHER = "_other"
SEED = 42

# Слово в датасете -> (как показывать по-русски, перевод на казахский, приоритет)
# priority 1 = из исходного списка (точное совпадение), 2 = близкая замена, 3 = полезное доп. слово
POOL = {
    "Привет!": ("привет", "сәлем", 1),
    "да": ("да", "иә", 1),
    "деньги": ("деньги", "ақша", 1),
    "вода": ("вода", "су", 1),
    "хорошо": ("хорошо", "жақсы", 1),
    "Плохо": ("плохо", "жаман", 1),
    "ждать": ("ждать", "күту", 1),
    "помочь": ("помочь", "көмектесу", 2),
    "благодарность": ("благодарность", "алғыс", 2),
    "болит": ("болит", "ауырады", 2),
    "боль": ("боль", "ауырсыну", 2),
    "понять": ("понять", "түсіну", 2),
    "еще нет": ("ещё нет", "әлі жоқ", 2),
    "бумага": ("бумага", "қағаз", 2),
    "платить": ("платить", "төлеу", 3),
    "цена": ("цена", "баға", 3),
    "сколько": ("сколько", "қанша", 3),
    "что": ("что", "не", 3),
    "кто?": ("кто?", "кім?", 3),
    "когда": ("когда", "қашан", 3),
    "время": ("время", "уақыт", 3),
    "сегодня": ("сегодня", "бүгін", 3),
    "завтра": ("завтра", "ертең", 3),
    "имя": ("имя", "есім", 3),
    "вопрос": ("вопрос", "сұрақ", 3),
    "говорить": ("говорить", "сөйлеу", 3),
    "слышать": ("слышать", "есту", 3),
    "голова": ("голова", "бас", 3),
    "Пока": ("пока", "сау бол", 3),
    "Добро пожаловать!": ("добро пожаловать", "қош келдіңіз", 3),
    "пить": ("пить", "ішу", 3),
    "знать": ("знать", "білу", 3),
}
OTHER_LABEL = ("—", "—")

N_SELECT_MIN, N_SELECT_MAX = 10, 12
PRIORITY_MIN_F1 = 0.6  # слова из исходного списка берём в приоритете, если F1 не ниже
FORCE = ["Привет!"]  # важно для демо — включается всегда
# Порог на слово подбирается так, чтобы на «живых» окнах чужих жестов (неподвижные руки,
# половинки жестов, паузы) это слово ложно срабатывало не чаще FP_TARGET.
FP_TARGET = 0.005
THR_MIN, THR_MAX = 0.5, 0.97
CV_FOLDS = 5

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, torch.get_num_threads()))


# ---------------------------------------------------------------- данные
def load_videos():
    d = np.load(CACHE)
    ids, off, frames = d["ids"], d["offsets"], d["frames"]
    ann = pd.read_csv(ANN, sep="\t")
    ann = ann[ann.text != "no_event"]
    idx = {v: i for i, v in enumerate(ids)}
    videos = []
    for r in ann.itertuples():
        i = idx.get(r.attachment_id)
        if i is None or off[i + 1] - off[i] < 4:
            continue
        videos.append(dict(
            id=r.attachment_id, text=r.text, user=r.user_id, train=bool(r.train),
            W=float(r.width), H=float(r.height), raw=frames[off[i]:off[i + 1]].astype(np.float32),
        ))
    return videos


# ---------------------------------------------------------------- аугментация
def augment(raw, W, H):
    """Аугментация в пикселях до раскладки по слотам. Возвращает признаки [SEQ_LEN, FEAT_DIM]."""
    px = to_pixels(raw, W, H)
    T = len(px)
    # временная обрезка (в браузере окно может захватить жест не целиком) и разная скорость
    keep = int(round(T * rng.uniform(0.75, 1.0)))
    keep = max(4, min(T, keep))
    start = rng.integers(0, T - keep + 1)
    px = px[start:start + keep]
    # зеркалирование: левая/правая рука, фронтальная камера
    if rng.random() < 0.5:
        px[..., 0] = W - px[..., 0]
        px[..., 2] = -px[..., 2]
    # поворот + анизотропный масштаб вокруг центра
    valid = ~np.isnan(px[..., 0])
    c = np.array([np.nanmean(px[..., 0]), np.nanmean(px[..., 1])])
    a = np.deg2rad(rng.uniform(-15, 15))
    sx, sy = rng.uniform(0.9, 1.1, 2)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag([sx, sy])
    xy = px[..., :2] - c
    px[..., :2] = xy @ R.T + c
    px[..., 2] *= (sx + sy) / 2
    # шум пропорционально размеру ладони
    ps = np.nanmedian(np.linalg.norm(px[:, :, 9, :2] - px[:, :, 0, :2], axis=-1))
    if np.isfinite(ps):
        px += rng.normal(0, 0.03 * ps, px.shape).astype(np.float32)
        # дрожание положения всей руки
        px[..., :2] += rng.normal(0, 0.1 * ps, (len(px), 2, 1, 2)).astype(np.float32)
    # пропуски детекции отдельной руки
    drop = rng.random((len(px), 2)) < 0.05
    px[drop] = np.nan
    px[~valid] = np.nan
    slots = assign_slots(px, W)
    return seq_features(slots[resample_idx(len(slots))])


def with_holds(raw):
    """Неподвижные руки до/после жеста — как в живом окне камеры."""
    T = len(raw)
    a, b = rng.integers(0, int(T * 0.4) + 1, 2)
    return np.concatenate([np.repeat(raw[:1], a, 0), raw, np.repeat(raw[-1:], b, 0)])


def partial(raw):
    """Кусок жеста (30–60%) — окно захватило начало, конец или середину."""
    T = len(raw)
    n = max(6, int(T * rng.uniform(0.3, 0.6)))
    s = rng.integers(0, max(1, T - n + 1))
    return raw[s:s + n]


def other_variant(v):
    """«Не слово» в разных видах, которые встречаются в окне камеры."""
    r = rng.random()
    if r < 0.4:
        return v["raw"]
    if r < 0.6:
        return static_sample(v)
    if r < 0.8:
        return partial(v["raw"])
    return with_holds(partial(v["raw"]))


def static_sample(v):
    """Синтетический «покой»: один кадр, повторённый с дрожанием → класс _other."""
    raw = v["raw"]
    f = raw[rng.integers(len(raw))]
    seq = np.repeat(f[None], rng.integers(20, 50), axis=0)
    seq = seq + rng.normal(0, 0.002, seq.shape).astype(np.float32)
    return seq


# ---------------------------------------------------------------- модель
class SignCNN(nn.Module):
    def __init__(self, n_classes, feat=FEAT_DIM, ch=128, p=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(0.1), nn.Conv1d(feat, ch, 1))
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(p))
            for _ in range(3)
        ])
        self.head = nn.Sequential(nn.Linear(ch * 2, ch), nn.ReLU(), nn.Dropout(p), nn.Linear(ch, n_classes))

    def forward(self, x):  # x: [B, SEQ_LEN, FEAT_DIM]
        h = self.inp(x.transpose(1, 2))
        for i, b in enumerate(self.blocks):
            h = b(h) + h if i else b(h)
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.head(h)


class WithSoftmax(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return F.softmax(self.m(x), -1)


# ---------------------------------------------------------------- обучение / оценка
def build_split(videos, words):
    """Делит видео на train/test для набора слов + _other (из слов вне набора)."""
    wset = set(words)
    tr = [v for v in videos if v["train"] and v["text"] in wset]
    te = [v for v in videos if not v["train"] and v["text"] in wset]
    others = sorted({v["text"] for v in videos} - wset - set(POOL))
    perm = rng.permutation(others)
    o_tr_words, o_te_words = set(perm[:300]), set(perm[300:400])  # разные слова в train и test
    o_tr = [v for v in videos if v["train"] and v["text"] in o_tr_words]
    o_te = [v for v in videos if not v["train"] and v["text"] in o_te_words]
    o_te = list(rng.choice(o_te, size=min(len(o_te), 150), replace=False))
    # третий непересекающийся набор слов — для подбора порогов
    o_cal = [v for v in videos if v["text"] in set(perm[400:])]
    o_cal = list(rng.choice(o_cal, size=min(len(o_cal), 800), replace=False))
    return tr, te, o_tr, o_te, o_cal


def train_model(words, tr, o_tr, epochs=70, log=""):
    classes = list(words) + [OTHER]
    ci = {c: i for i, c in enumerate(classes)}
    model = SignCNN(len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * 40)
    n_other = max(120, len(tr) // len(words) * 8)  # _other ~ как 4 слова за эпоху (по 2 копии)
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        items = []
        for v in tr:
            for _ in range(2):
                raw = with_holds(v["raw"]) if rng.random() < 0.5 else v["raw"]
                items.append((raw, v["W"], v["H"], ci[v["text"]]))
        for v in rng.choice(o_tr, size=n_other, replace=False):
            items.append((other_variant(v), v["W"], v["H"], ci[OTHER]))
        order = rng.permutation(len(items))
        X = np.stack([augment(*items[i][:3]) for i in order])
        Y = np.array([items[i][3] for i in order])
        model.train()
        bs = int(np.ceil(len(X) / 40))
        tot = 0.0
        for b in range(40):
            xb = torch.from_numpy(X[b * bs:(b + 1) * bs])
            yb = torch.from_numpy(Y[b * bs:(b + 1) * bs])
            if len(xb) < 2:
                sched.step(); continue
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); step += 1
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  {log} epoch {ep:3d} loss {tot / 40:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return model, classes


@torch.no_grad()
def predict(model, vids):
    model.eval()
    X = np.stack([raw_to_features(v["raw"], v["W"], v["H"]) for v in vids])
    return F.softmax(model(torch.from_numpy(X)), -1).numpy()


def evaluate(model, classes, te, o_te, train_users, title):
    vids = te + o_te
    y = np.array([classes.index(v["text"]) if v["text"] in classes else classes.index(OTHER) for v in vids])
    p = predict(model, vids)
    pred = p.argmax(1)
    unseen = np.array([v["user"] not in train_users for v in vids])
    n = len(classes)
    cm = np.zeros((n, n), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    rows = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        rec = tp / max(cm[i].sum(), 1)
        prec = tp / max(cm[:, i].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        m = (y == i) & unseen
        rec_u = (pred[m] == i).mean() if m.any() else np.nan
        rows.append(dict(word=c, n=int(cm[i].sum()), acc=rec, precision=prec, f1=f1, acc_unseen_users=rec_u, n_unseen=int(m.sum())))
    df = pd.DataFrame(rows)
    print(f"\n=== {title} ===")
    print(f"overall acc: {(pred == y).mean():.3f}   on unseen users: {(pred[unseen] == y[unseen]).mean():.3f} (n={unseen.sum()})")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return df, cm


def print_cm(cm, classes, title):
    short = [c[:6] for c in classes]
    print(f"\nconfusion matrix ({title}), rows = true, cols = predicted:")
    print("        " + " ".join(f"{s:>6}" for s in short))
    for c, row in zip(short, cm):
        print(f"{c:>7} " + " ".join(f"{x:6d}" if x else "     ." for x in row))


def cross_validate(videos, words, epochs=60):
    """Кросс-валидация по носителям внутри train: ~15 оценок на слово вместо 5."""
    tr, _, o_tr, _, _ = build_split(videos, words)
    users = np.array(sorted({v["user"] for v in tr + o_tr}))
    fold_of = {u: i % CV_FOLDS for i, u in enumerate(rng.permutation(users))}
    classes = list(words) + [OTHER]
    ys, preds = [], []
    for k in range(CV_FOLDS):
        fit = [v for v in tr if fold_of[v["user"]] != k]
        fit_o = [v for v in o_tr if fold_of[v["user"]] != k]
        val = [v for v in tr if fold_of[v["user"]] == k]
        val_o = [v for v in o_tr if fold_of[v["user"]] == k]
        val_o = list(rng.choice(val_o, size=min(len(val_o), 60), replace=False))
        model, _ = train_model(words, fit, fit_o, epochs=epochs, log=f"CV{k}")
        vids = val + val_o
        ys += [classes.index(v["text"]) if v["text"] in classes else len(classes) - 1 for v in vids]
        preds += list(predict(model, vids).argmax(1))
    y, pred = np.array(ys), np.array(preds)
    n = len(classes)
    cm = np.zeros((n, n), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    rows = []
    for i, c in enumerate(classes):
        rec = cm[i, i] / max(cm[i].sum(), 1)
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        rows.append(dict(word=c, n=int(cm[i].sum()), acc=rec, precision=prec,
                         f1=2 * prec * rec / max(prec + rec, 1e-9)))
    df = pd.DataFrame(rows)
    print(f"\n=== PHASE A — {CV_FOLDS}-fold CV by user (train split) ===")
    print(f"overall acc: {(pred == y).mean():.3f}")
    print(df.sort_values("f1", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return df, cm


def select_words(df):
    d = df[df.word != OTHER].copy()
    d["prio"] = d.word.map(lambda w: POOL[w][2])
    chosen = [w for w in FORCE if w in set(d.word)]
    chosen += [w for w in d[(d.prio == 1) & (d.f1 >= PRIORITY_MIN_F1)].sort_values("f1", ascending=False).word
               if w not in chosen]
    rest = d[~d.word.isin(chosen)].sort_values(["f1", "prio"], ascending=[False, True])
    for w in rest.word:
        if len(chosen) >= N_SELECT_MAX:
            break
        chosen.append(w)
    return chosen[:N_SELECT_MAX]


@torch.no_grad()
def predict_raw(model, seqs):
    """seqs: [(raw, W, H)] -> probs"""
    model.eval()
    X = np.stack([raw_to_features(r, W, H) for r, W, H in seqs])
    return F.softmax(model(torch.from_numpy(X)), -1).numpy()


def live_like(vids, k=5):
    """Каждое видео чужого жеста -> k вариантов того, что реально попадает в окно камеры."""
    out = []
    for v in vids:
        out.append((v["raw"], v["W"], v["H"]))
        out.append((static_sample(v), v["W"], v["H"]))
        for _ in range(k - 2):
            out.append((with_holds(partial(v["raw"])) if rng.random() < 0.5 else partial(v["raw"]), v["W"], v["H"]))
    return out


def calibrate(model, classes, o_cal):
    """Порог на каждое слово: наименьший, при котором ложные срабатывания на чужих жестах <= FP_TARGET."""
    P = predict_raw(model, live_like(o_cal))
    top, conf = P.argmax(1), P.max(1)
    thr = {}
    for i, c in enumerate(classes):
        if c == OTHER:
            continue
        thr[c] = THR_MAX
        for t in np.arange(THR_MIN, THR_MAX + 1e-9, 0.01):
            if ((top == i) & (conf >= t)).mean() <= FP_TARGET:
                thr[c] = round(float(t), 2)
                break
    return thr


def live_eval(model, classes, thr, te, o_te):
    """Оценка в условиях камеры: слова целиком и с паузами; чужие жесты в «живых» вариантах."""
    t = np.array([thr.get(c, 1.0) for c in classes])
    y = np.array([classes.index(v["text"]) for v in te])
    rows = []
    res = {}
    for name, f in [("full", lambda r: r), ("holds", with_holds)]:
        P = predict_raw(model, [(f(v["raw"]), v["W"], v["H"]) for v in te])
        res[name] = (P.argmax(1) == y) & (P.max(1) >= t[P.argmax(1)])
    Po = predict_raw(model, live_like(o_te))
    top, conf = Po.argmax(1), Po.max(1)
    fire = (top != classes.index(OTHER)) & (conf >= t[top])
    for i, c in enumerate(classes):
        if c == OTHER:
            continue
        m = y == i
        rows.append(dict(word=c, threshold=thr[c], recall_full=res["full"][m].mean(), recall_holds=res["holds"][m].mean(),
                         false_fires_per_1000=1000 * ((top == i) & fire).mean()))
    df = pd.DataFrame(rows)
    print("\n=== LIVE-LIKE EVAL (official test, per-word thresholds) ===")
    print(f"words recognised: full {res['full'].mean():.2f}, with pauses {res['holds'].mean():.2f}")
    print(f"unknown-gesture windows that fire a word: {fire.mean():.3f} (n={len(fire)})")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return df


def export_examples(model, classes, tr):
    """По одному показательному примеру на слово (самый уверенный train-пример) — анимация в словаре."""
    from features import assign_slots, to_pixels
    P = predict_raw(model, [(v["raw"], v["W"], v["H"]) for v in tr])
    out = {}
    for i, c in enumerate(classes):
        if c == OTHER:
            continue
        idx = [j for j, v in enumerate(tr) if v["text"] == c]
        v = tr[max(idx, key=lambda j: P[j, i])]
        slots = assign_slots(to_pixels(v["raw"], v["W"], v["H"]), v["W"])
        frames = []
        for fr in slots:
            frames.append([None if np.isnan(h[0, 0]) else
                           np.round(h[:, :2] / [v["W"], v["H"]], 3).tolist() for h in fr])
        out[c] = dict(W=v["W"], H=v["H"], frames=frames)
    p = OUT_MODEL / "examples.json"
    p.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("examples", p, f"({p.stat().st_size / 1024:.0f} KB)")


def export(model, classes, df, df_cv=None, thr=None):
    OUT_MODEL.mkdir(parents=True, exist_ok=True)
    m = WithSoftmax(model).eval()
    dummy = torch.zeros(1, SEQ_LEN, FEAT_DIM)
    path = OUT_MODEL / "sign_model.onnx"
    torch.onnx.export(m, dummy, path, input_names=["input"], output_names=["probs"],
                      dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17, dynamo=False)
    acc = dict(zip(df.word, df.acc))
    cv = dict(zip(df_cv.word, df_cv.acc)) if df_cv is not None else {}
    labels = [dict(id=c, ru=POOL[c][0] if c in POOL else OTHER_LABEL[0],
                   kk=POOL[c][1] if c in POOL else OTHER_LABEL[1],
                   is_other=c == OTHER, test_acc=round(float(acc.get(c, 0)), 3),
                   cv_acc=round(float(cv[c]), 3) if c in cv else None,
                   threshold=(thr or {}).get(c)) for c in classes]
    meta = dict(seq_len=SEQ_LEN, feat_dim=FEAT_DIM, labels=labels)
    (OUT_MODEL / "labels.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("exported", path, f"({path.stat().st_size / 1024:.0f} KB)")
    return path


def check_onnx(path, model, vids):
    import onnx
    onnx.checker.check_model(onnx.load(str(path)))
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnx check OK (onnxruntime не установлен — сравнение выходов пропущено)")
        return
    X = np.stack([raw_to_features(v["raw"], v["W"], v["H"]) for v in vids[:16]])
    a = ort.InferenceSession(str(path)).run(None, {"input": X})[0]
    with torch.no_grad():
        b = F.softmax(model.eval()(torch.from_numpy(X)), -1).numpy()
    print(f"onnx vs torch max diff: {np.abs(a - b).max():.2e}")


def save_fixture(vids):
    """Пара реальных последовательностей + эталонные признаки для проверки features.js."""
    fx = []
    for v in vids[:3]:
        raw = v["raw"].round(4)  # признаки считаем по тем же округлённым точкам, что в фикстуре
        fx.append(dict(W=v["W"], H=v["H"], text=v["text"],
                       raw=np.nan_to_num(raw, nan=-1).tolist(),
                       features=raw_to_features(raw, v["W"], v["H"]).round(6).tolist()))
    p = ROOT / "training" / "fixtures" / "features_fixture.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(fx), encoding="utf-8")


def main():
    t0 = time.time()
    videos = load_videos()
    present = {v["text"] for v in videos}
    pool = [w for w in POOL if w in present]
    missing = [w for w in POOL if w not in present]
    print(f"videos: {len(videos)}, pool words: {len(pool)}" + (f", missing: {missing}" if missing else ""))
    train_users = {v["user"] for v in videos if v["train"]}
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ---- фаза A: весь пул
    tr, te, o_tr, o_te, _ = build_split(videos, pool)
    print(f"phase A: train {len(tr)} + other {len(o_tr)}, test {len(te)} + other {len(o_te)}")
    model, classes = train_model(pool, tr, o_tr, log="A")
    dfA, cmA = evaluate(model, classes, te, o_te, train_users, "PHASE A — candidate pool")
    print_cm(cmA, classes, "phase A")
    dfA.to_csv(REPORTS / "phaseA_per_word.csv", index=False)
    pd.DataFrame(cmA, index=classes, columns=classes).to_csv(REPORTS / "phaseA_confusion.csv")

    dfCV, cmCV = cross_validate(videos, pool)
    print_cm(cmCV, list(pool) + [OTHER], "phase A CV")
    dfCV.to_csv(REPORTS / "phaseA_cv_per_word.csv", index=False)
    pd.DataFrame(cmCV, index=list(pool) + [OTHER], columns=list(pool) + [OTHER]).to_csv(REPORTS / "phaseA_cv_confusion.csv")

    chosen = select_words(dfCV)
    print(f"\nselected {len(chosen)} words: {chosen}")

    # ---- фаза B: финальная модель
    tr, te, o_tr, o_te, o_cal = build_split(videos, chosen)
    model, classes = train_model(chosen, tr, o_tr, epochs=90, log="B")
    dfB, cmB = evaluate(model, classes, te, o_te, train_users, "PHASE B — final model")
    print_cm(cmB, classes, "phase B")
    dfB.to_csv(REPORTS / "final_per_word.csv", index=False)
    pd.DataFrame(cmB, index=classes, columns=classes).to_csv(REPORTS / "final_confusion.csv")

    thr = calibrate(model, classes, o_cal)
    dfL = live_eval(model, classes, thr, te, o_te)
    dfL.to_csv(REPORTS / "final_live_eval.csv", index=False)

    path = export(model, classes, dfB, dfCV, thr)
    export_examples(model, classes, tr)
    check_onnx(path, model, te)
    save_fixture(te)
    print(f"\ntotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
