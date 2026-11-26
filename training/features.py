"""Препроцессинг точек рук. ВАЖНО: web/src/features.js повторяет этот код 1-в-1.

Вход: последовательность кадров raw [T, 2, 21, 3] (NaN = руки нет) в нормированных
координатах MediaPipe (x,y ∈ 0..1) + размеры кадра W, H.

1. Перевод в пиксели: x*W, y*H, z*W (убирает искажение пропорций кадра).
2. Раскладка рук по слотам (у Slovo нет пометки левая/правая):
   - две руки → слот 0 = левее в кадре, слот 1 = правее;
   - одна рука → слот, чья последняя известная x запястья ближе
     (начальные значения 0.35·W и 0.65·W).
3. Ресемплинг к SEQ_LEN кадрам (ближайший кадр по равномерной сетке).
4. Признаки на кадр и слот (FEAT_PER_HAND = 66):
   - 21×3 точки относительно запястья, делённые на размер ладони
     (расстояние запястье→основание среднего пальца, по x,y);
   - x,y запястья относительно среднего положения всех запястий в окне,
     делённые на средний размер ладони в окне (траектория движения);
   - флаг присутствия руки (1/0).
   Отсутствующая рука = нули. Итог: [SEQ_LEN, 132].
"""
import numpy as np

SEQ_LEN = 32
N_PTS = 21
FEAT_PER_HAND = N_PTS * 3 + 2 + 1
FEAT_DIM = FEAT_PER_HAND * 2
SLOT_DEFAULT_X = (0.35, 0.65)


def to_pixels(raw, W, H):
    p = raw.astype(np.float32).copy()
    p[..., 0] *= W
    p[..., 1] *= H
    p[..., 2] *= W
    return p


def assign_slots(px, W):
    """px: [T, 2, 21, 3] in pixels, slots arbitrary -> canonical slots."""
    T = px.shape[0]
    out = np.full_like(px, np.nan)
    last_x = [SLOT_DEFAULT_X[0] * W, SLOT_DEFAULT_X[1] * W]
    for t in range(T):
        hands = [px[t, s] for s in range(2) if not np.isnan(px[t, s, 0, 0])]
        if len(hands) >= 2:
            hands.sort(key=lambda h: h[0, 0])
            out[t, 0], out[t, 1] = hands[0], hands[1]
            last_x = [hands[0][0, 0], hands[1][0, 0]]
        elif len(hands) == 1:
            h = hands[0]
            x = h[0, 0]
            s = 0 if abs(x - last_x[0]) <= abs(x - last_x[1]) else 1
            out[t, s] = h
            last_x[s] = x
    return out


def resample_idx(T, n=SEQ_LEN):
    return np.round(np.linspace(0, T - 1, n)).astype(np.int64)


def palm_size(h):
    """h: [..., 21, 3] -> [...]"""
    return np.linalg.norm(h[..., 9, :2] - h[..., 0, :2], axis=-1)


def seq_features(px_slots):
    """px_slots: [SEQ_LEN, 2, 21, 3] canonical, pixels, NaN = missing -> [SEQ_LEN, FEAT_DIM]."""
    T = px_slots.shape[0]
    feats = np.zeros((T, 2, FEAT_PER_HAND), np.float32)
    present = ~np.isnan(px_slots[:, :, 0, 0])  # [T, 2]
    if not present.any():
        return feats.reshape(T, FEAT_DIM)
    sizes = palm_size(px_slots)  # [T, 2]
    mean_size = max(float(np.mean(sizes[present])), 1e-6)
    center = px_slots[:, :, 0, :2][present].mean(axis=0)  # [2]
    for t in range(T):
        for s in range(2):
            if not present[t, s]:
                continue
            h = px_slots[t, s]
            size = max(float(sizes[t, s]), 1e-6)
            local = (h - h[0]) / size
            glob = (h[0, :2] - center) / mean_size
            feats[t, s, :63] = local.reshape(-1)
            feats[t, s, 63:65] = glob
            feats[t, s, 65] = 1.0
    return feats.reshape(T, FEAT_DIM)


def raw_to_features(raw, W, H):
    """Full pipeline for an unaugmented sequence: raw [T,2,21,3] -> [SEQ_LEN, FEAT_DIM]."""
    px = assign_slots(to_pixels(raw, W, H), W)
    return seq_features(px[resample_idx(len(px))])
