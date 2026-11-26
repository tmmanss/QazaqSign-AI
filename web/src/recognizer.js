// Потоковое распознавание: буфер кадров -> признаки -> ONNX -> решение «засчитать слово».
import * as ort from 'onnxruntime-web/wasm';
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { FEAT_DIM, SEQ_LEN, framesToFeatures } from './features.js';
import { TinySession } from './tinyonnx.js';

ort.env.wasm.wasmPaths = { wasm: wasmUrl };
ort.env.wasm.numThreads = 1; // многопоточность требует COOP/COEP; модель крошечная, одного потока хватает

export const WINDOW_MS = 1500; // длина окна (медианный жест в Slovo ≈ 45 кадров ≈ 1.5 с)
export const INFER_EVERY_MS = 200;
const MIN_FRAMES = 8;
const MIN_SPAN_MS = 500;
export const PAUSE_MS = 1000; // столько без рук = конец фразы

export class Recognizer {
  constructor(runModel, backend, meta) {
    this.runModel = runModel; // async (Float32Array) -> number[]
    this.backend = backend;
    this.labels = meta.labels;
    this.otherIdx = this.labels.findIndex((l) => l.is_other);
    this.threshold = 0.6;
    this.needStreak = 3;
    this.frames = []; // { t, hands: [hand|null, hand|null] }
    this.reset();
  }

  static async load(base) {
    const meta = await (await fetch(`${base}model/labels.json`)).json();
    if (meta.seq_len !== SEQ_LEN || meta.feat_dim !== FEAT_DIM) throw new Error('labels.json не совпадает с features.js');
    const url = `${base}model/sign_model.onnx`;
    const dims = [1, SEQ_LEN, FEAT_DIM];
    try {
      const session = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
      const run = async (feats) => Array.from((await session.run({ input: new ort.Tensor('float32', feats, dims) })).probs.data);
      return new Recognizer(run, 'onnxruntime-web', meta);
    } catch (e) {
      // на телефонах с малым лимитом памяти WebAssembly onnxruntime может не запуститься
      console.warn('onnxruntime-web недоступен, использую JS-движок:', e);
      const tiny = await TinySession.create(url);
      const run = async (feats) => Array.from(tiny.run({ input: { data: feats, dims } }).probs.data);
      return new Recognizer(run, 'js', meta);
    }
  }

  reset() {
    this.frames = [];
    this.streakIdx = -1;
    this.streak = 0;
    this.lastWordIdx = -1;
    this.lastHandsT = 0;
    this.phraseOpen = false;
  }

  /** hands: массив до 2 рук (21 точка [x,y,z]); пустой массив = рук нет */
  pushFrame(t, hands) {
    if (hands.length) {
      // как в Slovo: кадры без рук в последовательность не попадают
      this.frames.push({ t, hands: [hands[0] || null, hands[1] || null] });
      this.lastHandsT = t;
    }
    while (this.frames.length && this.frames[0].t < t - WINDOW_MS) this.frames.shift();
  }

  /** Возвращает { probs, top3, emitted: labelIdx|null, phraseEnd: bool, streak } */
  async step(t, W, H) {
    const res = { probs: null, top3: [], emitted: null, phraseEnd: false, streak: 0 };

    // пауза без рук -> конец фразы
    if (t - this.lastHandsT > PAUSE_MS) {
      if (this.phraseOpen) res.phraseEnd = true;
      this.phraseOpen = false;
      this.lastWordIdx = -1;
      this.streak = 0;
      this.frames = [];
      return res;
    }

    const f = this.frames;
    if (f.length < MIN_FRAMES || f[f.length - 1].t - f[0].t < MIN_SPAN_MS) return res;

    const feats = framesToFeatures(f.map((x) => x.hands), W, H);
    const probs = await this.runModel(feats);
    res.probs = probs;
    res.top3 = probs
      .map((p, i) => ({ i, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 3);

    const best = res.top3[0];
    if (best.i !== this.otherIdx && best.p >= this.threshold) {
      this.streak = best.i === this.streakIdx ? this.streak + 1 : 1;
      this.streakIdx = best.i;
    } else {
      this.streak = 0;
      this.streakIdx = -1;
    }
    res.streak = this.streak;

    if (this.streak >= this.needStreak && best.i !== this.lastWordIdx) {
      res.emitted = best.i;
      this.lastWordIdx = best.i;
      this.phraseOpen = true;
      this.streak = 0;
      this.frames = []; // следующее слово собираем с чистого окна
    }
    return res;
  }
}
