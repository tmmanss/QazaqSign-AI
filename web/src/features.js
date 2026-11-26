// Порт training/features.py 1-в-1 — любые изменения делать синхронно в обоих файлах.
// Проверка совпадения с Python: npm run test:features
//
// Кадр (frame) = [hand0, hand1], где hand = массив 21 точки [x, y, z] (нормированные
// координаты MediaPipe) или null, если руки нет. Порядок рук в кадре не важен.

export const SEQ_LEN = 32;
export const N_PTS = 21;
export const FEAT_PER_HAND = N_PTS * 3 + 2 + 1; // 66
export const FEAT_DIM = FEAT_PER_HAND * 2; // 132
const SLOT_DEFAULT_X = [0.35, 0.65];

function toPixels(hand, W, H) {
  return hand.map(([x, y, z]) => [x * W, y * H, z * W]);
}

/** frames -> canonical slots: [ [slot0|null, slot1|null], ... ] in pixels */
export function assignSlots(frames, W, H) {
  const lastX = [SLOT_DEFAULT_X[0] * W, SLOT_DEFAULT_X[1] * W];
  return frames.map((fr) => {
    const hands = fr.filter(Boolean).map((h) => toPixels(h, W, H));
    const out = [null, null];
    if (hands.length >= 2) {
      const [a, b] = hands.slice(0, 2).sort((p, q) => p[0][0] - q[0][0]);
      out[0] = a;
      out[1] = b;
      lastX[0] = a[0][0];
      lastX[1] = b[0][0];
    } else if (hands.length === 1) {
      const h = hands[0];
      const x = h[0][0];
      const s = Math.abs(x - lastX[0]) <= Math.abs(x - lastX[1]) ? 0 : 1;
      out[s] = h;
      lastX[s] = x;
    }
    return out;
  });
}

// numpy.round = банковское округление (half to even)
function roundHalfEven(v) {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function resampleIdx(T, n = SEQ_LEN) {
  const idx = new Array(n);
  for (let i = 0; i < n; i++) idx[i] = roundHalfEven(n === 1 ? 0 : (i * (T - 1)) / (n - 1));
  return idx;
}

function palmSize(h) {
  return Math.hypot(h[9][0] - h[0][0], h[9][1] - h[0][1]);
}

/** slots: [SEQ_LEN][2] (pixels, null = missing) -> Float32Array(SEQ_LEN * FEAT_DIM) */
export function seqFeatures(slots) {
  const T = slots.length;
  const out = new Float32Array(T * FEAT_DIM);
  let sizeSum = 0, cx = 0, cy = 0, n = 0;
  for (const fr of slots) {
    for (const h of fr) {
      if (!h) continue;
      sizeSum += palmSize(h);
      cx += h[0][0];
      cy += h[0][1];
      n++;
    }
  }
  if (!n) return out;
  const meanSize = Math.max(sizeSum / n, 1e-6);
  cx /= n;
  cy /= n;
  for (let t = 0; t < T; t++) {
    for (let s = 0; s < 2; s++) {
      const h = slots[t][s];
      if (!h) continue;
      const size = Math.max(palmSize(h), 1e-6);
      const base = t * FEAT_DIM + s * FEAT_PER_HAND;
      const [wx, wy, wz] = h[0];
      for (let p = 0; p < N_PTS; p++) {
        out[base + p * 3] = (h[p][0] - wx) / size;
        out[base + p * 3 + 1] = (h[p][1] - wy) / size;
        out[base + p * 3 + 2] = (h[p][2] - wz) / size;
      }
      out[base + 63] = (wx - cx) / meanSize;
      out[base + 64] = (wy - cy) / meanSize;
      out[base + 65] = 1;
    }
  }
  return out;
}

/** Полный пайплайн: кадры окна -> признаки [SEQ_LEN * FEAT_DIM] */
export function framesToFeatures(frames, W, H) {
  const slots = assignSlots(frames, W, H);
  return seqFeatures(resampleIdx(slots.length).map((i) => slots[i]));
}
