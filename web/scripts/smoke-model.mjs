// Прогон ONNX-модели на фикстуре двумя движками — onnxruntime-web и запасным tinyonnx.js —
// с проверкой, что вероятности совпадают.
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web';
import { FEAT_DIM, SEQ_LEN, framesToFeatures } from '../src/features.js';
import { TinySession } from '../src/tinyonnx.js';

const fixtures = JSON.parse(readFileSync(new URL('../../training/fixtures/features_fixture.json', import.meta.url), 'utf8'));
const meta = JSON.parse(readFileSync(new URL('../public/model/labels.json', import.meta.url), 'utf8'));
const model = readFileSync(new URL('../public/model/sign_model.onnx', import.meta.url));
const session = await ort.InferenceSession.create(model);
const tiny = TinySession.fromBuffer(new Uint8Array(model));
let worst = 0;
for (const fx of fixtures) {
  const frames = fx.raw.map((fr) => fr.map((h) => (h[0][0] === -1 ? null : h)));
  const feats = framesToFeatures(frames, fx.W, fx.H);
  const a = Array.from((await session.run({ input: new ort.Tensor('float32', feats, [1, SEQ_LEN, FEAT_DIM]) })).probs.data);
  const t0 = performance.now();
  const b = Array.from(tiny.run({ input: { data: feats, dims: [1, SEQ_LEN, FEAT_DIM] } }).probs.data);
  const ms = performance.now() - t0;
  const diff = Math.max(...a.map((v, i) => Math.abs(v - b[i])));
  worst = Math.max(worst, diff);
  const top = a.map((v, i) => [v, meta.labels[i].ru]).sort((x, y) => y[0] - x[0]).slice(0, 3);
  console.log(`${fx.text} -> ${top.map(([v, w]) => `${w} ${(v * 100).toFixed(2)}%`).join(', ')} | tiny diff ${diff.toExponential(1)}, ${ms.toFixed(1)} ms`);
}
if (worst > 1e-4) {
  console.error('FAIL: tinyonnx расходится с onnxruntime');
  process.exit(1);
}
console.log('OK: tinyonnx совпадает с onnxruntime');
