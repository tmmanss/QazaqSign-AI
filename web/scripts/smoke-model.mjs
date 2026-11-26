// Прогон ONNX-модели через onnxruntime-web (в Node) на фикстуре: признаки из features.js -> вероятности.
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web';
import { FEAT_DIM, SEQ_LEN, framesToFeatures } from '../src/features.js';

const fixtures = JSON.parse(readFileSync(new URL('../../training/fixtures/features_fixture.json', import.meta.url), 'utf8'));
const meta = JSON.parse(readFileSync(new URL('../public/model/labels.json', import.meta.url), 'utf8'));
const model = readFileSync(new URL('../public/model/sign_model.onnx', import.meta.url));
const session = await ort.InferenceSession.create(model);
for (const fx of fixtures) {
  const frames = fx.raw.map((fr) => fr.map((h) => (h[0][0] === -1 ? null : h)));
  const out = await session.run({ input: new ort.Tensor('float32', framesToFeatures(frames, fx.W, fx.H), [1, SEQ_LEN, FEAT_DIM]) });
  const p = Array.from(out.probs.data);
  const top = p.map((v, i) => [v, meta.labels[i].ru]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  console.log(fx.text, '->', top.map(([v, w]) => `${w} ${(v * 100).toFixed(2)}%`).join(', '));
}
