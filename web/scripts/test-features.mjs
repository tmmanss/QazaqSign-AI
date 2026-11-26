// Сверяет web/src/features.js с эталоном из Python (training/fixtures/features_fixture.json).
import { readFileSync } from 'node:fs';
import { framesToFeatures } from '../src/features.js';

const url = new URL('../../training/fixtures/features_fixture.json', import.meta.url);
const fixtures = JSON.parse(readFileSync(url, 'utf8'));
let worst = 0;
for (const fx of fixtures) {
  // в фикстуре отсутствующая рука закодирована как -1
  const frames = fx.raw.map((fr) => fr.map((h) => (h[0][0] === -1 ? null : h)));
  const got = framesToFeatures(frames, fx.W, fx.H);
  const exp = fx.features.flat();
  let maxDiff = 0;
  exp.forEach((v, i) => (maxDiff = Math.max(maxDiff, Math.abs(v - got[i]))));
  worst = Math.max(worst, maxDiff);
  console.log(`${fx.text}: frames=${frames.length}, max diff=${maxDiff.toExponential(2)}`);
}
if (worst > 1e-3) {
  console.error('FAIL: features.js расходится с features.py');
  process.exit(1);
}
console.log('OK: features.js совпадает с features.py');
