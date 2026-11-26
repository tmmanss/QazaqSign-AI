// Готовит статические файлы MediaPipe в public/, чтобы сайт не зависел от CDN:
//  - wasm-рантайм @mediapipe/tasks-vision -> public/mediapipe/wasm/
//  - модель hand_landmarker.task          -> public/mediapipe/hand_landmarker.task (скачивается один раз)
import { copyFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const wasmSrc = join(root, 'node_modules/@mediapipe/tasks-vision/wasm');
const wasmDst = join(root, 'public/mediapipe/wasm');
const taskDst = join(root, 'public/mediapipe/hand_landmarker.task');
const TASK_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

mkdirSync(wasmDst, { recursive: true });
for (const f of readdirSync(wasmSrc)) copyFileSync(join(wasmSrc, f), join(wasmDst, f));
console.log('mediapipe wasm copied');

if (!existsSync(taskDst)) {
  console.log('downloading hand_landmarker.task ...');
  const res = await fetch(TASK_URL);
  if (!res.ok) throw new Error(`download failed: ${res.status}`);
  writeFileSync(taskDst, Buffer.from(await res.arrayBuffer()));
  console.log('hand_landmarker.task saved');
}
