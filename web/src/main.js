import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { HAND_CONNECTIONS, playDemo } from './demo.js';
import { INFER_EVERY_MS, Recognizer } from './recognizer.js';
import './style.css';

const BASE = import.meta.env.BASE_URL;
const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('overlay');
const ctx = canvas.getContext('2d');


const state = {
  landmarker: null,
  rec: null,
  facing: 'user',
  stream: null,
  lastVideoTime: -1,
  hands: [],
  phrases: [[]], // массив фраз, фраза = массив индексов слов
  debug: false,
  fps: 0,
  inferMs: 0,
};

// ---------------------------------------------------------------- хранилище (может быть недоступно)
const store = {
  get(k, d) {
    try {
      const v = localStorage.getItem(k);
      return v === null ? d : JSON.parse(v);
    } catch {
      return d;
    }
  },
  set(k, v) {
    try {
      localStorage.setItem(k, JSON.stringify(v));
    } catch {}
  },
};

// ---------------------------------------------------------------- загрузка моделей
async function createLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(`${BASE}mediapipe/wasm`);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: `${BASE}mediapipe/hand_landmarker.task`, delegate },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await HandLandmarker.createFromOptions(fileset, opts('GPU'));
  } catch (e) {
    console.warn('GPU delegate недоступен, переключаюсь на CPU', e);
    return HandLandmarker.createFromOptions(fileset, opts('CPU'));
  }
}

// ---------------------------------------------------------------- камера
async function startCamera() {
  state.stream?.getTracks().forEach((t) => t.stop());
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: state.facing, width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  state.stream = stream;
  video.srcObject = stream;
  video.classList.toggle('mirror', state.facing === 'user');
  await video.play();
  state.lastVideoTime = -1;
  state.rec?.reset();
}

// ---------------------------------------------------------------- отрисовка рук
function resizeCanvas() {
  const r = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
}

function drawHands(hands) {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !hands.length) return;
  // видео показано как object-fit: cover — повторяем то же преобразование
  const s = Math.max(canvas.width / vw, canvas.height / vh);
  const ox = (canvas.width - vw * s) / 2;
  const oy = (canvas.height - vh * s) / 2;
  const mirror = state.facing === 'user';
  const px = (p) => {
    const x = ox + p.x * vw * s;
    return [mirror ? canvas.width - x : x, oy + p.y * vh * s];
  };
  const lw = Math.max(2, canvas.width / 260);
  hands.forEach((hand, hi) => {
    const color = hi === 0 ? '#2dd4bf' : '#fbbf24';
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.lineCap = 'round';
    ctx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      const [x1, y1] = px(hand[a]);
      const [x2, y2] = px(hand[b]);
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();
    ctx.fillStyle = '#fff';
    for (const p of hand) {
      const [x, y] = px(p);
      ctx.beginPath();
      ctx.arc(x, y, lw * 1.3, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

// ---------------------------------------------------------------- цикл детекции
let frameCount = 0;
let fpsT = performance.now();
function detectLoop() {
  const now = performance.now();
  if (state.landmarker && video.readyState >= 2 && video.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = video.currentTime;
    const result = state.landmarker.detectForVideo(video, now);
    state.hands = result.landmarks || [];
    state.rec.pushFrame(
      now,
      state.hands.slice(0, 2).map((h) => h.map((p) => [p.x, p.y, p.z])),
    );
    drawHands(state.hands);
    setHandStatus(state.hands.length);
    frameCount++;
  }
  if (now - fpsT > 1000) {
    state.fps = (frameCount * 1000) / (now - fpsT);
    frameCount = 0;
    fpsT = now;
  }
  requestAnimationFrame(detectLoop);
}

let inferBusy = false;
async function inferTick() {
  if (inferBusy || !state.rec || !video.videoWidth) return;
  inferBusy = true;
  try {
    const t0 = performance.now();
    const r = await state.rec.step(performance.now(), video.videoWidth, video.videoHeight);
    if (r.probs) state.inferMs = performance.now() - t0;
    if (r.emitted !== null) addWord(r.emitted);
    if (r.phraseEnd) endPhrase();
    $('streak-bar').style.width = `${Math.min(1, r.streak / state.rec.needStreak) * 100}%`;
    if (state.debug) renderDebug(r);
  } catch (e) {
    console.error(e);
  } finally {
    inferBusy = false;
  }
}

// ---------------------------------------------------------------- субтитры
function capitalize(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

function phraseText(lang) {
  const labels = state.rec?.labels || [];
  return state.phrases
    .filter((p) => p.length)
    .map((p) => capitalize(p.map((i) => labels[i][lang]).join(' ')))
    .join('. ');
}

function renderSubs(flash = false) {
  const ru = phraseText('ru');
  const ruEl = $('sub-ru');
  ruEl.classList.toggle('placeholder', !ru);
  ruEl.textContent = ru || 'Покажите жест — перевод появится здесь';
  $('sub-kk').textContent = phraseText('kk');
  if (flash) {
    ruEl.classList.remove('flash');
    void ruEl.offsetWidth;
    ruEl.classList.add('flash');
  }
}

function addWord(idx) {
  state.phrases[state.phrases.length - 1].push(idx);
  if (navigator.vibrate) navigator.vibrate(30);
  renderSubs(true);
}

function endPhrase() {
  if (state.phrases[state.phrases.length - 1].length) state.phrases.push([]);
  // показываем только последние 3 фразы
  while (state.phrases.filter((p) => p.length).length > 3) state.phrases.shift();
}

function setHandStatus(n) {
  const el = $('status');
  el.classList.toggle('on', n > 0);
  $('status-text').textContent = n ? (n === 1 ? 'Рука в кадре' : 'Две руки в кадре') : 'Нет рук в кадре';
}

// ---------------------------------------------------------------- озвучка
let voices = [];
function loadVoices() {
  voices = window.speechSynthesis?.getVoices() || [];
  const kk = voices.find((v) => v.lang.toLowerCase().startsWith('kk'));
  $('voice-info').textContent = kk
    ? `Озвучка: русский и казахский (${kk.name}).`
    : 'Озвучка: русский. Казахский голос на этом устройстве не найден.';
}

function speak() {
  const synth = window.speechSynthesis;
  if (!synth) return;
  synth.cancel();
  const ru = phraseText('ru');
  if (!ru) return;
  const say = (text, lang) => {
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang;
    const v = voices.find((x) => x.lang.replace('_', '-').toLowerCase().startsWith(lang.slice(0, 2).toLowerCase()));
    if (v) u.voice = v;
    synth.speak(u);
  };
  say(ru, 'ru-RU');
  if (voices.some((v) => v.lang.toLowerCase().startsWith('kk'))) say(phraseText('kk'), 'kk-KZ');
}

// ---------------------------------------------------------------- словарь и отладка
function quality(l) {
  const a = l.cv_acc ?? l.test_acc;
  if (a >= 0.8) return ['good', 'надёжно'];
  if (a >= 0.6) return ['mid', 'средне'];
  return ['bad', 'слабо'];
}

let examples = {};
let demo = null;
let demoMirror = store.get('demoMirror', false);

function renderDict(labels) {
  const items = labels
    .filter((l) => !l.is_other)
    .map((l) => ({ l, q: quality(l), a: l.cv_acc ?? l.test_acc }))
    .sort((x, y) => y.a - x.a);
  $('dict-list').innerHTML = items
    .map(
      ({ l, q, a }) => `<li data-id="${l.id}">
        <button class="word-row">
          <span class="play" aria-hidden="true">▶</span>
          <div class="word-names"><div class="w-ru">${l.ru}</div><div class="w-kk">${l.kk}</div></div>
          <span class="badge ${q[0]}" title="точность ${Math.round(a * 100)}%">${q[1]} · ${Math.round(a * 100)}%</span>
        </button>
      </li>`,
    )
    .join('');
}

function closeDemo() {
  demo?.stop();
  demo = null;
  document.querySelectorAll('#dict-list .demo').forEach((el) => el.remove());
  document.querySelectorAll('#dict-list li.open').forEach((el) => el.classList.remove('open'));
}

function openDemo(li) {
  const wasOpen = li.classList.contains('open');
  closeDemo();
  if (wasOpen) return;
  const ex = examples[li.dataset.id];
  if (!ex) return;
  li.classList.add('open');
  const box = document.createElement('div');
  box.className = 'demo';
  box.innerHTML = `<canvas></canvas>
    <div class="demo-bar">
      <span>Так показывал носитель из датасета Slovo</span>
      <button class="btn ghost small mirror-btn">${demoMirror ? 'Как в оригинале' : 'Отзеркалить'}</button>
    </div>`;
  li.append(box);
  demo = playDemo(box.querySelector('canvas'), ex, demoMirror);
  box.querySelector('.mirror-btn').onclick = (e) => {
    demoMirror = !demoMirror;
    store.set('demoMirror', demoMirror);
    demo.setMirror(demoMirror);
    e.target.textContent = demoMirror ? 'Как в оригинале' : 'Отзеркалить';
  };
}

$('dict-list').addEventListener('click', (e) => {
  const row = e.target.closest('.word-row');
  if (row) openDemo(row.parentElement);
});

function renderDebug(r) {
  const labels = state.rec.labels;
  if (r.top3.length) {
    $('top3').innerHTML = r.top3
      .map(({ i, p }) => {
        const name = labels[i].is_other ? 'нет слова' : labels[i].ru;
        const need = labels[i].is_other ? null : state.rec.thresholdFor(i);
        const pass = need !== null && p >= need;
        const hint = need !== null ? ` <small>/ ${(need * 100).toFixed(0)}</small>` : '';
        return `<div class="bar ${pass ? 'pass' : ''}"><span>${name}</span><b>${(p * 100).toFixed(0)}%${hint}</b><i style="width:${p * 100}%"></i></div>`;
      })
      .join('');
  } else {
    $('top3').innerHTML = '<div class="muted">ожидание жеста…</div>';
  }
  $('stats').textContent = `${state.fps.toFixed(0)} к/с · окно ${state.rec.frames.length} кадров · модель ${state.inferMs.toFixed(1)} мс (${state.rec.backend}) · ${video.videoWidth}×${video.videoHeight}`;
}

function setDebug(on) {
  state.debug = on;
  $('debug').hidden = !on;
  $('btn-debug').classList.toggle('active', on);
  store.set('debug', on);
}

function bindSliders() {
  const thr = $('thr');
  const st = $('streak');
  thr.value = state.rec.threshold = store.get('minThreshold', 0.5);
  st.value = state.rec.needStreak = store.get('confirmCount', 2);
  const show = () => {
    $('thr-val').textContent = Number(thr.value).toFixed(2);
    $('streak-val').textContent = st.value;
  };
  thr.oninput = () => {
    state.rec.threshold = Number(thr.value);
    store.set('minThreshold', state.rec.threshold);
    show();
  };
  st.oninput = () => {
    state.rec.needStreak = Number(st.value);
    store.set('confirmCount', state.rec.needStreak);
    show();
  };
  show();
}

// ---------------------------------------------------------------- старт
async function start() {
  const btn = $('btn-start');
  const msg = $('start-msg');
  btn.disabled = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Камера недоступна: откройте сайт по HTTPS.');
    msg.textContent = 'Загрузка моделей…';
    // по очереди, а не параллельно: два wasm-модуля сразу могут не влезть в память телефона
    await startCamera();
    state.landmarker ||= await createLandmarker();
    msg.textContent = 'Загрузка модели жестов…';
    state.rec = await Recognizer.load(BASE);
    const rec = state.rec;
    bindSliders();
    renderDict(rec.labels);
    $('start').hidden = true;
    requestAnimationFrame(detectLoop);
    setInterval(inferTick, INFER_EVERY_MS);
  } catch (e) {
    console.error(e);
    msg.textContent =
      e.name === 'NotAllowedError'
        ? 'Нет доступа к камере. Разрешите его в настройках браузера и попробуйте снова.'
        : `Ошибка: ${e.message || e}`;
    btn.disabled = false;
  }
}

$('btn-start').onclick = start;
$('btn-flip').onclick = async () => {
  state.facing = state.facing === 'user' ? 'environment' : 'user';
  try {
    await startCamera();
  } catch (e) {
    console.error(e);
  }
};
$('btn-clear').onclick = () => {
  state.phrases = [[]];
  if (state.rec) state.rec.lastWordIdx = -1;
  renderSubs();
};
$('btn-speak').onclick = speak;
$('btn-dict').onclick = $('btn-start-dict').onclick = () => ($('dict').hidden = false);
fetch(`${BASE}model/labels.json`)
  .then((r) => r.json())
  .then((m) => renderDict(m.labels))
  .catch(console.warn);
fetch(`${BASE}model/examples.json`)
  .then((r) => (r.ok ? r.json() : {}))
  .then((ex) => (examples = ex))
  .catch(console.warn);
const closeDict = () => {
  closeDemo();
  $('dict').hidden = true;
};
$('btn-dict-close').onclick = closeDict;
$('dict').onclick = (e) => {
  if (e.target.id === 'dict') closeDict();
};
$('btn-debug').onclick = () => setDebug(!state.debug);

setDebug(store.get('debug', false));
if (window.speechSynthesis) {
  loadVoices();
  window.speechSynthesis.onvoiceschanged = loadVoices;
}
if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register(`${BASE}sw.js`).catch(console.warn);
}
