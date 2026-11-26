// Service worker: офлайн-кэш. Тяжёлые неизменные файлы (MediaPipe, wasm, сборка) — cache-first,
// страница и /model/ — network-first, чтобы обновления сайта и переобученная модель приходили сразу.
const CACHE = 'qazaqsign-v1';
const HEAVY = /\/(mediapipe|assets|icons)\//;

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== location.origin) return;
  if (HEAVY.test(new URL(req.url).pathname)) {
    e.respondWith(
      caches.match(req).then(
        (hit) =>
          hit ||
          fetch(req).then((res) => {
            if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
            return res;
          }),
      ),
    );
  } else {
    e.respondWith(
      fetch(req)
        .then((res) => {
          if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
          return res;
        })
        .catch(() => caches.match(req)),
    );
  }
});
