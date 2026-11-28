// Service worker: офлайн-кэш. Тяжёлые неизменные файлы (MediaPipe, wasm, сборка) — cache-first,
// страница и /model/ — network-first, чтобы обновления сайта и переобученная модель приходили сразу.
const CACHE = 'audarmashy-v1';
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

function fetchAndCache(req) {
  return fetch(req).then((res) => {
    // кэшируем только обычные успешные ответы (не редиректы, не opaque)
    if (res.ok && res.type === 'basic' && !res.redirected) {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
    }
    return res;
  });
}

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  // служебные запросы DevTools: перехватывать нельзя, иначе fetch() падает
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  const url = new URL(req.url);
  if (url.origin !== location.origin) return;

  if (HEAVY.test(url.pathname)) {
    e.respondWith(caches.match(req).then((hit) => hit || fetchAndCache(req)));
    return;
  }

  e.respondWith(
    fetchAndCache(req).catch(async () => {
      // офлайн: отдаём из кэша; для навигации — закэшированную главную
      const hit = (await caches.match(req)) || (req.mode === 'navigate' ? await caches.match('/') : undefined);
      return hit || Response.error();
    }),
  );
});
