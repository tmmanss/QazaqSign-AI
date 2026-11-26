// Анимация жеста для словаря: скелет рук носителя из Slovo (model/examples.json).
export const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const SLOT_COLORS = ['#2dd4bf', '#fbbf24'];
const FPS = 30;
const PAUSE_MS = 700;

/**
 * Проигрывает пример по кругу. example = { W, H, frames: [[hand|null, hand|null], ...] },
 * hand = 21 точка [x, y] в долях кадра. Возвращает объект { setMirror(bool), stop() }.
 */
export function playDemo(canvas, example, mirror = false) {
  const ctx = canvas.getContext('2d');
  const { W, H, frames } = example;
  // рамка по всем точкам жеста, чтобы руки были крупными
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const fr of frames)
    for (const h of fr)
      if (h)
        for (const [x, y] of h) {
          x0 = Math.min(x0, x * W); x1 = Math.max(x1, x * W);
          y0 = Math.min(y0, y * H); y1 = Math.max(y1, y * H);
        }
  const pad = 0.15 * Math.max(x1 - x0, y1 - y0);
  x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;

  let raf = 0;
  let t0 = performance.now();
  const total = (frames.length / FPS) * 1000 + PAUSE_MS;

  const draw = (now) => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.round(canvas.clientWidth * dpr);
    const ch = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw;
      canvas.height = ch;
    }
    const s = Math.min(cw / (x1 - x0), ch / (y1 - y0));
    const ox = (cw - (x1 - x0) * s) / 2;
    const oy = (ch - (y1 - y0) * s) / 2;
    const P = ([x, y]) => {
      const px = ox + (x * W - x0) * s;
      return [mirror ? cw - px : px, oy + (y * H - y0) * s];
    };

    const el = (now - t0) % total;
    const i = Math.min(frames.length - 1, Math.floor((el / 1000) * FPS));
    ctx.clearRect(0, 0, cw, ch);

    // прогресс внизу
    ctx.fillStyle = 'rgba(255,255,255,0.12)';
    ctx.fillRect(0, ch - 3 * dpr, cw, 3 * dpr);
    ctx.fillStyle = '#2dd4bf';
    ctx.fillRect(0, ch - 3 * dpr, (cw * (i + 1)) / frames.length, 3 * dpr);

    const lw = 2.5 * dpr;
    frames[i].forEach((h, slot) => {
      if (!h) return;
      ctx.strokeStyle = SLOT_COLORS[slot];
      ctx.lineWidth = lw;
      ctx.lineCap = 'round';
      ctx.beginPath();
      for (const [a, b] of HAND_CONNECTIONS) {
        const [ax, ay] = P(h[a]);
        const [bx, by] = P(h[b]);
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
      }
      ctx.stroke();
      ctx.fillStyle = '#fff';
      for (const p of h) {
        const [x, y] = P(p);
        ctx.beginPath();
        ctx.arc(x, y, lw * 1.2, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    raf = requestAnimationFrame(draw);
  };
  raf = requestAnimationFrame(draw);

  return {
    setMirror(m) {
      mirror = m;
    },
    stop() {
      cancelAnimationFrame(raf);
    },
  };
}
