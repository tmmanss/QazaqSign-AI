// Мини-движок ONNX на чистом JS — запасной вариант, если onnxruntime-web не запускается
// (например, «RangeError: Out of memory» на телефонах с малым лимитом памяти для WebAssembly).
// Читает .onnx напрямую (минимальный разбор protobuf) и выполняет только операции нашей
// модели: Transpose, Conv (1D), Relu, Add, ReduceMean, ReduceMax, Concat, Gemm, Softmax.
// Неизвестная операция -> ошибка при загрузке, а не тихо неверный результат.

// ---------------------------------------------------------------- protobuf
class Reader {
  constructor(buf, start = 0, end = buf.length) {
    this.b = buf;
    this.p = start;
    this.end = end;
  }
  varint() {
    let r = 0n;
    let shift = 0n;
    for (;;) {
      const x = this.b[this.p++];
      r |= BigInt(x & 0x7f) << shift;
      if (x < 0x80) return r;
      shift += 7n;
    }
  }
  int() {
    return Number(BigInt.asIntN(64, this.varint()));
  }
  bytes() {
    const n = Number(this.varint());
    const s = this.p;
    this.p += n;
    return [s, this.p];
  }
  /** Перебор полей сообщения: cb(fieldNo, wireType) должен прочитать значение или вернуть false (пропустить) */
  fields(cb) {
    while (this.p < this.end) {
      const key = Number(this.varint());
      const no = key >>> 3;
      const wt = key & 7;
      const save = this.p;
      if (cb(no, wt) === false) {
        this.p = save;
        this.skip(wt);
      }
    }
  }
  skip(wt) {
    if (wt === 0) this.varint();
    else if (wt === 1) this.p += 8;
    else if (wt === 2) this.bytes();
    else if (wt === 5) this.p += 4;
    else throw new Error(`protobuf: wire type ${wt}`);
  }
  sub() {
    const [s, e] = this.bytes();
    return new Reader(this.b, s, e);
  }
  str() {
    const [s, e] = this.bytes();
    return new TextDecoder().decode(this.b.subarray(s, e));
  }
  ints(wt, out) {
    if (wt === 2) {
      const r = this.sub();
      while (r.p < r.end) out.push(r.int());
    } else out.push(this.int());
  }
}

function f32(buf, s, e) {
  const copy = buf.slice(s, e); // выравнивание для Float32Array
  return new Float32Array(copy.buffer, copy.byteOffset, (e - s) / 4);
}

function parseTensor(r) {
  const t = { dims: [], name: '', dataType: 0, data: null };
  const floats = [];
  r.fields((no, wt) => {
    if (no === 1) r.ints(wt, t.dims);
    else if (no === 2) t.dataType = r.int();
    else if (no === 8) t.name = r.str();
    else if (no === 9) {
      const [s, e] = r.bytes();
      t.data = f32(r.b, s, e);
    } else if (no === 4) {
      if (wt === 2) {
        const [s, e] = r.bytes();
        floats.push(...f32(r.b, s, e));
      } else {
        floats.push(new DataView(r.b.buffer, r.b.byteOffset + r.p, 4).getFloat32(0, true));
        r.p += 4;
      }
    } else return false;
  });
  if (t.dataType !== 1) throw new Error(`tinyonnx: тензор ${t.name} не float32`);
  if (!t.data) t.data = new Float32Array(floats);
  return t;
}

function parseAttr(r) {
  const a = { name: '', f: undefined, i: undefined, ints: [] };
  r.fields((no, wt) => {
    if (no === 1) a.name = r.str();
    else if (no === 2) {
      a.f = new DataView(r.b.buffer, r.b.byteOffset + r.p, 4).getFloat32(0, true);
      r.p += 4;
    } else if (no === 3) a.i = r.int();
    else if (no === 8) r.ints(wt, a.ints);
    else return false;
  });
  return a;
}

function parseNode(r) {
  const n = { inputs: [], outputs: [], op: '', attrs: {} };
  r.fields((no) => {
    if (no === 1) n.inputs.push(r.str());
    else if (no === 2) n.outputs.push(r.str());
    else if (no === 4) n.op = r.str();
    else if (no === 5) {
      const a = parseAttr(r.sub());
      n.attrs[a.name] = a.ints.length ? a.ints : a.i !== undefined ? a.i : a.f;
    } else return false;
  });
  return n;
}

function parseModel(buf) {
  const g = { nodes: [], init: {}, inputs: [], outputs: [] };
  const m = new Reader(buf);
  m.fields((no) => {
    if (no !== 7) return false;
    const r = m.sub();
    r.fields((gno) => {
      if (gno === 1) g.nodes.push(parseNode(r.sub()));
      else if (gno === 5) {
        const t = parseTensor(r.sub());
        g.init[t.name] = t;
      } else if (gno === 11 || gno === 12) {
        const v = r.sub();
        let name = '';
        v.fields((vno) => (vno === 1 ? ((name = v.str()), true) : false));
        (gno === 11 ? g.inputs : g.outputs).push(name);
      } else return false;
    });
  });
  g.inputs = g.inputs.filter((n) => !g.init[n]);
  return g;
}

// ---------------------------------------------------------------- операции
const size = (dims) => dims.reduce((a, b) => a * b, 1);
const norm = (axis, rank) => (axis < 0 ? axis + rank : axis);

const OPS = {
  Transpose([x], { perm }) {
    const rank = x.dims.length;
    perm = perm || [...Array(rank).keys()].reverse();
    const dims = perm.map((p) => x.dims[p]);
    const inStride = [];
    for (let i = rank - 1, s = 1; i >= 0; i--) (inStride[i] = s), (s *= x.dims[i]);
    const out = new Float32Array(x.data.length);
    const idx = new Array(rank).fill(0);
    for (let o = 0; o < out.length; o++) {
      let src = 0;
      for (let k = 0; k < rank; k++) src += idx[k] * inStride[perm[k]];
      out[o] = x.data[src];
      for (let k = rank - 1; k >= 0 && ++idx[k] === dims[k]; k--) idx[k] = 0;
    }
    return { dims, data: out };
  },

  Conv([x, w, b], a) {
    if (x.dims.length !== 3 || (a.group ?? 1) !== 1) throw new Error('tinyonnx: только Conv1d, group=1');
    if ((a.strides || [1])[0] !== 1 || (a.dilations || [1])[0] !== 1) throw new Error('tinyonnx: stride/dilation ≠ 1');
    const [N, C, L] = x.dims;
    const [O, , K] = w.dims;
    const pad = (a.pads || [0, 0])[0];
    const Lout = L + (a.pads ? a.pads[0] + a.pads[1] : 0) - K + 1;
    const out = new Float32Array(N * O * Lout);
    const X = x.data;
    const W = w.data;
    for (let n = 0; n < N; n++)
      for (let o = 0; o < O; o++) {
        const ob = (n * O + o) * Lout;
        const bias = b ? b.data[o] : 0;
        for (let t = 0; t < Lout; t++) out[ob + t] = bias;
        for (let c = 0; c < C; c++) {
          const xb = (n * C + c) * L;
          const wb = (o * C + c) * K;
          for (let k = 0; k < K; k++) {
            const wv = W[wb + k];
            const shift = k - pad;
            const t0 = Math.max(0, -shift);
            const t1 = Math.min(Lout, L - shift);
            for (let t = t0; t < t1; t++) out[ob + t] += wv * X[xb + t + shift];
          }
        }
      }
    return { dims: [N, O, Lout], data: out };
  },

  Relu([x]) {
    return { dims: x.dims, data: x.data.map((v) => (v > 0 ? v : 0)) };
  },

  Add([a, b]) {
    if (a.data.length !== b.data.length) throw new Error('tinyonnx: Add без broadcast');
    return { dims: a.dims, data: a.data.map((v, i) => v + b.data[i]) };
  },

  ReduceMean: (inp, a) => reduceLast(inp, a, 'mean'),
  ReduceMax: (inp, a) => reduceLast(inp, a, 'max'),

  Concat(xs, { axis }) {
    const rank = xs[0].dims.length;
    axis = norm(axis, rank);
    const outer = size(xs[0].dims.slice(0, axis));
    const dims = [...xs[0].dims];
    dims[axis] = xs.reduce((s, x) => s + x.dims[axis], 0);
    const out = new Float32Array(size(dims));
    let p = 0;
    for (let o = 0; o < outer; o++)
      for (const x of xs) {
        const chunk = size(x.dims.slice(axis));
        out.set(x.data.subarray(o * chunk, (o + 1) * chunk), p);
        p += chunk;
      }
    return { dims, data: out };
  },

  Gemm([x, w, b], { alpha = 1, beta = 1, transA = 0, transB = 0 }) {
    if (transA) throw new Error('tinyonnx: Gemm transA');
    const [M, K] = x.dims;
    const N = transB ? w.dims[0] : w.dims[1];
    const out = new Float32Array(M * N);
    for (let m = 0; m < M; m++)
      for (let n = 0; n < N; n++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += x.data[m * K + k] * (transB ? w.data[n * K + k] : w.data[k * N + n]);
        out[m * N + n] = alpha * s + (b ? beta * b.data[b.data.length === N ? n : 0] : 0);
      }
    return { dims: [M, N], data: out };
  },

  Softmax([x], { axis = -1 }) {
    if (norm(axis, x.dims.length) !== x.dims.length - 1) throw new Error('tinyonnx: Softmax только по последней оси');
    const L = x.dims[x.dims.length - 1];
    const out = new Float32Array(x.data.length);
    for (let r = 0; r < x.data.length / L; r++) {
      let mx = -Infinity;
      for (let i = 0; i < L; i++) mx = Math.max(mx, x.data[r * L + i]);
      let s = 0;
      for (let i = 0; i < L; i++) s += out[r * L + i] = Math.exp(x.data[r * L + i] - mx);
      for (let i = 0; i < L; i++) out[r * L + i] /= s;
    }
    return { dims: x.dims, data: out };
  },
};

function reduceLast([x], { axes, keepdims = 1 }, kind) {
  const rank = x.dims.length;
  if (!axes || axes.length !== 1 || norm(axes[0], rank) !== rank - 1) throw new Error('tinyonnx: Reduce только по последней оси');
  const L = x.dims[rank - 1];
  const rows = x.data.length / L;
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let acc = kind === 'max' ? -Infinity : 0;
    for (let i = 0; i < L; i++) {
      const v = x.data[r * L + i];
      acc = kind === 'max' ? Math.max(acc, v) : acc + v;
    }
    out[r] = kind === 'max' ? acc : acc / L;
  }
  return { dims: keepdims ? [...x.dims.slice(0, -1), 1] : x.dims.slice(0, -1), data: out };
}

// ---------------------------------------------------------------- сессия
export class TinySession {
  constructor(graph) {
    for (const n of graph.nodes) if (!OPS[n.op]) throw new Error(`tinyonnx: операция ${n.op} не поддерживается`);
    this.g = graph;
  }

  static async create(url) {
    const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
    return new TinySession(parseModel(buf));
  }

  static fromBuffer(buf) {
    return new TinySession(parseModel(buf));
  }

  /** feeds: { name: { data: Float32Array, dims: number[] } } -> { name: { data, dims } } */
  run(feeds) {
    const env = { ...this.g.init, ...feeds };
    for (const n of this.g.nodes) {
      const ins = n.inputs.map((name) => (name ? env[name] : undefined));
      env[n.outputs[0]] = OPS[n.op](ins, n.attrs);
    }
    return Object.fromEntries(this.g.outputs.map((o) => [o, env[o]]));
  }
}
