// chronos2.js — Amazon Chronos-2 (120M) forecasting in the browser via ONNX Runtime Web.
// The ONNX graph holds only the neural network (patch embedding -> 12 encoder blocks -> quantile head).
// Everything around it (mean/std scaling + arcsinh, patching, time encoding, un-scaling) is reproduced
// here, mirroring Chronos2Model._prepare_patched_context / _prepare_patched_future / forward from
// amazon-science/chronos-forecasting. Same interface as timesfm.js.

const P = 16;            // patch length (input and output)
const CTX = 8192;        // model context length; also the time-encoding scale
const MAX_OUT = 64;      // output patches per pass (64 * 16 = 1024 steps)
const NQ = 21;           // quantile levels 0.01, 0.05, 0.10, ..., 0.95, 0.99
const MED = 10;          // index of q0.50

export const LEVELS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99];
// timesfm.js layout: [mean, q10, q20, ..., q90]. Chronos-2 has no mean head, so slot 0 holds the median.
const TFM_COLS = [MED, 2, 4, 6, 8, 10, 12, 14, 16, 18];

export class Chronos2 {
  constructor(session, ort, info) { this.session = session; this.ort = ort; this.info = info; }

  /**
   * @param ort       the onnxruntime-web module
   * @param modelUrl  URL of chronos2_q8.onnx / chronos2_q4.onnx
   * @param backend   "webgpu" | "wasm"
   */
  static async load(ort, modelUrl, backend = "webgpu", onProgress) {
    const buf = await fetchWithProgress(modelUrl, onProgress);
    const t0 = performance.now();
    const providers = backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
    const session = await ort.InferenceSession.create(buf, { executionProviders: providers, graphOptimizationLevel: "all" });
    return new Chronos2(session, ort, { backend, bytes: buf.byteLength, sessionMs: performance.now() - t0 });
  }

  /** One forward pass: context (NaN = missing) -> NQ x (m*P) quantiles in data units. */
  async pass(context, m) {
    const x = context.slice(-CTX);
    let n = 0, s = 0;
    for (const v of x) if (Number.isFinite(v)) { n++; s += v; }
    const loc = n ? s / n : 0;
    let ss = 0;
    for (const v of x) if (Number.isFinite(v)) ss += (v - loc) ** 2;
    let scale = n ? Math.sqrt(ss / n) : 1;
    if (scale === 0) scale = 1e-5;
    const pad = (P - (x.length % P)) % P, N = (x.length + pad) / P, F = 3 * P;
    const patches = new Float32Array(N * F), pmask = new Float32Array(N);
    for (let p = 0; p < N; p++) for (let i = 0; i < P; i++) {
      const j = p * P + i, v = x[j - pad];
      const ok = j >= pad && Number.isFinite(v);
      patches[p * F + i] = (j - N * P) / CTX;                       // time encoding -C..-1, scaled
      patches[p * F + P + i] = ok ? Math.asinh((v - loc) / scale) : 0;
      patches[p * F + 2 * P + i] = ok ? 1 : 0;
      if (ok) pmask[p] = 1;
    }
    const future = new Float32Array(m * F);
    for (let k = 0; k < m * P; k++) future[Math.floor(k / P) * F + (k % P)] = k / CTX;
    const T = (d, s) => new this.ort.Tensor("float32", d, s);
    const res = await this.session.run({ patches: T(patches, [1, N, F]), patch_mask: T(pmask, [1, N]), future: T(future, [1, m, F]) });
    const raw = res.quantiles.data; // [1, m, NQ*P], (q p) within each output patch
    const out = Array.from({ length: NQ }, () => new Float64Array(m * P));
    for (let a = 0; a < m; a++) for (let q = 0; q < NQ; q++) for (let i = 0; i < P; i++)
      out[q][a * P + i] = Math.sinh(raw[a * NQ * P + q * P + i]) * scale + loc;
    return out;
  }

  /**
   * Forecast one univariate series.
   * @param series   array of numbers (NaN allowed — treated as missing)
   * @param horizon  steps to forecast (up to 1024 in one pass; longer horizons feed the median back in)
   * @param opts     { maxContext: 8192, fixQuantileCrossing: true }
   * @returns { point: Float32Array(h), quantiles: Float32Array[h][10] ([median, q10..q90], like timesfm.js),
   *            levels: Float32Array[h][21] (all of LEVELS), passes, ms }
   */
  async forecast(series, horizon, opts = {}) {
    const maxContext = Math.min(opts.maxContext ?? CTX, CTX);
    let ctx = Array.from(series, v => (Number.isFinite(v) ? v : NaN)).slice(-maxContext);
    const t0 = performance.now();
    const levels = [];
    let passes = 0;
    while (levels.length < horizon) {
      const m = Math.min(MAX_OUT, Math.ceil((horizon - levels.length) / P));
      const q = await this.pass(ctx, m);
      for (let t = 0; t < m * P; t++) levels.push(Float32Array.from({ length: NQ }, (_, k) => q[k][t]));
      ctx = ctx.concat(Array.from(q[MED])).slice(-maxContext);
      passes++;
    }
    levels.length = horizon;
    if (opts.fixQuantileCrossing ?? true) // keep q01 <= q05 <= ... <= q99
      for (const r of levels) for (let k = 1; k < NQ; k++) r[k] = Math.max(r[k], r[k - 1]);
    const quantiles = levels.map(r => Float32Array.from(TFM_COLS, k => r[k]));
    return { point: Float32Array.from(levels, r => r[MED]), quantiles, levels, passes, ms: performance.now() - t0 };
  }
}

async function fetchWithProgress(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  const total = +r.headers.get("content-length") || 0;
  if (!r.body || !onProgress) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader(); const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onProgress(got, total);
  }
  const buf = new Uint8Array(got); let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  return buf;
}
