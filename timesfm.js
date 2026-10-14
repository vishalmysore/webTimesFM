// timesfm.js — TimesFM 2.5 (200M) forecasting in the browser via ONNX Runtime Web.
// The ONNX graph holds only the transformer core (patch embed -> 20 layers -> output heads).
// Everything around it (padding, patch masks, running RevIN stats, de-normalisation,
// autoregressive roll-out for horizons > 128) is reproduced here, mirroring
// TimesFM_2p5_200M_torch_module.decode() from google-research/timesfm.

const P = 32;        // input patch length
const O = 128;       // output patch length (steps produced per forward pass)
const Q = 10;        // [mean, q10, q20, ..., q90]
const MEDIAN = 5;    // decode index used for the point forecast / AR feedback
const TOL = 1e-6;

export const QUANTILES = ["mean", "q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"];

/** Running mean / std after each patch (util.update_running_stats). */
function runningStats(x, m, nPatches) {
  const mus = new Float64Array(nPatches), sgs = new Float64Array(nPatches);
  let n = 0, mu = 0, sg = 0;
  for (let p = 0; p < nPatches; p++) {
    let k = 0, s = 0;
    for (let i = 0; i < P; i++) if (m[p * P + i] < 0.5) { k++; s += x[p * P + i]; }
    if (k > 0) {
      const im = s / k;
      let v = 0;
      for (let i = 0; i < P; i++) if (m[p * P + i] < 0.5) { const d = x[p * P + i] - im; v += d * d; }
      const iv = v / k, nn = n + k, nm = (n * mu + im * k) / nn;
      const nv = (n * sg * sg + k * iv + n * (mu - nm) ** 2 + k * (im - nm) ** 2) / nn;
      n = nn; mu = nm; sg = Math.sqrt(Math.max(nv, 0));
    }
    mus[p] = mu; sgs[p] = sg;
  }
  return [mus, sgs];
}

export class TimesFM {
  constructor(session, ort, info) { this.session = session; this.ort = ort; this.info = info; }

  /**
   * @param ort       the onnxruntime-web module
   * @param modelUrl  URL of timesfm25_q4.onnx / timesfm25_fp32.onnx
   * @param backend   "webgpu" | "wasm"
   */
  static async load(ort, modelUrl, backend = "webgpu", onProgress) {
    const buf = await fetchWithProgress(modelUrl, onProgress);
    const t0 = performance.now();
    const providers = backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
    const session = await ort.InferenceSession.create(buf, { executionProviders: providers, graphOptimizationLevel: "all" });
    return new TimesFM(session, ort, { backend, bytes: buf.byteLength, sessionMs: performance.now() - t0 });
  }

  /**
   * Forecast one univariate series.
   * @param series   array of numbers (NaN allowed — treated as missing)
   * @param horizon  steps to forecast
   * @param opts     { maxContext: 1024, continuousQuantiles: true }
   *                  maxContext: context is truncated to the most recent maxContext points.
   *                  continuousQuantiles: use the 1024-step quantile head for q10..q90 (as the
   *                  official ForecastConfig.use_continuous_quantile_head does) — smoother bands.
   * @returns { point: Float32Array(h), quantiles: Float32Array[h][10], passes, ms }
   */
  async forecast(series, horizon, opts = {}) {
    const maxContext = opts.maxContext ?? 1024;
    const continuous = (opts.continuousQuantiles ?? true) && horizon <= 1024;
    let spread = null; // [1024][10] de-normalised, from the first pass
    let s = Array.from(series).slice(-maxContext);
    const pad = (P - (s.length % P)) % P;
    let x = [...new Array(pad).fill(0), ...s.map(v => (Number.isFinite(v) ? v : 0))];
    let m = [...new Array(pad).fill(1), ...s.map(v => (Number.isFinite(v) ? 0 : 1))];
    const out = [];
    let passes = 0;
    const t0 = performance.now();
    while (out.length < horizon) {
      const nP = x.length / P;
      const [mus, sgs] = runningStats(x, m, nP);
      const xn = new Float32Array(x.length), mk = new Float32Array(m);
      for (let p = 0; p < nP; p++) {
        const sd = sgs[p] < TOL ? 1 : sgs[p];
        for (let i = 0; i < P; i++) {
          const j = p * P + i;
          xn[j] = m[j] > 0.5 ? 0 : (x[j] - mus[p]) / sd;
        }
      }
      const feeds = {
        inputs: new this.ort.Tensor("float32", xn, [1, nP, P]),
        masks: new this.ort.Tensor("float32", mk, [1, nP, P]),
      };
      const res = await this.session.run(feeds, passes === 0 && continuous ? ["point", "quantile_spread"] : ["point"]);
      const pt = res.point.data; // [1,128,10], normalised with stats of the last patch
      const mu = mus[nP - 1], sg = sgs[nP - 1];
      const step = [];
      for (let t = 0; t < O; t++) {
        const row = new Float32Array(Q);
        for (let q = 0; q < Q; q++) row[q] = pt[t * Q + q] * sg + mu;
        step.push(row);
      }
      if (res.quantile_spread) {
        const qs = res.quantile_spread.data;
        spread = Array.from({ length: 1024 }, (_, t) => Float32Array.from({ length: Q }, (_, q) => qs[t * Q + q] * sg + mu));
      }
      out.push(...step);
      x = x.concat(step.map(r => r[MEDIAN]));
      m = m.concat(new Array(O).fill(0));
      passes++;
    }
    const quantiles = out.slice(0, horizon);
    if (spread) for (let t = 0; t < horizon; t++)
      for (const q of [1, 2, 3, 4, 6, 7, 8, 9]) quantiles[t][q] = spread[t][q] - spread[t][MEDIAN] + quantiles[t][MEDIAN];
    return { point: Float32Array.from(quantiles, r => r[MEDIAN]), quantiles, passes, ms: performance.now() - t0 };
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
