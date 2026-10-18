# webTimesFM

Time-series forecasting that runs entirely in your browser. This page runs Google Research's [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (200M) and Amazon's [Chronos-2](https://huggingface.co/amazon/chronos-2) (120M) on WebGPU or WASM through [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). Pick one model, or **Compare both** to draw the two forecasts on one chart and score each against the actual values and a "no change" guess. There's no server and nothing to install, and your data never leaves the tab.

**Try it:** https://vishalmysore.github.io/webTimesFM/

**New here? Start with:** [What Is TimesFM? Google's Time-Series Foundation Model Explained](docs/what-is-timesfm.md)

**Article:** [Forecast Anything in Your Browser](docs/article.md): real examples on COVID, poverty, farming, climate and more.

**Article:** [I Asked Google's AI to Predict the World Series, Oil Prices and Bitcoin](docs/current-affairs.md): current-affairs forecasts, updated daily.

You can choose 8-bit (almost identical to full precision, the default) or 4-bit (smaller but less accurate on short series). The files load from Hugging Face, and each model card explains the conversion, the export and quantisation scripts, and accuracy against the official PyTorch code:

| Model | 8-bit | 4-bit | Hugging Face |
|---|---|---|---|
| TimesFM 2.5 | 261 MB | 146 MB | [VishalMysore/timesfm-2.5-200m-onnx-web](https://huggingface.co/VishalMysore/timesfm-2.5-200m-onnx-web) |
| Chronos-2 | 153 MB | 96 MB | [VishalMysore/chronos-2-onnx-web](https://huggingface.co/VishalMysore/chronos-2-onnx-web) |

## TimesFM 2.5 vs Chronos-2

Both models, 8-bit, in the browser, on the most recent points of 10 real series (MAE, lower is better; measured 18 October 2026):

| Series (points hidden) | "No change" | TimesFM 2.5 | Chronos-2 |
|---|---|---|---|
| CO₂ at Mauna Loa (60 months) | 10.24 | **0.56** | 0.97 |
| Sunspots (120 months) | 50.1 | **42.5** | 44.1 |
| Bitcoin (30 days) | 7,153 | **6,949** | 7,272 |
| Wheat price (24 months) | 15.07 | 21.84 | **13.14** |
| Maize price (24 months) | 25.01 | 29.46 | **16.30** |
| World extreme poverty (5 years) | **0.46** | 1.88 | 0.95 |
| US unemployment (24 months) | **0.100** | 0.126 | 0.280 |
| COVID-19 cases, USA (60 days) | **28,940** | 35,320 | 42,390 |
| Brent oil (30 days) | **22.0** | 22.2 | 24.7 |
| WTI oil (30 days) | **6.79** | 7.91 | 8.46 |

TimesFM was closer on smooth cycles (CO₂, sunspots), Chronos-2 on crop prices, and "no change" won half the tests. Ten series is a small test, not a benchmark. The full write-up is in [Time-Series Forecasting Models Compared](docs/time-series-models-compared.md#timesfm-25-vs-chronos-2-side-by-side-in-the-browser).

## Files

- `index.html`: demo page with a chart, timing and a self-test against the official PyTorch forecasts
- `timesfm.js`: small wrapper that handles patching, masks, RevIN normalisation, autoregressive roll-out and the quantile head
- `chronos2.js`: the same for Chronos-2: mean/std scaling with arcsinh, patching, time encoding and un-scaling. Same interface as `timesfm.js`
- `examples.json`: 7 real-world series: COVID-19 cases (USA), world extreme poverty, wheat and maize prices, US unemployment, CO₂ at Mauna Loa, sunspots
- `scripts/build_examples.py`: rebuilds `examples.json` from the public sources (World Bank, FRED/IMF, NOAA, SILSO, JHU)
- `current.json`: data for the **Current affairs** tab: Brent and WTI oil and Bitcoin prices (FRED), and MLB team-strength histories with the live playoff series score (MLB Stats API)
- `scripts/build_news.py`: adds a daily **news tone** for oil, Bitcoin and each playoff team (up to 100 Google News headlines a day, scored with the VADER sentiment model). Unusual tone raises a warning and a stormy-news band on the forecast chart; it never moves the forecast itself. `news_history.json` keeps the daily tones.
- `scripts/build_current.py`: refreshes `current.json`. A GitHub Action (`.github/workflows/current.yml`) runs it every day at 12:30 UTC and commits the result, so the tab stays current. You can also run it by hand: `python scripts/build_current.py current.json`

The **Current affairs** tab estimates World Series odds (Elo ratings, TimesFM strength forecasts and 100,000 simulated playoffs, all in the browser) and forecasts oil and Bitcoin for the next 30 days, with caveats and sources.

The page has a **Tutorial** tab that shows how to turn your own data (spreadsheet, CSV, open-data download) into a series, check accuracy by holding back recent points, and read the forecast band. It also has practice datasets to try.

## Use it in your own page

Copy `timesfm.js` next to your page:

```html
<script type="module">
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.webgpu.min.mjs";
import { TimesFM } from "./timesfm.js";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";

const model = await TimesFM.load(ort,
  "https://huggingface.co/VishalMysore/timesfm-2.5-200m-onnx-web/resolve/main/timesfm25_q8.onnx",
  navigator.gpu ? "webgpu" : "wasm");

const { point, quantiles } = await model.forecast([/* your numbers, oldest first */], 128);
</script>
```

For Chronos-2, copy `chronos2.js` instead. Everything else is the same:

```js
import { Chronos2 } from "./chronos2.js";
const model = await Chronos2.load(ort,
  "https://huggingface.co/VishalMysore/chronos-2-onnx-web/resolve/main/chronos2_q8.onnx",
  navigator.gpu ? "webgpu" : "wasm");
const { point, quantiles } = await model.forecast([/* your numbers, oldest first */], 64);
```

These are unofficial community conversions. Google and Amazon do not make or endorse them. Apache-2.0.
