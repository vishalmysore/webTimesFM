# webTimesFM

Time-series forecasting that runs entirely in your browser. This page runs Google Research's [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (200M) on WebGPU or WASM through [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). There's no server and nothing to install, and your data never leaves the tab.

**Try it:** https://vishalmysore.github.io/webTimesFM/

**Article:** [Forecast Anything in Your Browser](docs/article.md): real examples on COVID, poverty, farming, climate and more.

You can choose the 8-bit model (261 MB, almost identical to full precision, the default) or the 4-bit model (146 MB, smaller but less accurate on short series). Both load from Hugging Face: [VishalMysore/timesfm-2.5-200m-onnx-web](https://huggingface.co/VishalMysore/timesfm-2.5-200m-onnx-web). The model card there explains the conversion, the export and quantisation scripts, and accuracy against PyTorch.

## Files

- `index.html`: demo page with a chart, timing and a self-test against the official PyTorch forecasts
- `timesfm.js`: small wrapper that handles patching, masks, RevIN normalisation, autoregressive roll-out and the quantile head
- `examples.json`: 7 real-world series: COVID-19 cases (USA), world extreme poverty, wheat and maize prices, US unemployment, CO₂ at Mauna Loa, sunspots
- `scripts/build_examples.py`: rebuilds `examples.json` from the public sources (World Bank, FRED/IMF, NOAA, SILSO, JHU)
- `current.json`: data for the **Current affairs** tab: Brent and WTI oil and Bitcoin prices (FRED), and MLB team-strength histories with the live playoff series score (MLB Stats API)
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

This is an unofficial community conversion. Google does not make or endorse it. Apache-2.0.
