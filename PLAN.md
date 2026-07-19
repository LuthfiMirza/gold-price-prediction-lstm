# PLAN.md — Gold Prediction

Project: Prediksi Harga Emas (XAU/USD) — LSTM + Streamlit Dashboard
Target: Personal learning project / portfolio (riset, bukan alat trading)
Timeline: fleksibel, dikerjakan per fase
Owner: Luthfi Mirza Darsono

## HOW TO USE THIS FILE

This file is a sequential execution plan for Claude Code.

Rules — do not break these:

1. One prompt per session. Copy exactly one `PROMPT` block into Claude Code. Do not combine.
2. BUILD prompts and AUDIT prompts are separate messages. Never send them together.
3. No phase advances without evidence. Every `AUDIT` block requires real terminal output
   (command output, test results, screenshots of the dashboard) pasted back.
   "It should work" is not evidence.
4. If an audit fails, fix before advancing. Do not stack phases on a broken foundation.
5. Prompts are written in English on purpose. Keep them in English when pasting.
6. **Commit after every phase that passes its audit — one atomic commit, not a pile-up
   at the end.** See "Git Workflow" below for the exact style to use. This is what makes
   the contribution history readable and keeps GitHub activity visible per real step.

## PROGRESS TRACKER

Tick only after the audit passes.

- [x] Phase 0 — Repository & environment scaffold
- [x] Phase 1 — Data ingestion module (`data_fetch.py`)
- [ ] Phase 2 — Baseline naive forecaster + backtesting framework
- [ ] Phase 3 — LSTM training pipeline (`model_train.py`, horizon day/week/month)
- [ ] Phase 4 — Model evaluation (MAPE model vs baseline, walk-forward)
- [ ] Phase 5 — Streamlit dashboard scaffold (horizon selector, layout)
- [ ] Phase 6 — Prediction display with confidence interval + chart
- [ ] Phase 7 — Data freshness status & API-failure warning
- [ ] Phase 8 — Prediction history log vs realized price
- [ ] Phase 9 — Manual retrain trigger + training metadata
- [ ] Phase 10 — Auto-refresh mechanism
- [ ] Phase 11 — Support/resistance levels + CSV export
- [ ] Phase 12 — External macro features (DXY, Fed rate, oil price)
- [ ] Phase 13 — Model comparison (XGBoost / Prophet side-by-side)
- [ ] Phase 14 — Documentation polish & disclaimer review

---

## GIT WORKFLOW — COMMIT DISCIPLINE

Every phase below ends with a `COMMIT` note. Follow it exactly:

- **One commit = one logical change.** If a phase produces two unrelated pieces of
  work (e.g. a bug fix noticed while building the feature), split into two commits.
- **Title**: imperative, specific, verb-first. "Add naive baseline forecaster",
  not "Update model stuff".
- **Body**: explain *why*, not *what* — the diff already shows what changed.
  Mention the reasoning, a constraint you hit, or context a future reader
  (including future-you) wouldn't get from the code alone.
- **No emoji, no "various fixes", no fabricated ticket refs.**
- **Never commit** `.keras`, `.h5`, `.pkl`, `.env`, or `venv/` — already covered
  by `.gitignore`.
- Push after each commit (or batch a few related commits) — ask before the
  first push of a session so we agree on remote/branch, then subsequent
  pushes in that session can go straight through.

Example for this project:

```
Add naive baseline forecaster for backtesting comparison

LSTM accuracy numbers are meaningless without a floor to beat. The naive
forecaster (predict = last known price) gives that floor — if LSTM can't
beat this on MAPE, it isn't earning its complexity.
```

---

## PHASE 0 — Repository & environment scaffold

**Goal**: project skeleton, dependency list, and venv instructions in place.

```
PROMPT (BUILD):
Set up the Python project scaffold for a gold price prediction tool. Create:
- requirements.txt with: yfinance, pandas, numpy, scikit-learn, tensorflow,
  streamlit, matplotlib or plotly, python-dotenv
- A .venv-friendly structure (no venv committed)
- Empty placeholder files: data_fetch.py, model_train.py, app.py
- Each placeholder file should have a one-line module docstring in Indonesian
  describing its purpose (see CLAUDE.md for the convention)
Do not implement any logic yet — just scaffolding.
```

```
AUDIT:
Run: pip install -r requirements.txt (in an activated venv) and paste the
final lines of output showing success or the exact error.
Run: python -c "import yfinance, pandas, numpy, sklearn, tensorflow, streamlit"
and paste the output (should be silent / no error).
```

**COMMIT**: `Scaffold project structure and pin dependencies`
Body: note which TensorFlow/Keras version was pinned and why (e.g. Keras 3
native `.keras` format instead of deprecated `.h5`).

---

## PHASE 1 — Data ingestion module

**Goal**: `data_fetch.py` reliably pulls GC=F historical + latest price, resamples to day/week/month.

```
PROMPT (BUILD):
Implement data_fetch.py:
- fetch_historical(period="5y") -> DataFrame indexed by datetime, using
  yfinance symbol "GC=F"
- resample_data(df, horizon) -> resamples to "day" (raw), "week", or "month"
  using OHLC-aware resampling (keep open/high/low/close, not just close)
- get_latest_price() -> current price + timestamp, with a try/except that
  raises a clear custom exception (DataFetchError) on failure instead of
  silently returning None
- All functions have short Indonesian docstrings per CLAUDE.md convention
```

```
AUDIT:
Run: python -c "from data_fetch import fetch_historical, resample_data; df=fetch_historical(); print(df.tail()); print(resample_data(df,'week').tail())"
Paste the actual printed DataFrame rows.
Also simulate failure: temporarily break the network or symbol and paste
the raised DataFetchError traceback to confirm it fails loudly, not silently.
```

**COMMIT**: `Add data ingestion module with day/week/month resampling`
Body: mention the OHLC-aware resampling decision and the DataFetchError
choice (why silent None was rejected — ties to Phase 7's warning banner).

---

## PHASE 2 — Baseline naive forecaster + backtesting framework

**Goal**: a MAPE-measurable baseline that any real model must beat.

```
PROMPT (BUILD):
Add to a new file backtest.py:
- naive_forecast(series) -> predicts next value = last known value (shifted by 1)
- walk_forward_mape(series, predict_fn, window) -> walk-forward validation:
  slide a window through the series, call predict_fn on each window, compare
  to the actual next value, return MAPE
- Test walk_forward_mape using naive_forecast against real week-resampled
  gold data from data_fetch.py
```

```
AUDIT:
Run: python -c "from data_fetch import fetch_historical, resample_data; from backtest import naive_forecast, walk_forward_mape; s=resample_data(fetch_historical(),'week')['Close']; print(walk_forward_mape(s, naive_forecast, 60))"
Paste the actual MAPE number. This becomes the number every later model must beat.
```

**COMMIT**: `Add naive baseline forecaster for backtesting comparison`
(see example body above)

---

## PHASE 3 — LSTM training pipeline

**Goal**: `model_train.py` trains and saves one LSTM model per horizon.

```
PROMPT (BUILD):
Implement model_train.py:
- LOOKBACK = 60 (module-level constant, documented as not-to-change lightly)
- build_dataset(series, lookback) -> X, y arrays for supervised learning
- build_model() -> 2-layer LSTM with dropout (Keras Sequential)
- train(horizon) -> fetches data, resamples per horizon, fits MinMaxScaler
  ON THE TRAIN SPLIT ONLY (not full series, avoid leakage), trains the model
  with a validation split, saves model_lstm_{horizon}.keras,
  scaler_{horizon}.pkl, series_{horizon}.pkl
- CLI entrypoint: python model_train.py --horizon day|week|month
- Predict RETURNS (percent change) internally if easy, otherwise document
  clearly that it predicts absolute price and why
```

```
AUDIT:
Run: python model_train.py --horizon week
Paste the full training log including final train loss and val loss for
at least the last 5 epochs. State explicitly whether val loss vs train loss
suggests overfitting.
Repeat for --horizon day and --horizon month, paste both logs too.
```

**COMMIT**: split into two commits if needed:
1. `Add LSTM training pipeline with per-horizon model persistence`
2. `Fit scaler on train split only to prevent data leakage` (if this was a
   fix discovered during audit rather than done correctly the first time)

---

## PHASE 4 — Model evaluation

**Goal**: prove (or disprove) that LSTM beats the naive baseline, per horizon.

```
PROMPT (BUILD):
Add evaluate.py:
- evaluate_horizon(horizon) -> loads the trained LSTM + scaler, runs
  walk-forward MAPE using the LSTM's prediction function, and compares
  against naive_forecast's MAPE from backtest.py on the same series/window
- Print a small table: horizon | LSTM MAPE | naive MAPE | winner
```

```
AUDIT:
Run: python evaluate.py for day, week, and month.
Paste the actual table output for all three horizons. If LSTM loses to
naive on any horizon, say so explicitly — do not proceed to hide or
reframe a losing result.
```

**COMMIT**: `Add model evaluation comparing LSTM against naive baseline`
Body: state the actual MAPE numbers found per horizon — this is the kind
of context that won't be visible from the diff.

---

## PHASE 5 — Streamlit dashboard scaffold

**Goal**: `app.py` boots, horizon selector works, shows raw current price.

```
PROMPT (BUILD):
Implement the skeleton of app.py:
- Sidebar with horizon radio (Harian/Mingguan/Bulanan)
- Header showing project title and selected horizon
- Fetch and display current price using data_fetch.get_latest_price()
- No prediction logic yet — this phase is just the shell rendering live data
```

```
AUDIT:
Run: streamlit run app.py, open http://localhost:8501, and paste a
screenshot showing the sidebar horizon selector and the live current price
rendering correctly for at least two different horizon selections.
```

**COMMIT**: `Scaffold Streamlit dashboard with horizon selector`

---

## PHASE 6 — Prediction display with confidence interval + chart

**Goal**: the dashboard's core value — prediction number + range + chart.

```
PROMPT (BUILD):
Extend app.py:
- Load the trained model/scaler for the selected horizon (show a clear
  "model not trained yet, click to train" state if missing)
- Show the 3 summary cards: current price, predicted next price with a
  confidence range (e.g. +/- one backtested MAPE band), percent change
- Render a chart (plotly) of historical prices + the prediction point/band
  at the end, with a toggle between line and candlestick
```

```
AUDIT:
Screenshot the dashboard for each horizon showing: the 3 cards populated
with real numbers, the chart rendering historical + prediction, and the
candlestick toggle working.
```

**COMMIT**: split naturally:
1. `Add prediction summary cards to dashboard`
2. `Add historical + prediction chart with candlestick toggle`

---

## PHASE 7 — Data freshness status & API-failure warning

**Goal**: dashboard never silently shows stale or missing data.

```
PROMPT (BUILD):
Add to app.py:
- A status badge showing "Data segar - X menit lalu" based on the fetched
  timestamp, with color thresholds (green < 15min, yellow < 1h, red > 1h)
- Wrap the data_fetch calls in try/except for DataFetchError, and on
  failure show a visible warning banner (st.warning) instead of crashing
  or silently rendering old data
```

```
AUDIT:
Screenshot the normal state (green badge). Then simulate a fetch failure
(e.g. temporarily rename the yfinance symbol to something invalid) and
paste a screenshot of the warning banner appearing instead of a crash.
```

**COMMIT**: `Add data freshness indicator and API-failure warning banner`

---

## PHASE 8 — Prediction history log vs realized price

**Goal**: track record of real predictions vs what actually happened.

```
PROMPT (BUILD):
Add prediction_log.py:
- log_prediction(horizon, predicted_value, target_date) -> appends to a
  local CSV/SQLite log with timestamp
- get_realized_comparisons(horizon) -> for logged predictions whose
  target_date has passed, fetch the actual price and compute the error
Wire this into app.py: every time a prediction is shown, log it (dedup by
target_date so it doesn't log duplicates on every refresh). Add a table
in the dashboard showing past predictions vs realized prices.
```

```
AUDIT:
Run the dashboard across two different simulated "days" (or manually edit
target_date to be in the past) and paste the resulting comparison table
showing at least one realized row with predicted vs actual vs error %.
```

**COMMIT**: `Add prediction history tracking against realized prices`
Body: explain the dedup-by-target_date decision (why re-logging on every
refresh would be wrong).

---

## PHASE 9 — Manual retrain trigger + training metadata

**Goal**: user can see and force model freshness.

```
PROMPT (BUILD):
- Save a small metadata file per horizon (last trained timestamp, MAPE at
  train time) alongside the model
- Add a "Latih ulang model sekarang" button in the sidebar that calls
  model_train.train(horizon) synchronously and shows a spinner
- Display "Model terakhir dilatih: <date>" near the button
```

```
AUDIT:
Screenshot before/after clicking the retrain button, showing the
"last trained" timestamp updating and the spinner/completion state.
```

**COMMIT**: `Add manual retrain trigger with training metadata display`

---

## PHASE 10 — Auto-refresh mechanism

**Goal**: dashboard re-fetches data and re-predicts on an interval without full retraining.

```
PROMPT (BUILD):
Add streamlit-autorefresh with a configurable interval (default 10 minutes,
not seconds). Confirm in code comments that this only re-runs the fetch +
predict path, never model_train.train().
```

```
AUDIT:
Set the interval to something short for testing (e.g. 15 seconds), run the
dashboard, and paste two consecutive screenshots/timestamps showing the
data freshness badge updating automatically without manual interaction.
Then revert the interval back to the production default and confirm.
```

**COMMIT**: `Add configurable auto-refresh for live price and prediction`

---

## PHASE 11 — Support/resistance levels + CSV export

**Goal**: extra context + data portability.

```
PROMPT (BUILD):
- Add a simple support/resistance calculation (e.g. rolling min/max or
  pivot points) over the historical window, shown as horizontal lines on
  the chart
- Add a "Unduh CSV" button exporting historical + predicted data
```

```
AUDIT:
Screenshot the chart with support/resistance lines visible, and paste the
first few lines of the downloaded CSV file content.
```

**COMMIT**: split:
1. `Add support/resistance levels to price chart`
2. `Add CSV export for historical and predicted data`

---

## PHASE 12 — External macro features

**Goal**: multivariate input — DXY, Fed rate, oil price — to materially improve accuracy.

```
PROMPT (BUILD):
Extend data_fetch.py to also pull DXY (DX-Y.NYB), a Fed funds rate proxy,
and oil price (CL=F) from yfinance, aligned to the same date index as gold.
Extend model_train.py to optionally train a multivariate LSTM using these
as extra input features (keep the univariate path working as a fallback/
comparison, controlled by a flag).
```

```
AUDIT:
Run evaluate.py comparing univariate vs multivariate MAPE per horizon,
paste the actual numbers. State plainly whether the added features helped,
hurt, or made no meaningful difference.
```

**COMMIT**: `Add multivariate LSTM using DXY, Fed rate, and oil as features`
Body: state the measured MAPE delta from Phase 4's baseline table — the
justification for keeping or reverting this change lives here.

---

## PHASE 13 — Model comparison (XGBoost / Prophet)

**Goal**: know whether LSTM is even the right tool for this data size.

```
PROMPT (BUILD):
Add model_compare.py implementing XGBoost and Prophet forecasters using the
same walk-forward MAPE harness from backtest.py, so all models (naive,
LSTM, XGBoost, Prophet) are judged identically. Surface this comparison
table in the dashboard's accuracy panel.
```

```
AUDIT:
Paste the full comparison table (all 4 models x all 3 horizons) with real
MAPE numbers, plus a dashboard screenshot showing the panel.
```

**COMMIT**: `Add XGBoost and Prophet to model comparison harness`

---

## PHASE 14 — Documentation & disclaimer review

**Goal**: README/CLAUDE.md match the shipped app; disclaimer is unmissable.

```
PROMPT (BUILD):
Review README.md and CLAUDE.md against the actual final app.py behavior —
fix any drift (file names, CLI flags, feature list). Confirm the financial
disclaimer renders on every horizon view, not just the default one.
```

```
AUDIT:
Diff-read README/CLAUDE.md against the actual code and paste a short list
of any corrections made. Screenshot the disclaimer visible on all three
horizon views.
```

**COMMIT**: `Sync documentation with final dashboard behavior`
