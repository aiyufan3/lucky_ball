# Lucky Ball — Lottery Data Analysis System (SSQ & KL8)

> **CN**：基于 Python 的「双色球 & 快乐8」数据抓取、统计分析、可视化、机器学习推荐与回测系统。**仅供技术学习与数据研究，不构成任何投注建议。**
>
> **EN**: A Python toolkit for data fetching, statistical analysis, visualization, ML‑based recommendations and backtesting for **Double Chromosphere (SSQ)** and **Keno / Happy 8 (KL8)**. **For research/education only.**

---

## Table of Contents
- [Highlights](#highlights)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Data Pipelines](#data-pipelines)
  - [SSQ: Double Color Ball](#ssq-double-color-ball)
  - [KL8: Happy 8 / Keno](#kl8-happy-8--keno)
- [Backtesting & Metrics](#backtesting--metrics)
- [Generated Artifacts](#generated-artifacts)
- [Automation (GitHub Actions)](#automation-github-actions)
- [Configuration](#configuration)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Contribution](#contribution)
- [License & Legal Notice](#license--legal-notice)
- [Acknowledgements](#acknowledgements)

---

## Highlights

| Feature | Description |
|---|---|
| **Daily auto-fetch** | Pulls latest draws from the official China Welfare Lottery API. |
| **Rich statistics** | Frequency / hot–cold trends, odd–even mix, sums, span distributions, weekday conditioning. |
| **Visualization** | Frequency histograms, EMA heatmaps, dual-section frequency chart style for KL8. |
| **ML‑assisted SSQ** | LSTM models for red/blue, **three‑way fused prior** (short window / weekday / global), temperature smoothing, ARIMA sum constraints + Monte Carlo low‑entropy selection. |
| **Backtesting** | Rolling evaluation; **Hit@k (Recall@k)** for red, **Top‑k hit rate** for blue; baseline vs ML. |
| **Budget demo (KL8)** | Example **profit plan** generator to allocate a budget across recommended sets (learning purpose). |
| **Automation** | One‑click GitHub Actions for scheduled updates, auto‑commit, tagged release with plots & reports. |

> ⚠️ **Disclaimer**: Lottery draws are random. Historical data cannot predict future outcomes. This project is strictly for learning & research.

---

## Project Structure

```
.
├── .github/workflows/update-lottery-data.yml   # CI: scheduled fetch + analysis + release
├── data/
│   ├── lottery_data.json                       # SSQ history (auto)
│   ├── kl8_history.json                        # KL8 history (auto)
│   └── payouts_kl8.json                        # KL8 payout template (optional)
├── pics/                                       # Exported charts (auto)
│   ├── kl8_frequency_hist.png
│   ├── kl8_ema_heatmap.png
│   └── kl8_dual_frequency_style.png
├── reports/
│   ├── colorballs_analysis_report.md    # SSQ analysis (auto)
│   ├── kl8_analysis_report.md                  # KL8 analysis (auto)
│   └── kl8_profit_plan.md                      # KL8 budget demo (auto when enabled)
├── backtest.py                                 # SSQ rolling backtest
├── lottery_analyzer.py                         # SSQ data/ML/plots/reports
└── super_eight.py                              # KL8 data/plots/recommendations/plan
```

---

## Quickstart

### 1) Environment
- Python **3.11+** recommended (better deps & SSL on macOS)
- `pip install -r requirements.txt`

### 2) Run SSQ end‑to‑end
```bash
python lottery_analyzer.py
```
This will **fetch data**, train LSTM (if enough data), generate **recommendations**, create **plots**, write **reports/HJSON**, and update the recommendations block (see below).

### 3) Run KL8 end‑to‑end (examples)
```bash
# Fetch recent 30 draws and write data/kl8_history.json
python super_eight.py --fetch --limit 30 --report --plots --plots_dual --seed 42

# Generate 5 recommendation sets + budget plan (¥22 at ¥2 each), with plots
python super_eight.py \
  --fetch --limit 30 \
  --recommend 5 \
  --plan --budget 22 --price_per_bet 2 \
  --plots --plots_dual --split 40 \
  --report --seed 42
```

> The KL8 script supports **frequency + EMA trend fusion**, quadrant coverage, and odd–even balance constraints. Charts are saved under `pics/` and the report under `reports/`.

---

## Data Pipelines

### SSQ: Double Color Ball
File: `lottery_analyzer.py`

- **Fetching**: robust pagination, rotating User‑Agents, retries; saves to `data/lottery_data.json`.
- **Statistics**:
  - Frequency (red 1–33, blue 1–16)
  - Odd–even mix, sum buckets, span buckets
  - Weekday‑conditional priors with shrinkage (tunable `SHRINK_BETA_WEEKDAY`)
- **ML Models**:
  - Two **LSTM** predictors (red: multi‑label; blue: single‑label)
  - Engineered features: sums, span, odd/even ratios, cyclical weekday encoding
  - Temperature smoothing (`TAU_RED`, `TAU_BLUE`) to avoid over‑confidence
  - **Three‑way fused prior**: short‑window / weekday / global (weights `FUSION_L1_SHORT`, `FUSION_L2_WEEKDAY`)
  - **ARIMA** forecast on red‑sum → range constraint
  - **Monte Carlo** low‑entropy combination sampling (no‑replacement)
- **Outputs**:
  - `reports/colorballs_analysis_report.md`
  - `data/lottery_aggregated_data.hjson` (with rich comments)
  - `pics/lottery_frequency_analysis.png`

#### Update recommendations block (auto)
The analyzer can update a Markdown block bracketed by anchors. Add this block to any doc you want auto‑updated:

```
<!-- BEGIN:recommendations -->
## 🎯 今日推荐号码 / Today’s Recommendations

**⚠️ 以下推荐号码基于历史统计分析，仅供参考，不保证中奖！**

*(This section will be updated by the automation run.)*
<!-- END:recommendations -->
```

> If you prefer updating **README.md**, call `update_readme_recommendations(readme_path="README.md")` in your workflow/script.

---

### KL8: Happy 8 / Keno
File: `super_eight.py`

- **Fetch**: `--fetch [--limit N]` uses the unified official API (`name=kl8`). Saves `data/kl8_history.json`.
- **Recommend**: `--recommend K` generates K sets of 20 numbers via **frequency + EMA trend** weights with quadrant & odd/even constraints.
- **Plots**: `--plots` (full histogram), `--plots_dual` (1–40 / 41–80 split), `--split` to set the split point.
- **Report**: `--report` writes `reports/kl8_analysis_report.md` (hot/cold tables, latest draw, optional backtest summary).
- **Plan (demo)**: `--plan --budget X --price_per_bet Y` writes `reports/kl8_profit_plan.md` with a simple proportional allocation over recommended sets (**learning purpose**). If you provide a payouts JSON, the script exposes EV/Kelly helpers to extend to analytical EV allocation.

Common one‑liners:
```bash
# Only fetch the freshest N draws
python super_eight.py --fetch --limit 50

# Recommendations + plots + report
python super_eight.py --fetch --limit 50 --recommend 5 --plots --plots_dual --report

# Budget demo plan
python super_eight.py --fetch --limit 30 --recommend 6 --plan --budget 36 --price_per_bet 2
```

---

## Backtesting & Metrics

### SSQ rolling backtest (`backtest.py`)
**Protocol**: train on data up to time *t* → predict *t+1* → compare with truth. Baselines use time‑decayed marginals; ML uses LSTM‑assisted probabilities fused with priors.

**Key metrics**
- **Red Hit@k (Recall@k)**: fraction of the 6 true red balls captured in the top‑k probabilities, averaged over periods.
- **Blue Top‑k hit rate**: whether the true blue is within top‑k.
- Variants compared: `BASE_global`, `BASE_weekday`, `BASE_short`, `BASE_long`, `BASE_mix_β`, `ML_auto`, `ML_fixed`.

**Examples**
```bash
python backtest.py \
  --start 2015-01-01 \
  --seq-len 10 --epochs 3 --hidden-size 64 --lr 1e-3 \
  --k 6 10 12 16 --blue-k 1 2 3 4 \
  --half-life 60 --alpha-fixed 0.40 \
  --short-win 30 --long-win 180 --mix-betas 0.2 0.35 0.5

# Baseline only
python backtest.py --no-ml --k 6 10 12 --blue-k 1 2 --half-life 60
```
The script prints compact tables for red/blue across variants and k.

---

## Generated Artifacts
- **Data**: `data/lottery_data.json`, `data/kl8_history.json`
- **Reports**:
  - `reports/colorballs_analysis_report.md`
  - `reports/kl8_analysis_report.md`
  - `reports/kl8_profit_plan.md` (when `--plan` is used)
- **Charts**: `pics/kl8_frequency_hist.png`, `pics/kl8_ema_heatmap.png`, `pics/kl8_dual_frequency_style.png`, `pics/lottery_frequency_analysis.png`
- **Aggregates**: `data/lottery_aggregated_data.hjson`

> Release assets produced by CI will include updated **data**, **plots**, and **reports**.

---

## Automation (GitHub Actions)
This repo includes a workflow that can run **daily (UTC+8 23:00)** and on manual dispatch.

**What it does**
1. Fetch SSQ & KL8 latest data
2. Generate reports & plots
3. Commit changes and (optionally) create/update a date‑tagged release

**Author/Commit identity**
The workflow is configured to author commits as the **triggering user** with GitHub noreply email:
```bash
git config --local user.name "${{ github.actor }}"
git config --local user.email "${{ github.actor }}@users.noreply.github.com"
```

**Tagging**
The release step checks for an existing tag and **skips** creation if it already exists to avoid failures.

> If you want to use a public email, add a secret like `COMMIT_EMAIL` and set `user.email` accordingly.

---

## Configuration
Tunable knobs (see in‑code defaults):

- **SSQ priors & fusion**: `FUSION_L1_SHORT`, `FUSION_L2_WEEKDAY`, `SHORT_WINDOW`, `SHRINK_BETA_WEEKDAY`
- **Smoothing**: `TAU_RED`, `TAU_BLUE`
- **Backtest**: `--half-life`, `--short-win`, `--long-win`, `--mix-betas`, `--alpha-fixed`
- **KL8**: EMA `alpha`, plot `--split`, plan `--budget/--price_per_bet`

---

## Troubleshooting & FAQ
- **Plots show garbled Chinese**: Install a CJK font (e.g., *Noto Sans CJK SC*) or ensure one of the fallback fonts is available. The scripts set cross‑platform fallbacks.
- **Network / API throttling**: The fetchers use retries and UA rotation. If repeated failures occur, wait and rerun.
- **macOS SSL warnings**: Prefer Python **3.11+**.
- **Where are my files?**: See [Generated Artifacts](#generated-artifacts). CI artifacts and release assets will mirror these paths.

---

## Contribution
PRs & issues are welcome! Typical flow:
1. Fork → branch → commit → PR
2. Please include a brief description and, when applicable, screenshots of plots or snippets of report diffs.

---

## License & Legal Notice
This project is released under the **MIT License**. See [LICENSE](LICENSE).

**Important**
- This repository is for **technical learning and data analysis** only.
- Lottery results are **random**; historical data **cannot** predict the future.
- The authors are **not responsible** for any losses caused by using this code.
- **18+ only**; follow your local laws & regulations.

---

## Acknowledgements
- Original author: [snjyor](https://github.com/snjyor) (forked and maintained by brain404)
- China Welfare Lottery for open data endpoints
- All open‑source libraries used by this project