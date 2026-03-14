# Poisson-Based-Simulation---Premier-League
# ⚽ Premier League Monte Carlo Simulation

> A probabilistic, player-level simulation framework for the English Premier League, grounded in peer-reviewed sports-science literature.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status: Portfolio Project](https://img.shields.io/badge/Status-Portfolio%20Project-orange)]()

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Methodology Summary](#-methodology-summary)
- [Results](#-results)
- [Validation](#-validation)
- [Installation & Usage](#-installation--usage)
- [Output Files](#-output-files)
- [Known Limitations](#-known-limitations)
- [References](#-references)

---

## 🔭 Overview

This project simulates a full Premier League season — all 380 fixtures — up to **N = 1,000 times**, producing probability distributions over final standings rather than a single deterministic prediction. Each team's expected goal rate (λ) is derived from individual player career statistics, normalised per appearance, weighted by positional role, and adjusted for age-related performance decay.

The simulation is validated against the **actual 2020-21 PL season** using:

| Metric | Value | Benchmark |
|--------|-------|-----------|
| RMSE (points) | **11.83** | 8–14 pts (Dixon & Coles family) |
| MAE (points) | **10.14** | — |
| Brier Skill Score — Championship | **+0.66** | 0 = naive |
| Brier Skill Score — Top-4 | **+0.37** | 0 = naive |
| Brier Skill Score — Relegation | **+0.37** | 0 = naive |

---

## ✨ Key Features

- **Player-level data pipeline** — no team-level aggregation shortcut; each player's per-appearance contribution is computed individually before being weighted into a team score.
- **Age decay function** — calibrated to Dendir (2016): linear 4%/yr decline after age 30, floored at 70%.
- **Four-layer contextual match modifiers**: home advantage · momentum streaks · derby/Top-6 rivalry compression · red card events.
- **Stochastic first-XI selection** — each match samples 11 players from the squad with appearance-probability weights.
- **Full probabilistic validation** — RMSE, MAE, Brier Score, Brier Skill Score, Log Loss.

---

## 🗂 Project Architecture

```
pl-simulation/
│
├── data/
│   └── dataset - 2020-09-24.csv     # Raw player career statistics
│
├── src/
│   ├── data_pipeline.py             # Player → team strength parameters
│   ├── match_engine.py              # Single-match Poisson simulator + modifiers
│   ├── simulation.py                # Monte Carlo orchestrator (N seasons)
│   └── validator.py                 # Backtesting + Brier Score validation
│
├── outputs/                         # Auto-generated; git-ignored
│   ├── team_stats.csv
│   ├── player_scores.csv
│   ├── simulation_summary.csv
│   ├── backtesting.csv
│   └── brier_scores.csv
│
├── docs/
│   ├── academic_report.docx         # Technical report (CV-grade)
│   └── study_guide.docx             # Plain-English walkthrough
│
├── tests/
│   └── test_pipeline.py             # Unit tests (coming soon)
│
├── main.py                          # Entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧮 Methodology Summary

### 1 · Attack Lambda Estimation

```
λ_atk_player = (goals + 0.5 × assists) / appearances × pos_weight × age_decay
λ_atk_team   = Σ(λ_atk_player × app_weight)   # appearance-weighted mean
λ_atk_scaled = λ_atk_team × (1.36 / league_mean_raw)   # anchor to PL historical avg
```

Position weights for attack: `GK=0.0 · DEF=0.15 · MID=0.55 · FWD=1.0`

### 2 · Age Decay

```python
def age_decay(age: float) -> float:
    return max(0.70, 1.0 - 0.04 * max(0, age - 30))
```

Academic basis: Dendir (2016, *Journal of Sports Analytics*) — physical peak at 26–28; 3–5%/yr post-30 decline.

### 3 · Match Simulation

```
λ_home = (atk_home + def_away) / 2  ×  1.10  ×  modifiers_home
λ_away = (atk_away + def_home) / 2  ×  0.90  ×  modifiers_away

home_goals = np.random.poisson(λ_home)
away_goals = np.random.poisson(λ_away)
```

### 4 · Contextual Modifiers Stack

| Layer | Effect | Academic Basis |
|-------|--------|----------------|
| Home advantage | ×1.10 home / ×0.90 away | Nevill & Holder (1999) |
| Win streak (1–5 games) | +1.0% to +1.5% | Miller & Sanjurjo (2018) |
| Win streak (6+ games) | −1.5% (regression) | Audas et al. (2002) |
| Loss streak (1–6 games) | −1.0% to −3.0% | Roebber et al. (2022) |
| Loss streak (7+ games) | +1.5% (regression) | Audas et al. (2002) |
| Derby / Top-6 | Stronger ×0.92, Weaker ×1.15 | PL fixture analysis |
| Red card (after 65') | Sender −30%, Opponent +15% | Červený et al. (2018) |

---

## 📊 Results

### Simulation Summary (Top 10 by Average Points)

| Club | Avg PTS | Std PTS | Champion % | Top-4 % | Relegated % |
|------|---------|---------|------------|---------|-------------|
| Manchester City | 71.5 | 7.97 | 50.6% | 89.6% | 0.0% |
| Liverpool | 66.8 | 7.86 | 25.0% | 75.9% | 0.1% |
| Tottenham Hotspur | 63.7 | 7.86 | 12.0% | 64.2% | 0.2% |
| Arsenal | 56.4 | 7.78 | 3.0% | 24.6% | 1.0% |
| Manchester United | 55.9 | 7.70 | 2.2% | 22.8% | 1.3% |
| Leicester City | 55.7 | 8.05 | 2.0% | 23.8% | 3.0% |
| Everton | 53.9 | 8.00 | 1.2% | 17.9% | 4.2% |
| Chelsea | 52.9 | 7.82 | 0.9% | 13.9% | 4.2% |
| Sheffield United | 35.9 | 7.19 | 0.0% | 0.1% | 72.0% |

---

## ✅ Validation

### Backtesting vs. 2020-21 Actual Season

```
RMSE (points) :  11.83  ← within 8–14 pt benchmark for Poisson family
MAE  (points) :  10.14
Bias (points) :  −1.08  ← slight underprediction overall
```

### Brier Score (Probabilistic Calibration)

```
Category        Brier Score   Skill Score   Log Loss   Rating
─────────────────────────────────────────────────────────────
Championship      0.0162        +0.660       0.061    Excellent
Top-4 (UCL)       0.1015        +0.366       0.316    Good
Relegation        0.0797        +0.375       0.317    Good
```

> **All Brier Skill Scores are substantially positive**, confirming the model outperforms a naive equal-probability baseline across all three categorical outcomes.

### Notable Prediction Errors

| Club | Actual PTS | Predicted PTS | Error | Root Cause |
|------|-----------|--------------|-------|------------|
| Fulham | 28 | 54.1 | +26.1 | Newly promoted; players' career stats earned at stronger former clubs |
| West Ham | 65 | 49.4 | −15.6 | Moyes tactical system not capturable from career statistics |
| Man United | 74 | 55.9 | −18.1 | Squad cohesion and Solskjaer's system exceeded individual stats |
| Liverpool | 69 | 66.8 | −2.2 | ✓ Well calibrated |
| Tottenham | 62 | 63.7 | +1.7 | ✓ Well calibrated |

---

## ⚙️ Installation & Usage

### Requirements

```bash
pip install pandas numpy
```

Python 3.10+ required (uses `int | None` union type syntax).

### Run

```bash
# Clone the repository
git clone https://github.com/your-username/pl-simulation.git
cd pl-simulation

# Install dependencies
pip install -r requirements.txt

# Run simulation (1,000 seasons, seed 42)
python main.py
```

### Configuration

Edit the parameters at the top of `main.py`:

```python
CSV_PATH  = "data/dataset - 2020-09-24.csv"
N_SEASONS = 1000   # Number of simulated seasons
SEED      = 42     # Random seed for reproducibility (None = random)
```

Set `force_rerun=True` in `sim.prepare()` to recompute team stats from scratch after changing the dataset.

---

## 📄 Output Files

| File | Description |
|------|-------------|
| `outputs/team_stats.csv` | Attack λ, defence rating, discipline rates per club |
| `outputs/player_scores.csv` | Normalised attack/defence scores per player |
| `outputs/simulation_summary.csv` | Full 1,000-season aggregate statistics |
| `outputs/backtesting.csv` | Sim predictions vs. actual 2020-21 results |
| `outputs/brier_scores.csv` | Brier Score, Skill Score, and Log Loss per category |

---

## ⚠️ Known Limitations

1. **Career-aggregate data** — The dataset uses cumulative PL career totals. Players who accumulated statistics at stronger former clubs inflate their current team's rating (see: Fulham, 2020-21).
2. **Pre-season snapshot** — No within-season updates for transfers, injuries, or form changes.
3. **Poisson independence** — Goals are treated as independent events. The Dixon-Coles low-score correction is not implemented.
4. **Single-season validation** — 20 teams × 1 season = 20 observations; statistically insufficient for robust p-values. The validation is a calibration check, not a formal significance test.

---

## 📚 References

- Audas, Dobson & Goddard (2002). *Journal of Economics and Business*, 54(6), 633–650.
- Brier, G. W. (1950). *Monthly Weather Review*, 78(1), 1–3.
- Carmichael, Thomas & Ward (2001). *Journal of Sports Economics*, 2(3), 228–243.
- Červený, van Ours & van Tuijl (2018). *Empirical Economics*, 55(4), 1979–2004.
- Dawson, Dobson & Gerrard (2000). *Scottish Journal of Political Economy*, 47(4), 399–421.
- Dendir, S. (2016). *Journal of Sports Analytics*, 2(2), 89–105.
- Dixon, M. J. & Coles, S. G. (1997). *Journal of the Royal Statistical Society: C*, 46(2), 265–280.
- Gilovich, Vallone & Tversky (1985). *Cognitive Psychology*, 17(3), 295–314.
- Maher, M. J. (1982). *Statistica Neerlandica*, 36(3), 109–118.
- Miller, J. B. & Sanjurjo, A. (2018). *Econometrica*, 86(6), 2019–2047.
- Nevill, A. M. & Holder, R. L. (1999). *Sports Medicine*, 28(4), 221–236.
- Titman et al. (2015). *Journal of the Royal Statistical Society: A*, 178(3), 659–683.

---

## 📝 License

All Rights Reserved.

---

*Dataset source: Premier League career statistics, cut-off September 2020. Simulation validated against the 2020-21 PL season.*
