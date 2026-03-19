"""
main.py — Premier League Simulation Entry Point
================================================

Usage
-----
    python main.py

Pipeline
--------
1. Feature engineering  (data_pipeline via PremierLeagueSimulation.prepare)
2. Monte Carlo simulation  (PremierLeagueSimulation.run)
3. Print detailed summary  (PremierLeagueSimulation.print_summary)
4. Backtesting + Brier Score validation  (validator module)
5. Save all outputs to outputs/
"""

import pandas as pd
from simulation import PremierLeagueSimulation
from validator import (
    compute_backtesting,
    compute_aggregate_metrics,
    compute_brier_scores,
    print_validation_report,
)

# ---------------------------------------------------------------------------
# Configuration — edit only this section
# ---------------------------------------------------------------------------

CSV_PATH  = "data/dataset - 2020-09-24.csv"
N_SEASONS = 1000
SEED      = 42

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Step 1-3 — Simulation
    sim = PremierLeagueSimulation(
        csv_path  = CSV_PATH,
        n_seasons = N_SEASONS,
        seed      = SEED,
    )
    sim.prepare(force_rerun=False)
    summary = sim.run()
    sim.print_summary(summary)

    # Step 4 — Validation
    print("\n" + "─" * 60)
    print("  Running validation...")
    print("─" * 60)

    bt      = compute_backtesting(summary)
    metrics = compute_aggregate_metrics(bt)
    brier   = compute_brier_scores(summary)

    print_validation_report(
        summary   = summary,
        bt        = bt,
        metrics   = metrics,
        brier     = brier,
        n_seasons = N_SEASONS,
    )

    # Step 5 — Save outputs
    summary.to_csv("outputs/simulation_summary.csv")
    bt.to_csv("outputs/backtesting.csv")

    brier_rows = [
        {
            "Category":        cat,
            "BrierScore":      b["BrierScore"],
            "BrierSkillScore": b["BrierSkillScore"],
            "LogLoss":         b["LogLoss"],
        }
        for cat, b in brier.items()
    ]
    pd.DataFrame(brier_rows).to_csv("outputs/brier_scores.csv", index=False)

    print("\n📄 Files saved:")
    print("   outputs/simulation_summary.csv")
    print("   outputs/backtesting.csv")
    print("   outputs/brier_scores.csv")
