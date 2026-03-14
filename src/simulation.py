"""
simulation.py — Premier League Monte Carlo Orchestrator
=======================================================

Connects data_pipeline.py and match_engine.py, runs N-season Monte Carlo
simulations, and produces a detailed per-team summary DataFrame.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from data_pipeline import run_pipeline
from match_engine import (
    MatchEngine,
    STREAK_MODIFIERS,
    WINLESS_STREAK_MODIFIERS,
    DERBY_COEFFICIENTS,
)

OUTPUTS_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Model parameter reference table
# ---------------------------------------------------------------------------
# Each entry documents one modelling decision: name, formula/value,
# academic source, and a short rationale.

PARAMETER_DOCS = [
    (
        "Poisson Distribution",
        "np.random.poisson(λ)",
        "Maher (1982); Dixon & Coles (1997)",
        "Goals in football are rare, independent events — the standard "
        "justification for Poisson; industry baseline for match simulation.",
    ),
    (
        "Lambda Formula",
        "(AvgAttack_home + AvgDefense_away) / 2",
        "Dixon & Coles (1997)",
        "Combines home attack strength with away defensive weakness "
        "to derive expected goals per match.",
    ),
    (
        "Home Advantage",
        "home×1.10 / away×0.90",
        "Nevill & Holder (1999, Sports Medicine)",
        "PL data shows ~10 % lambda uplift for the home side. "
        "Symmetric: total multiplier always sums to 2.0.",
    ),
    (
        "Weighted Player Contribution",
        "Σ(goals + 0.5×assists) / apps × pos_w × app_w",
        "Carmichael et al. (2001, BJIR)",
        "Each player is normalised by their own appearance count; "
        "contribution is then weighted by their share of team appearances.",
    ),
    (
        "Age Decay",
        "max(0.70, 1.0 − 0.04×(age−30))",
        "Dendir (2016, EJSS)",
        "Physical peak at 26-28; ~4 % annual decline after 30. "
        "Floor at 0.70 buffers for accumulated experience.",
    ),
    (
        "Win Streak (Hot Hand)",
        "1-2 games: +1.0% / 3-5: +1.5% / 6+: −1.5%",
        "Gilovich et al. (1985); Miller & Sanjurjo (2018)",
        "Hot-hand effect is weak in team sports. "
        "Long streaks trigger mean-reversion pressure.",
    ),
    (
        "Loss Streak (Loser Effect)",
        "1-2: −1.0% / 3-5: −2.5% / 6: −2.0% / 7+: +1.5%",
        "Audas et al. (2002); Roebber et al. (2022, PLOS ONE)",
        "Cortisol rise and confidence erosion. "
        "After 7+ losses, opponent motivation drops and regression begins.",
    ),
    (
        "Winless Streak",
        "1-2: neutral / 3-6: −0.8→−1.5% / 7+: −1.8%→−1.2%",
        "Audas et al. (2002); Cohen-Zada et al. (2017)",
        "Draws preserve psychological floor; softer than pure loss series. "
        "Disabled when Streak != 0 to avoid double-counting.",
    ),
    (
        "Derby Compression",
        "Strong: ×0.92 / Weak: ×1.15",
        "PL derby match goal-difference analysis",
        "Power gap narrows in local derbies and Top-6 fixtures, "
        "simulating real-world variance increase in high-pressure games.",
    ),
    (
        "Lineup Simulation",
        "11 sampled players × damping(0.35)",
        "Dawson et al. (2000, IJSEP); Carmichael et al. (2001)",
        "Players drawn proportional to appearances each match. "
        "Individual variation is limited (35 % damping) as team quality dominates.",
    ),
]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PremierLeagueSimulation:
    """
    Orchestrate the data_pipeline → MatchEngine → Monte Carlo chain.

    Parameters
    ----------
    csv_path : str
        Path to the raw player dataset CSV.
    n_seasons : int, optional
        Number of Monte Carlo iterations. Default is 1000.
    seed : int or None, optional
        Random seed for reproducibility. ``None`` means no fixed seed.
    """

    def __init__(
        self,
        csv_path:  str,
        n_seasons: int = 1000,
        seed:      int | None = None,
    ):
        self.csv_path  = Path(csv_path)
        self.n_seasons = n_seasons
        self.seed      = seed
        self.engine:      MatchEngine | None  = None
        self.all_seasons: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Step 1 — preparation
    # ------------------------------------------------------------------

    def prepare(self, force_rerun: bool = False) -> None:
        """
        Run the data pipeline and initialise the MatchEngine.

        Parameters
        ----------
        force_rerun : bool, optional
            If ``False`` (default), reuses cached CSVs in ``outputs/`` when
            available. Set to ``True`` after updating the source dataset.
        """
        OUTPUTS_DIR.mkdir(exist_ok=True)

        team_stats_path    = OUTPUTS_DIR / "team_stats.csv"
        player_scores_path = OUTPUTS_DIR / "player_scores.csv"

        if (
            not force_rerun
            and team_stats_path.exists()
            and player_scores_path.exists()
        ):
            print("📂 Loading from cache (pipeline skipped)...")
            team_stats    = pd.read_csv(team_stats_path)
            player_scores = pd.read_csv(player_scores_path)
        else:
            print("🔧 Running pipeline...")
            team_stats, player_scores = run_pipeline(
                csv_path=self.csv_path,
                output_team_stats=str(team_stats_path),
                output_player_scores=str(player_scores_path),
            )

        print("⚙️  Initialising MatchEngine...")
        self.engine = MatchEngine(
            teams_data=team_stats,
            player_rosters=player_scores,
        )
        print(f"✅ Ready — {len(self.engine.teams)} teams loaded.\n")

    # ------------------------------------------------------------------
    # Step 2 — simulation
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Run the Monte Carlo simulation for ``n_seasons`` seasons.

        Returns
        -------
        pd.DataFrame
            Per-team summary with averages, standard deviations, percentiles,
            streak statistics, and positional probabilities.

        Raises
        ------
        RuntimeError
            If ``prepare()`` has not been called first.
        """
        if self.engine is None:
            raise RuntimeError("Call .prepare() before .run().")

        if self.seed is not None:
            np.random.seed(self.seed)

        print(f"🏃 Simulating {self.n_seasons} seasons...")

        season_tables = []
        for i in range(self.n_seasons):
            table           = self.engine.run_season()
            table           = table.copy()
            table["season"] = i
            table["rank"]   = range(1, len(table) + 1)
            season_tables.append(table)

            if (i + 1) % 200 == 0:
                print(f"   {i + 1}/{self.n_seasons} seasons complete...")

        self.all_seasons = pd.concat(season_tables)
        return self._build_summary()

    # ------------------------------------------------------------------
    # Internal — summary builder
    # ------------------------------------------------------------------

    def _build_summary(self) -> pd.DataFrame:
        """
        Aggregate all simulated seasons into a per-team summary table.

        Returns
        -------
        pd.DataFrame
            Index: team name. Columns include AvgPTS, StdPTS, percentiles,
            streak stats, ChampionPct, Top4Pct, TopHalfPct, RelegationPct.
        """
        df = self.all_seasons

        base = df.groupby(df.index).agg(
            AvgPTS = ("PTS", "mean"),
            StdPTS = ("PTS", "std"),
            AvgW   = ("W",   "mean"),
            AvgD   = ("D",   "mean"),
            AvgL   = ("L",   "mean"),
            AvgGF  = ("GF",  "mean"),
            AvgGA  = ("GA",  "mean"),
            AvgGD  = ("GD",  "mean"),
        ).round(2)

        streak = df.groupby(df.index).agg(
            AvgPeakWinStreak     = ("PeakWinStreak",     "mean"),
            MaxPeakWinStreak     = ("PeakWinStreak",     "max"),
            AvgPeakLossStreak    = ("PeakLossStreak",    "mean"),
            MaxPeakLossStreak    = ("PeakLossStreak",    "max"),
            AvgPeakWinlessStreak = ("PeakWinlessStreak", "mean"),
            MaxPeakWinlessStreak = ("PeakWinlessStreak", "max"),
        ).round(2)

        # Points percentiles
        pct = df.groupby(df.index)["PTS"].quantile([0.10, 0.50, 0.90]).unstack()
        pct.columns = ["PTS_p10", "PTS_median", "PTS_p90"]
        pct = pct.round(1)

        # Positional probabilities
        n       = self.n_seasons
        n_teams = df.index.nunique()

        champions = (
            df.groupby("season")["PTS"].idxmax()
            .value_counts(normalize=True).mul(100).round(1)
            .rename("ChampionPct")
        )
        top4 = (
            df[df["rank"] <= 4]
            .groupby(df[df["rank"] <= 4].index).size()
            .div(n).mul(100).round(1).rename("Top4Pct")
        )
        top_half = (
            df[df["rank"] <= 10]
            .groupby(df[df["rank"] <= 10].index).size()
            .div(n).mul(100).round(1).rename("TopHalfPct")
        )
        relegation = (
            df[df["rank"] >= n_teams - 2]
            .groupby(df[df["rank"] >= n_teams - 2].index).size()
            .div(n).mul(100).round(1).rename("RelegationPct")
        )

        return (
            base
            .join(streak)
            .join(pct)
            .join(champions,  how="left")
            .join(top4,       how="left")
            .join(top_half,   how="left")
            .join(relegation, how="left")
            .fillna(0)
            .sort_values("AvgPTS", ascending=False)
        )

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def print_summary(self, summary: pd.DataFrame) -> None:
        """
        Print a multi-section simulation report to stdout.

        Sections
        --------
        1. Model parameters and academic references.
        2. Streak and derby coefficients.
        3. Core season statistics (average per season).
        4. Points distribution percentiles.
        5. Streak peak statistics.
        6. Positional probability table.

        Parameters
        ----------
        summary : pd.DataFrame
            Output of ``run()``.
        """
        W = 88

        def header(title: str) -> None:
            print("\n" + "=" * W)
            print(f"  {title}")
            print("=" * W)

        def subheader(title: str) -> None:
            print(f"\n  ── {title} {'─'*(W-6-len(title))}")

        # Section 1 — Model parameters
        header(
            f"PREMIER LEAGUE SIMULATION — MODEL PARAMETERS  "
            f"({self.n_seasons} seasons, seed={self.seed})"
        )

        print(f"\n  {'#':<3} {'Parameter':<28} {'Source':<36} Description (summary)")
        print(f"  {'─'*3} {'─'*28} {'─'*36} {'─'*16}")
        for i, (name, value, source, desc) in enumerate(PARAMETER_DOCS, 1):
            short_desc = desc[:45] + "..." if len(desc) > 45 else desc
            print(f"  {i:<3} {name:<28} {source:<36} {short_desc}")

        subheader("Formulas and Values")
        for i, (name, value, _, desc) in enumerate(PARAMETER_DOCS, 1):
            print(f"\n  {i:>2}. {name}")
            print(f"      Formula : {value}")
            print(f"      Detail  : {desc}")

        # Section 2 — Streak coefficients
        header("STREAK COEFFICIENTS — Empirically Grounded Values")

        subheader("Win & Loss Streaks (STREAK_MODIFIERS)")
        print(f"\n  {'Streak':<8} {'Win':>14} {'Effect':>8}   {'Loss':>14} {'Effect':>8}")
        print(f"  {'─'*8} {'─'*14} {'─'*8}   {'─'*14} {'─'*8}")

        win_keys  = sorted([k for k in STREAK_MODIFIERS["win"]  if isinstance(k, int)])
        loss_keys = sorted([k for k in STREAK_MODIFIERS["loss"] if isinstance(k, int)])
        all_int_keys = sorted(set(win_keys + loss_keys))

        for k in all_int_keys:
            wv = STREAK_MODIFIERS["win"].get(k)
            lv = STREAK_MODIFIERS["loss"].get(k)
            ws = f"{wv:.3f}" if wv else "  —  "
            we = f"({(wv-1)*100:+.1f}%)" if wv else ""
            ls = f"{lv:.3f}" if lv else "  —  "
            le = f"({(lv-1)*100:+.1f}%)" if lv else ""
            print(f"  {k:<8} {ws:>14} {we:>8}   {ls:>14} {le:>8}")

        wfb  = next(v for k, v in STREAK_MODIFIERS["win"].items()  if isinstance(k, str))
        lfb  = next(v for k, v in STREAK_MODIFIERS["loss"].items() if isinstance(k, str))
        wfbk = next(k for k in STREAK_MODIFIERS["win"]  if isinstance(k, str))
        lfbk = next(k for k in STREAK_MODIFIERS["loss"] if isinstance(k, str))
        print(
            f"  {wfbk+' (fb)':<8} {wfb:>14.3f} {f'({(wfb-1)*100:+.1f}%)':>8}   "
            f"{lfb:>14.3f} {f'({(lfb-1)*100:+.1f}%)':>8}   ← {lfbk} fallback"
        )

        subheader("Winless Streak (WINLESS_STREAK_MODIFIERS)")
        explanations = {
            1: "1-2 draws — no alarm",
            2: "1-2 draws — no alarm",
            3: "Tactical stagnation begins",
            4: "Tactical stagnation continues",
            5: "Goal drought erodes confidence",
            6: "Stagnation deepens",
            7: "Stagnation peak",
            8: "Stagnation peak continues",
            "9+": "Audas et al. negative persistence → slight regression",
        }
        print(f"\n  {'Streak':<8} {'Modifier':>10} {'Effect':>8}   Description")
        print(f"  {'─'*8} {'─'*10} {'─'*8}   {'─'*30}")
        for k, v in WINLESS_STREAK_MODIFIERS.items():
            label = f"{k}+ (fb)" if isinstance(k, str) else str(k)
            expl  = explanations.get(k, "")
            print(f"  {label:<8} {v:>10.3f} {f'({(v-1)*100:+.1f}%)':>8}   {expl}")

        subheader("Derby Compression (DERBY_COEFFICIENTS)")
        print(
            f"\n  {'Match Type':<22} {'Strong Mod':>15} {'Effect':>8}   "
            f"{'Weak Mod':>15} {'Effect':>8}"
        )
        print(f"  {'─'*22} {'─'*15} {'─'*8}   {'─'*15} {'─'*8}")
        for mt, (strong, weak) in DERBY_COEFFICIENTS.items():
            print(
                f"  {mt.value:<22} {strong:>15.2f} {f'({(strong-1)*100:+.1f}%)':>8}   "
                f"{weak:>15.2f} {f'({(weak-1)*100:+.1f}%)':>8}"
            )

        # Section 3 — Core statistics
        header("CORE SEASON STATISTICS  (Average per Season)")
        print(
            f"\n  {'Team':<28} {'AvgPTS':>7} {'±Std':>6} "
            f"{'AvgW':>5} {'AvgD':>5} {'AvgL':>5} "
            f"{'AvgGF':>6} {'AvgGA':>6} {'AvgGD':>6}"
        )
        print(f"  {'─'*28} {'─'*7} {'─'*6} {'─'*5} {'─'*5} {'─'*5} {'─'*6} {'─'*6} {'─'*6}")
        for team, r in summary.iterrows():
            print(
                f"  {team:<28} {r['AvgPTS']:>7.1f} {r['StdPTS']:>6.1f} "
                f"{r['AvgW']:>5.1f} {r['AvgD']:>5.1f} {r['AvgL']:>5.1f} "
                f"{r['AvgGF']:>6.1f} {r['AvgGA']:>6.1f} {r['AvgGD']:>6.1f}"
            )

        # Section 4 — Points distribution
        subheader("Points Distribution — Percentiles")
        print(
            f"\n  {'Team':<28} {'P10 (bad season)':>17} "
            f"{'Median':>8} {'P90 (good season)':>17}  Range"
        )
        print(f"  {'─'*28} {'─'*17} {'─'*8} {'─'*17}  {'─'*8}")
        for team, r in summary.iterrows():
            rng = r["PTS_p90"] - r["PTS_p10"]
            print(
                f"  {team:<28} {r['PTS_p10']:>17.0f} {r['PTS_median']:>8.0f} "
                f"{r['PTS_p90']:>17.0f}  {rng:>6.0f} pts"
            )
        print(f"\n  P10: worst-decile points threshold")
        print(f"  P90: best-decile points threshold")
        print(f"  Range: wide = inconsistent; narrow = stable")

        # Section 5 — Streak statistics
        header("STREAK STATISTICS — In-Season Peak Values")
        print(
            f"\n  {'Team':<28} "
            f"{'WinStr':>7} {'WinMax':>7}  "
            f"{'LossStr':>8} {'LossMax':>8}  "
            f"{'Winless':>8} {'WlsMax':>7}"
        )
        print(f"  {'─'*28} {'─'*7} {'─'*7}  {'─'*8} {'─'*8}  {'─'*8} {'─'*7}")
        for team, r in summary.iterrows():
            print(
                f"  {team:<28} "
                f"{r['AvgPeakWinStreak']:>7.1f} {r['MaxPeakWinStreak']:>7.0f}  "
                f"{r['AvgPeakLossStreak']:>8.1f} {r['MaxPeakLossStreak']:>8.0f}  "
                f"{r['AvgPeakWinlessStreak']:>8.1f} {r['MaxPeakWinlessStreak']:>7.0f}"
            )
        print(f"\n  Avg = mean peak streak length per season across {self.n_seasons} runs")
        print(f"  Max = absolute maximum observed across all {self.n_seasons} seasons")

        # Section 6 — Positional probabilities
        header("POSITIONAL PROBABILITY TABLE")
        print(
            f"\n  {'Team':<28} {'Champion%':>10} {'Top4%':>7} "
            f"{'TopHalf%':>10} {'Relegated%':>11}"
        )
        print(f"  {'─'*28} {'─'*10} {'─'*7} {'─'*10} {'─'*11}")
        for team, r in summary.iterrows():
            print(
                f"  {team:<28} "
                f"{r['ChampionPct']:>9.1f}% "
                f"{r['Top4Pct']:>6.1f}% "
                f"{r['TopHalfPct']:>9.1f}% "
                f"{r['RelegationPct']:>10.1f}%"
            )
        print(f"\n  Top4%      → UEFA Champions League qualification probability")
        print(f"  TopHalf%   → Probability of finishing in the top 10")
        print(f"  Relegated% → Probability of finishing in the bottom 3")

        print("\n" + "=" * W + "\n")
