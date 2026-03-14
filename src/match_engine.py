"""
match_engine.py — Poisson-Based Match Simulation Engine
========================================================

Simulates individual Premier League matches using a Poisson goal model
extended with the following components:

  - Home-advantage multiplier (Nevill & Holder 1999)
  - Momentum / streak modifiers (Gilovich et al. 1985; Audas et al. 2002)
  - Derby & Top-6 lambda compression
  - Team-specific red-card simulation (Červený et al. 2018)
  - Dixon-Coles low-score correction (Dixon & Coles 1997)
  - Optional match-day lineup sampling from player_scores.csv
"""

import math
import pandas as pd
import numpy as np
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Match type
# ---------------------------------------------------------------------------

class MatchType(Enum):
    NORMAL = "NORMAL"
    TOP6   = "TOP6"
    DERBY  = "DERBY"


# ---------------------------------------------------------------------------
# Derby compression coefficients
# ---------------------------------------------------------------------------
# Applied to the stronger team's lambda (strong_mod < 1) and the weaker
# team's lambda (weak_mod > 1) to reflect the motivation-equalisation effect
# observed in high-pressure fixtures.
DERBY_COEFFICIENTS: dict[MatchType, tuple[float, float]] = {
    # (strong_mod, weak_mod)
    MatchType.DERBY: (0.92, 1.15),  # Local derbies: large compression
    MatchType.TOP6:  (0.96, 1.08),  # Top-6 clashes: moderate compression
}


# ---------------------------------------------------------------------------
# Streak modifiers
# ---------------------------------------------------------------------------
# Based on Gilovich, Vallone & Tversky (1985) and Miller & Sanjurjo (2018).
# Team-sport literature shows that momentum effects are small and short-lived;
# P(Win | k-game streak) ≈ season win rate for k ≥ 5.
# Coefficients are intentionally kept in the ±1–2 % band.
#
# Structure: streak_length → modifier
#   Positive key = win-streak length
#   Negative key = loss-streak length (stored as absolute value)
STREAK_MODIFIERS: dict[str, dict] = {
    # Win streaks (hot-hand — weak, short-lived)
    "win": {
        1: 1.010,   # Slight positive correlation, 1-2 games
        2: 1.010,
        3: 1.015,   # Very weak extra effect; base quality already priced in
        4: 1.015,
        5: 1.010,   # Transition to plateau
        "6+": 0.985,  # Mean reversion pressure; opponent motivation rises
    },

    # Loss streaks (loser effect & negative psychological momentum)
    # Audas, Dobson & Goddard (2002) — PL & Football League, 30 years,
    # 60 000+ matches: negative persistence observed after controlling for
    # team quality. Roebber et al. (2022, PLOS ONE): cortisol / testosterone
    # mechanism; Taylor & Demick (1994): precipitating-event chain.
    "loss": {
        1: 0.990,   # Physiological disruption beginning; team dynamics intact
        2: 0.990,
        3: 0.975,   # Confidence erosion + intra-squad tension
        4: 0.975,
        5: 0.970,   # Crisis deepens; forward line loses creativity
        6: 0.980,   # Manager intervention threshold — gradual recovery signal
        "7+": 1.015,  # Audas et al. negative persistence: expectation so low
                       # that opponent motivation drops; mean reversion kicks in
    },
}


# ---------------------------------------------------------------------------
# Winless-streak modifiers
# ---------------------------------------------------------------------------
# A "winless streak" is any consecutive run without a win (losses + draws).
# Draws preserve a psychological floor, so modifiers are softer than pure
# loss-streak modifiers. Active only when Streak == 0 (last result was a draw)
# to prevent double-counting with STREAK_MODIFIERS["loss"].
#
# Audas et al. (2002): negative persistence weakens when draws dilute the
# losing run. Vergin (2000): extra-streak effect converges to zero for long
# runs; mean reversion starts earlier in mixed sequences.
# Cohen-Zada et al. (2017): draws break the precipitating-event chain.
WINLESS_STREAK_MODIFIERS: dict = {
    1: 1.000,       # 1-2 draws: no alarm
    2: 1.000,
    3: 0.992,       # Tactical stagnation begins
    4: 0.992,
    5: 0.985,       # Goal drought erodes confidence
    6: 0.985,
    7: 0.982,       # Stagnation peak
    8: 0.982,
    "9+": 0.988,    # Audas et al. negative persistence → slight regression
}


# ---------------------------------------------------------------------------
# Red-card simulation constants
# ---------------------------------------------------------------------------
# Červený, van Ours & van Tuijl (2018, Empirical Economics):
#   94 red cards, hazard-rate model; average red-card minute = 65;
#   none in the first 15 minutes; sanctioned team scores ~75 % of base rate;
#   opponent scores ~124 %.
# Badiella et al. (2022, Annals of Operations Research):
#   5 leagues, 1 826 matches; away red-card effect stable across the match.
# RunRepeat (2021, 19 985 matches): 37 % of reds in final 15 min, 12 % in
#   first 30 min. Triangular(15, 65, 90) captures this distribution.
# Titman et al. (2015, JRSS): yellow cards do not affect scoring rate →
#   not modelled here.
RC_TRIANGULAR_MIN:  float = 15.0   # No cards before minute 15 (Červený 2018)
RC_TRIANGULAR_MODE: float = 65.0   # Modal red-card minute
RC_TRIANGULAR_MAX:  float = 90.0   # End of normal time
RC_SANCTIONED_COEF: float = 0.25   # Max lambda reduction for sanctioned team
RC_OPPONENT_COEF:   float = 0.15   # Max lambda boost for the opposing team


# ---------------------------------------------------------------------------
# Dixon-Coles ρ correction
# ---------------------------------------------------------------------------
# Dixon & Coles (1997, JRSS-C): independent Poisson underestimates the
# frequency of 0-0 and 1-1 scorelines. The τ correction applies only to
# score pairs (0,0), (1,0), (0,1), (1,1); all others leave τ = 1.0.
#
# ρ < 0 (PL calibration):
#   τ(0,0) = 1 − λ_h·λ_a·ρ  → increases with negative ρ  ✓
#   τ(1,1) = 1 − ρ            → > 1 with negative ρ         ✓
#   τ(1,0) = 1 + λ_a·ρ        → decreases (balancing)
# Set to 0.0 to use standard independent Poisson.
DC_RHO: float = -0.065


# ---------------------------------------------------------------------------
# Rivalry matrix
# ---------------------------------------------------------------------------

DEFAULT_RIVALRY_MATRIX: dict[tuple[str, str], MatchType] = {
    # Local derbies
    ("Arsenal",           "Tottenham-Hotspur"): MatchType.DERBY,
    ("Manchester-City",   "Manchester-United"): MatchType.DERBY,
    ("Liverpool",         "Everton"):           MatchType.DERBY,
    ("Chelsea",           "Arsenal"):           MatchType.DERBY,
    ("Chelsea",           "Tottenham-Hotspur"): MatchType.DERBY,

    # Top-6 fixtures
    ("Manchester-City",   "Liverpool"):         MatchType.TOP6,
    ("Manchester-City",   "Chelsea"):           MatchType.TOP6,
    ("Manchester-City",   "Arsenal"):           MatchType.TOP6,
    ("Manchester-City",   "Tottenham-Hotspur"): MatchType.TOP6,
    ("Manchester-United", "Liverpool"):         MatchType.TOP6,
    ("Manchester-United", "Chelsea"):           MatchType.TOP6,
    ("Manchester-United", "Arsenal"):           MatchType.TOP6,
    ("Manchester-United", "Tottenham-Hotspur"): MatchType.TOP6,
    ("Liverpool",         "Chelsea"):           MatchType.TOP6,
    ("Liverpool",         "Arsenal"):           MatchType.TOP6,
    ("Liverpool",         "Tottenham-Hotspur"): MatchType.TOP6,
}


# ---------------------------------------------------------------------------
# MatchEngine
# ---------------------------------------------------------------------------

class MatchEngine:
    """
    Simulate Premier League matches with a Poisson goal model.

    The engine layers the following adjustments on top of the base Poisson draw:

    1. **Home advantage** — multiplicative lambda modifier (default ×1.10 / ×0.90).
    2. **Streak modifier** — small momentum signal for win/loss runs
       (Gilovich et al. 1985; Audas et al. 2002).
    3. **Winless-streak modifier** — softer version for mixed draw/loss runs;
       mutually exclusive with the streak modifier to prevent double-counting.
    4. **Derby compression** — narrows the lambda gap in local derbies and
       Top-6 fixtures.
    5. **Red-card simulation** — team-specific probability; time-weighted
       lambda adjustment (Červený et al. 2018).
    6. **Dixon-Coles τ correction** — corrects under-frequency of 0-0 and
       1-1 scorelines (Dixon & Coles 1997).
    7. **Lineup sampling** — when ``player_rosters`` is supplied, 11 players
       are drawn per team each match; normalised scores produce a multiplicative
       factor around 1.0, preserving season-level expected goals.

    Parameters
    ----------
    teams_data : pd.DataFrame
        Must contain columns: ``Club``, ``AvgAttack``, ``AvgDefense``.
        ``AvgAttack``  — expected goals scored per match (higher = stronger offence).
        ``AvgDefense`` — expected goals conceded per match (higher = weaker defence).
    home_advantage : float, optional
        Home-team lambda multiplier. Away multiplier = 2.0 − home_advantage.
        Default is 1.10 (Nevill & Holder 1999).
    rivalry_matrix : dict or None, optional
        Mapping of ``(team_a, team_b)`` to ``MatchType``. Looked up
        bi-directionally. Pass ``{}`` to disable derby compression entirely.
        ``None`` uses ``DEFAULT_RIVALRY_MATRIX``.
    streak_modifiers : dict or None, optional
        Override for ``STREAK_MODIFIERS``. ``None`` uses the module default.
    winless_streak_modifiers : dict or None, optional
        Override for ``WINLESS_STREAK_MODIFIERS``. ``None`` uses the module default.
    player_rosters : pd.DataFrame or None, optional
        Contents of ``player_scores.csv``. Required columns:
        ``Club``, ``Appearances``, ``attack_score_norm``, ``defense_score_norm``.
        ``None`` disables lineup sampling.
    dc_rho : float, optional
        Dixon-Coles ρ parameter. Default ``DC_RHO``. Set to 0.0 for
        standard independent Poisson.
    """

    def __init__(
        self,
        teams_data: pd.DataFrame,
        home_advantage: float = 1.10,
        rivalry_matrix: Optional[dict[tuple[str, str], MatchType]] = None,
        streak_modifiers: Optional[dict] = None,
        winless_streak_modifiers: Optional[dict] = None,
        player_rosters: Optional[pd.DataFrame] = None,
        dc_rho: float = DC_RHO,
    ):
        self.teams: dict[str, dict[str, float]] = {
            row["Club"]: {
                "attack":   row["AvgAttack"],
                "defense":  row["AvgDefense"],
                # RedRate: team-specific red-card probability per match.
                # Falls back to the PL average if the column is absent.
                "red_rate": row.get("RedRate", 0.0705),
            }
            for _, row in teams_data.iterrows()
        }

        self.home_advantage = home_advantage
        self.away_advantage = 2.0 - home_advantage  # Symmetric: 1.10 → 0.90

        self.rivalry_matrix: dict[tuple[str, str], MatchType] = (
            DEFAULT_RIVALRY_MATRIX if rivalry_matrix is None else rivalry_matrix
        )
        self.streak_modifiers: dict = (
            STREAK_MODIFIERS if streak_modifiers is None else streak_modifiers
        )
        self.winless_streak_modifiers: dict = (
            WINLESS_STREAK_MODIFIERS
            if winless_streak_modifiers is None
            else winless_streak_modifiers
        )
        self.dc_rho: float = dc_rho

        # Pre-process player rosters into per-team numpy arrays for fast
        # sampling during match simulation.
        self.rosters: dict[str, dict[str, np.ndarray]] = {}
        if player_rosters is not None:
            self._build_roster_index(player_rosters)

        self._reset_standings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_season(self) -> pd.DataFrame:
        """
        Simulate a full double round-robin season (20 teams → 380 matches).

        Returns
        -------
        pd.DataFrame
            Final league table sorted by points, goal difference, goals scored.
        """
        self._reset_standings()
        teams = list(self.teams.keys())
        for home in teams:
            for away in teams:
                if home != away:
                    self._play_match(home, away)
        return self._build_table()

    # ------------------------------------------------------------------
    # Internal — match simulation
    # ------------------------------------------------------------------

    def _play_match(self, home: str, away: str) -> None:
        """Simulate one match and update the standings."""

        # 1. Base lambda: blend of attack strength and opponent defence weakness.
        #    AvgDefense = goals conceded, so high value means weaker defence.
        base_home = (self.teams[home]["attack"] + self.teams[away]["defense"]) / 2
        base_away = (self.teams[away]["attack"] + self.teams[home]["defense"]) / 2

        # 2. Home advantage + two-layer momentum adjustment.
        #    _streak_modifier and _winless_streak_modifier are mutually
        #    exclusive (active when Streak != 0 and Streak == 0, respectively),
        #    preventing double-counting.
        home_lambda = (
            base_home * self.home_advantage
            * self._streak_modifier(home)
            * self._winless_streak_modifier(home)
        )
        away_lambda = (
            base_away * self.away_advantage
            * self._streak_modifier(away)
            * self._winless_streak_modifier(away)
        )

        # 3. Lineup sampling (when roster data is available).
        #    Each factor has E = 1.0, so the season-level goal budget is preserved.
        #    Formula: home_lambda × (home_atk_factor / away_def_factor)
        if self.rosters:
            home_atk, away_def = self._lineup_factors(home, away)
            away_atk, home_def = self._lineup_factors(away, home)
            home_lambda *= home_atk / max(away_def, 0.30)
            away_lambda *= away_atk / max(home_def, 0.30)

        # 4. Derby / Top-6 lambda compression.
        home_lambda, away_lambda = self._apply_rivalry_compression(
            home, away, home_lambda, away_lambda
        )

        # 5. Red-card events (independent for each team).
        home_rc, home_rc_min = self._trigger_red_card(self.teams[home]["red_rate"])
        away_rc, away_rc_min = self._trigger_red_card(self.teams[away]["red_rate"])

        if home_rc:
            home_lambda, away_lambda = self._apply_red_card_impact(
                home_lambda, away_lambda, home_rc_min
            )
        if away_rc:
            away_lambda, home_lambda = self._apply_red_card_impact(
                away_lambda, home_lambda, away_rc_min
            )

        # 6. Score draw with optional Dixon-Coles correction.
        home_goals, away_goals = self._draw_scores(home_lambda, away_lambda)

        self._update_standings(home, away, home_goals, away_goals)

    # ------------------------------------------------------------------
    # Internal — lineup sampling
    # ------------------------------------------------------------------

    # Minimum appearances for a player to be eligible for lineup sampling.
    # Fewer than 5 matches produces highly variable per-match ratios.
    MIN_LINEUP_APPEARANCES: int = 5

    # Cap on normalised scores. Values above 3.5× the team mean are treated
    # as statistical outliers (e.g. 2 goals in 2 appearances).
    MAX_NORM_CLIP: float = 3.5

    # Damping factor applied to the raw lineup deviation.
    # Dawson et al. (2000, IJSEP) and Carmichael et al. (2001, BJIR) show that
    # individual player variation has a limited causal effect on match outcome;
    # team quality is the dominant predictor. 0.35 retains 35 % of the raw
    # variance while dampening extreme single-match fluctuations.
    LINEUP_DAMPING: float = 0.35

    def _build_roster_index(self, df: pd.DataFrame) -> None:
        """
        Convert the player-scores DataFrame into per-team numpy arrays.

        Filters applied
        ---------------
        1. ``MIN_LINEUP_APPEARANCES`` (≥ 5): removes low-sample outliers.
        2. ``MAX_NORM_CLIP`` (≤ 3.5): trims extreme normalised values.
        3. Re-normalisation: after clipping, the appearance-weighted mean
           is rescaled to 1.0 to maintain ``E[lineup_factor] = 1.0``.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: Club, Appearances, attack_score_norm,
            defense_score_norm.

        Raises
        ------
        ValueError
            If required columns are missing from ``df``.
        """
        required = {"Club", "Appearances", "attack_score_norm", "defense_score_norm"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"player_rosters must contain: {required}\n"
                f"Found: {set(df.columns)}"
            )

        for club, grp in df.groupby("Club"):
            active = grp[grp["Appearances"] >= self.MIN_LINEUP_APPEARANCES].copy()
            active = active.reset_index(drop=True)
            if len(active) < 11:
                continue  # Not enough eligible players; skip lineup sampling

            app   = active["Appearances"].values.astype(float)
            probs = app / app.sum()

            # Clip extreme outliers.
            atk_raw = np.clip(
                active["attack_score_norm"].values.astype(float), 0.0, self.MAX_NORM_CLIP
            )
            def_raw = np.clip(
                active["defense_score_norm"].values.astype(float), 0.0, self.MAX_NORM_CLIP
            )

            # Re-normalise so the appearance-weighted mean remains 1.0.
            atk_mean = float(np.dot(atk_raw, probs))
            def_mean = float(np.dot(def_raw, probs))
            atk_norm = atk_raw / atk_mean if atk_mean > 1e-9 else np.ones_like(atk_raw)
            def_norm = def_raw / def_mean if def_mean > 1e-9 else np.ones_like(def_raw)

            self.rosters[club] = {
                "probs":         probs,
                "attack_norms":  atk_norm,
                "defense_norms": def_norm,
            }

    def _lineup_factors(
        self, attacking_team: str, defending_team: str
    ) -> tuple[float, float]:
        """
        Sample 11 players per side and return dampened lineup factors.

        Damping formula::

            raw_factor   = mean(sampled_11_scores)        # E = 1.0
            final_factor = 1.0 + (raw_factor - 1.0) * LINEUP_DAMPING

        Parameters
        ----------
        attacking_team : str
            Team name for the attack side.
        defending_team : str
            Team name for the defence side.

        Returns
        -------
        tuple of float
            ``(attack_factor, defense_factor)``, each with E = 1.0.
        """
        d = self.LINEUP_DAMPING

        if attacking_team in self.rosters:
            r   = self.rosters[attacking_team]
            idx = np.random.choice(len(r["probs"]), size=11, replace=False, p=r["probs"])
            raw = float(np.mean(r["attack_norms"][idx]))
            atk_factor = 1.0 + (raw - 1.0) * d
        else:
            atk_factor = 1.0

        if defending_team in self.rosters:
            r   = self.rosters[defending_team]
            idx = np.random.choice(len(r["probs"]), size=11, replace=False, p=r["probs"])
            raw = float(np.mean(r["defense_norms"][idx]))
            def_factor = 1.0 + (raw - 1.0) * d
        else:
            def_factor = 1.0

        return atk_factor, def_factor

    # ------------------------------------------------------------------
    # Internal — score generation
    # ------------------------------------------------------------------

    def _draw_scores(
        self, home_lambda: float, away_lambda: float
    ) -> tuple[int, int]:
        """
        Draw a match scoreline using optional Dixon-Coles τ correction.

        When ``dc_rho == 0.0`` this reduces to two independent Poisson draws.
        Otherwise, a joint probability matrix is built and the correction is
        applied to the top-left 2×2 block before sampling.

        Algorithm
        ---------
        1. Compute marginal Poisson probability vectors up to ``MAX_GOALS``.
        2. Outer-product to form the independent joint matrix.
        3. Apply τ correction to score pairs (0,0), (1,0), (0,1), (1,1).
        4. Re-normalise the matrix (τ shifts probability mass slightly).
        5. Sample one (home_goals, away_goals) pair.

        Parameters
        ----------
        home_lambda : float
            Expected goals for the home team.
        away_lambda : float
            Expected goals for the away team.

        Returns
        -------
        tuple of int
            ``(home_goals, away_goals)``
        """
        MAX_GOALS = 10  # P(goals > 10) is negligible

        if self.dc_rho == 0.0:
            return (
                int(np.random.poisson(home_lambda)),
                int(np.random.poisson(away_lambda)),
            )

        h_probs = np.array([
            np.exp(-home_lambda) * (home_lambda ** g) / math.factorial(g)
            for g in range(MAX_GOALS + 1)
        ])
        a_probs = np.array([
            np.exp(-away_lambda) * (away_lambda ** g) / math.factorial(g)
            for g in range(MAX_GOALS + 1)
        ])

        matrix = np.outer(h_probs, a_probs)

        # Apply Dixon-Coles τ to the low-score block only.
        rho = self.dc_rho
        matrix[0, 0] *= max(1.0 - home_lambda * away_lambda * rho, 1e-10)
        matrix[0, 1] *= 1.0 + home_lambda * rho
        matrix[1, 0] *= 1.0 + away_lambda * rho
        matrix[1, 1] *= max(1.0 - rho, 1e-10)

        total = matrix.sum()
        if total > 0:
            matrix /= total

        flat       = matrix.flatten()
        idx        = np.random.choice(len(flat), p=flat)
        home_goals = idx // (MAX_GOALS + 1)
        away_goals = idx %  (MAX_GOALS + 1)

        return int(home_goals), int(away_goals)

    # ------------------------------------------------------------------
    # Internal — red card
    # ------------------------------------------------------------------

    def _trigger_red_card(self, red_rate: float) -> tuple[bool, float]:
        """
        Determine whether a team receives a red card and, if so, when.

        Minute is drawn from Triangular(15, 65, 90) following Červený et al.
        (2018): no cards in the first 15 minutes; mode at minute 65.

        Parameters
        ----------
        red_rate : float
            Per-match red-card probability for the team (from ``team_stats.csv``).

        Returns
        -------
        tuple
            ``(card_occurred: bool, minute: float)``
        """
        if np.random.random() < red_rate:
            minute = np.random.triangular(
                RC_TRIANGULAR_MIN,
                RC_TRIANGULAR_MODE,
                RC_TRIANGULAR_MAX,
            )
            return True, float(minute)
        return False, 0.0

    def _apply_red_card_impact(
        self,
        sanctioned_lambda: float,
        opponent_lambda:   float,
        minute:            float,
    ) -> tuple[float, float]:
        """
        Apply a time-weighted lambda adjustment after a red card.

        Formula (weighted average, not the original scaling proposal)::

            impact           = (90 − minute) / 90     # remaining time fraction
            sanctioned_lambda *= (1 − impact * 0.25)  # up to −25 % (full match)
            opponent_lambda  *= (1 + impact * 0.15)   # up to +15 % (full match)

        This avoids the mathematical error in the naive (90/minute × λ) formula,
        which triples lambda at minute 30.

        Empirical basis: Červený et al. (2018) — sanctioned team ~75 % of base
        scoring rate; opponent ~124 %. Badiella et al. (2022) — effect is
        time-dependent.

        Limitation: Červený (2018) finds the highest impact between minutes 45
        and 75; this linear model underestimates that non-linearity.

        Parameters
        ----------
        sanctioned_lambda : float
            Goal-scoring lambda for the team that received the red card.
        opponent_lambda : float
            Goal-scoring lambda for the opposing team.
        minute : float
            Match minute at which the red card was shown.

        Returns
        -------
        tuple of float
            ``(new_sanctioned_lambda, new_opponent_lambda)``
        """
        impact = float(np.clip((RC_TRIANGULAR_MAX - minute) / RC_TRIANGULAR_MAX, 0.0, 1.0))
        new_sanctioned = sanctioned_lambda * (1.0 - impact * RC_SANCTIONED_COEF)
        new_opponent   = opponent_lambda   * (1.0 + impact * RC_OPPONENT_COEF)
        return new_sanctioned, new_opponent

    # ------------------------------------------------------------------
    # Internal — rivalry compression
    # ------------------------------------------------------------------

    def _apply_rivalry_compression(
        self,
        home: str,
        away: str,
        home_lambda: float,
        away_lambda: float,
    ) -> tuple[float, float]:
        """
        Compress the lambda gap in derby and Top-6 fixtures.

        "Strong" and "weak" sides are determined from the already-adjusted
        lambdas (including home advantage), so the compression operates on the
        effective match-day balance rather than raw team ratings.

        Parameters
        ----------
        home : str
            Home team name.
        away : str
            Away team name.
        home_lambda : float
            Home team's adjusted lambda before compression.
        away_lambda : float
            Away team's adjusted lambda before compression.

        Returns
        -------
        tuple of float
            ``(home_lambda, away_lambda)`` after compression.
        """
        match_type = self._get_match_type(home, away)
        if match_type == MatchType.NORMAL:
            return home_lambda, away_lambda

        strong_mod, weak_mod = DERBY_COEFFICIENTS[match_type]

        if home_lambda >= away_lambda:
            home_lambda *= strong_mod
            away_lambda *= weak_mod
        else:
            away_lambda *= strong_mod
            home_lambda *= weak_mod

        return home_lambda, away_lambda

    # ------------------------------------------------------------------
    # Internal — streak modifiers
    # ------------------------------------------------------------------

    def _streak_modifier(self, team: str) -> float:
        """
        Return the momentum multiplier for pure win / loss streaks.

        Returns 1.0 (neutral) when the last result was a draw (Streak == 0).
        The modifier is looked up from ``STREAK_MODIFIERS``; integer keys are
        matched exactly, and the string fallback (e.g. "6+") is used for
        streak lengths beyond the defined range.

        Parameters
        ----------
        team : str
            Team name.

        Returns
        -------
        float
            Lambda multiplier derived from ``STREAK_MODIFIERS``.
        """
        s = self.standings[team]["Streak"]
        if s == 0:
            return 1.0

        direction = "win" if s > 0 else "loss"
        length    = abs(s)
        table     = self.streak_modifiers[direction]

        if length in table:
            return table[length]
        fallback_key = next(k for k in table if isinstance(k, str))
        return table[fallback_key]

    def _winless_streak_modifier(self, team: str) -> float:
        """
        Return the modifier for a mixed draw/loss winless streak.

        Active only when ``Streak == 0`` (last result was a draw). When
        ``Streak != 0`` (pure win or loss run), this method returns 1.0 so
        that ``_streak_modifier`` handles the adjustment exclusively —
        preventing double-counting.

        Parameters
        ----------
        team : str
            Team name.

        Returns
        -------
        float
            Lambda multiplier derived from ``WINLESS_STREAK_MODIFIERS``.
            Returns 1.0 if the streak modifier is already active.
        """
        if self.standings[team]["Streak"] != 0:
            return 1.0  # Pure win/loss run: defer to _streak_modifier

        w = self.standings[team]["WinlessStreak"]
        if w == 0:
            return 1.0

        table = self.winless_streak_modifiers
        if w in table:
            return table[w]
        fallback_key = next(k for k in table if isinstance(k, str))
        return table[fallback_key]

    # ------------------------------------------------------------------
    # Internal — rivalry lookup
    # ------------------------------------------------------------------

    def _get_match_type(self, t1: str, t2: str) -> MatchType:
        """Look up match type in both directions of the rivalry matrix."""
        return (
            self.rivalry_matrix.get((t1, t2))
            or self.rivalry_matrix.get((t2, t1))
            or MatchType.NORMAL
        )

    # ------------------------------------------------------------------
    # Internal — standings
    # ------------------------------------------------------------------

    def _reset_standings(self) -> None:
        self.standings: dict[str, dict] = {
            team: {
                "PTS": 0, "W": 0, "D": 0, "L": 0,
                "GF": 0, "GA": 0,
                "Streak": 0,            # Positive = win run, negative = loss run; 0 on draw
                "WinlessStreak": 0,     # Consecutive matches without a win
                "PeakWinStreak": 0,     # Longest win streak this season
                "PeakLossStreak": 0,    # Longest loss streak this season
                "PeakWinlessStreak": 0, # Longest winless run this season
            }
            for team in self.teams
        }

    def _update_standings(
        self, home: str, away: str, hg: int, ag: int
    ) -> None:
        self.standings[home]["GF"] += hg
        self.standings[home]["GA"] += ag
        self.standings[away]["GF"] += ag
        self.standings[away]["GA"] += hg

        if hg > ag:
            self._record_win(home)
            self._record_loss(away)
        elif ag > hg:
            self._record_win(away)
            self._record_loss(home)
        else:
            # Draw: resets the pure win/loss streak but continues the winless run.
            for team in (home, away):
                self.standings[team]["PTS"] += 1
                self.standings[team]["D"]   += 1
                self.standings[team]["Streak"] = 0
                self.standings[team]["WinlessStreak"] += 1
                self.standings[team]["PeakWinlessStreak"] = max(
                    self.standings[team]["PeakWinlessStreak"],
                    self.standings[team]["WinlessStreak"],
                )

    def _record_win(self, team: str) -> None:
        s = self.standings[team]
        s["PTS"] += 3
        s["W"]   += 1
        s["Streak"]        = max(1, s["Streak"] + 1)
        s["WinlessStreak"] = 0  # Win resets the winless streak
        s["PeakWinStreak"] = max(s["PeakWinStreak"], s["Streak"])

    def _record_loss(self, team: str) -> None:
        s = self.standings[team]
        s["L"]             += 1
        s["Streak"]        = min(-1, s["Streak"] - 1)
        s["WinlessStreak"] += 1
        s["PeakLossStreak"]    = max(s["PeakLossStreak"],    abs(s["Streak"]))
        s["PeakWinlessStreak"] = max(s["PeakWinlessStreak"], s["WinlessStreak"])

    def _build_table(self) -> pd.DataFrame:
        table = pd.DataFrame.from_dict(self.standings, orient="index")
        table["GD"] = table["GF"] - table["GA"]
        return table.sort_values(["PTS", "GD", "GF"], ascending=False)
