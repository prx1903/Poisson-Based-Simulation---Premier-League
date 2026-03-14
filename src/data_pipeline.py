"""
data_pipeline.py — Feature Engineering for Premier League Simulation
=====================================================================

Transforms raw player-level career statistics into two team-level
feature tables consumed by MatchEngine:

  team_stats.csv    — Club, AvgAttack, AvgDefense, RedRate, YellowRate, FoulRate
  player_scores.csv — Per-player normalised attack / defence scores (for lineup sampling)

Design decisions vs. the naive baseline
----------------------------------------
1. Per-appearance normalisation instead of dividing by the squad's max appearances.
2. No arbitrary squad-size penalty (removed 22-player threshold and 2.5 % penalty).
3. Age-decay factor applied per player, not per team.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Position weights
# ---------------------------------------------------------------------------
# Attack weights reflect empirical PL goal-scoring share by position
# (Forward ~47 %, Midfielder ~37 %, Defender ~11 %, GK ~0 %).
# Midfielders are set below their raw share (0.55 vs ~0.79) because assists
# are already captured in the formula — full weight would double-count.
POSITION_ATTACK_WEIGHT: dict[str, float] = {
    "Goalkeeper": 0.00,
    "Defender":   0.15,
    "Midfielder": 0.55,
    "Forward":    1.00,
}

# Defence weights: GK is scored on clean-sheet rate; outfield players
# on tackles + interceptions + blocks per appearance.
POSITION_DEFENSE_WEIGHT: dict[str, float] = {
    "Goalkeeper": 1.00,
    "Defender":   0.60,
    "Midfielder": 0.20,
    "Forward":    0.00,
}

# ---------------------------------------------------------------------------
# Age-decay parameters
# ---------------------------------------------------------------------------
# Dendir (2016, European Journal of Sport Science): physical peak at 26-28,
# ~3-5 % annual decline after 30. Cumming & Ste-Marie (2001) support the
# general athlete performance curve. Rate set at 4 %/yr; floor at 70 %
# to account for experience buffering the physiological decline.
DECAY_START_AGE:  int   = 30
DECAY_RATE:       float = 0.04
DECAY_FLOOR:      float = 0.70

# PL historical average goals per match (2010-2020), used to rescale
# the weighted team attack score to a realistic lambda range.
PL_LEAGUE_AVG_GOALS_PER_MATCH: float = 1.36

DATA_PATH = Path("data/dataset - 2020-09-24.csv")

# ---------------------------------------------------------------------------
# Disciplinary constants
# ---------------------------------------------------------------------------
# Statathlon (2018, 15 200 matches, 4 major leagues): PL red-card rate
# per match = 0.141 → per team ≈ 0.0705.
# Titman et al. (2015, JRSS): yellow cards do not significantly affect
# scoring rate; red cards affect both teams.
PL_RED_CARD_RATE_PER_TEAM:    float = 0.0705
PL_YELLOW_CARD_RATE_PER_TEAM: float = 1.80


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_age_decay(age: float) -> float:
    """
    Return a performance decay multiplier based on player age.

    Applies no decay below ``DECAY_START_AGE`` (30). Above that threshold
    each additional year reduces the multiplier by ``DECAY_RATE`` (4 %),
    with a floor of ``DECAY_FLOOR`` (0.70) to reflect accumulated experience.

    Parameters
    ----------
    age : float
        Player age. NaN is treated as below the decay threshold (returns 1.0).

    Returns
    -------
    float
        Multiplier in [DECAY_FLOOR, 1.0].
    """
    if pd.isna(age) or age <= DECAY_START_AGE:
        return 1.0
    return max(DECAY_FLOOR, 1.0 - DECAY_RATE * (age - DECAY_START_AGE))


def compute_player_attack_raw(
    goals: float,
    assists: float,
    appearances: int,
    position: str,
    age: float,
) -> float:
    """
    Compute a player's raw attack contribution score.

    Formula::

        score = (goals + 0.5 * assists) / appearances
                * position_weight * age_decay

    Assists are weighted at 0.5 because they are a secondary contribution
    (the goal still depends on the receiving player's finish).

    Parameters
    ----------
    goals : float
        Career goal total. NaN treated as 0.
    assists : float
        Career assist total. NaN treated as 0.
    appearances : int
        Total career appearances. Must be > 0; returns 0.0 otherwise.
    position : str
        Player position key matching ``POSITION_ATTACK_WEIGHT``.
    age : float
        Player age used for decay calculation.

    Returns
    -------
    float
        Raw attack score (un-normalised).
    """
    if appearances <= 0:
        return 0.0
    g = goals   if not pd.isna(goals)   else 0.0
    a = assists if not pd.isna(assists) else 0.0
    pos_w = POSITION_ATTACK_WEIGHT.get(position, 0.0)
    decay = compute_age_decay(age)
    return (g + 0.5 * a) / appearances * pos_w * decay


def compute_player_defense_raw(
    clean_sheets: float,
    goals_conceded: float,
    tackles: float,
    interceptions: float,
    blocked_shots: float,
    appearances: int,
    position: str,
    age: float,
) -> float:
    """
    Compute a player's raw defence contribution score.

    Metric selection by position:

    - **Goalkeeper** — ``clean_sheets / appearances`` (primary quality signal;
      goals conceded is influenced by the entire team, not just the keeper).
    - **Defender / Midfielder** — ``(tackles + interceptions + blocked_shots)
      / appearances`` (defensive activity rate).
    - **Forward** — always 0.0 (negligible defensive contribution).

    Parameters
    ----------
    clean_sheets : float
        Career clean-sheet total (GK only). NaN treated as 0.
    goals_conceded : float
        Career goals-conceded total (unused; retained for API compatibility).
    tackles : float
        Career tackle total. NaN treated as 0.
    interceptions : float
        Career interception total. NaN treated as 0.
    blocked_shots : float
        Career blocked-shot total. NaN treated as 0.
    appearances : int
        Total career appearances. Must be > 0; returns 0.0 otherwise.
    position : str
        Player position key matching ``POSITION_DEFENSE_WEIGHT``.
    age : float
        Player age used for decay calculation.

    Returns
    -------
    float
        Raw defence score (un-normalised).
    """
    if appearances <= 0:
        return 0.0
    pos_w = POSITION_DEFENSE_WEIGHT.get(position, 0.0)
    if pos_w == 0.0:
        return 0.0
    decay = compute_age_decay(age)

    cs  = clean_sheets  if not pd.isna(clean_sheets)  else 0.0
    tck = tackles       if not pd.isna(tackles)       else 0.0
    icp = interceptions if not pd.isna(interceptions) else 0.0
    blk = blocked_shots if not pd.isna(blocked_shots) else 0.0

    if position == "Goalkeeper":
        raw = cs / appearances
    else:
        raw = (tck + icp + blk) / appearances

    return raw * pos_w * decay


# ---------------------------------------------------------------------------
# Team-level aggregation
# ---------------------------------------------------------------------------

def compute_team_attack(team_df: pd.DataFrame) -> float:
    """
    Compute the appearance-weighted raw attack score for one team.

    Each player's raw attack score is weighted by their share of the team's
    total appearances, so players with more experience carry more weight::

        weight_i = appearances_i / sum(appearances)
        team_attack = sum(attack_score_raw_i * weight_i)

    Parameters
    ----------
    team_df : pd.DataFrame
        Rows for a single club from the master dataset.

    Returns
    -------
    float
        Weighted raw attack score (before league-wide rescaling).
    """
    active = team_df[team_df["Appearances"] > 0].copy()
    if active.empty:
        return 0.0

    total_app = active["Appearances"].sum()
    weights   = active["Appearances"] / total_app

    raw_scores = active.apply(
        lambda r: compute_player_attack_raw(
            goals=r.get("Goals", 0),
            assists=r.get("Assists", 0),
            appearances=int(r["Appearances"]),
            position=r.get("Position", ""),
            age=r.get("Age", np.nan),
        ),
        axis=1,
    )
    return float((raw_scores * weights).sum())


def compute_team_defense(
    team_df: pd.DataFrame,
    league_avg_fallback: float = 1.56,
) -> float:
    """
    Compute the goalkeeper-anchored defence score for one team.

    The primary signal is the squad's starting goalkeeper (highest appearances
    among GKs). If no goalkeeper record exists, the league average is returned.

    Parameters
    ----------
    team_df : pd.DataFrame
        Rows for a single club from the master dataset.
    league_avg_fallback : float, optional
        Value returned when no goalkeeper data is available.

    Returns
    -------
    float
        Defence lambda (higher value = weaker defence = more goals conceded).
    """
    gks = team_df[team_df["Position"] == "Goalkeeper"].copy()
    if gks.empty:
        return league_avg_fallback

    # Use the keeper with the most appearances as the primary representative.
    gk = gks.sort_values("Appearances", ascending=False).iloc[0]
    age   = gk.get("Age", np.nan)
    apps  = int(gk["Appearances"]) if gk["Appearances"] > 0 else 1
    cs    = gk.get("Clean sheets", 0) if not pd.isna(gk.get("Clean sheets", 0)) else 0.0
    gc    = gk.get("Goals conceded", 0) if not pd.isna(gk.get("Goals conceded", 0)) else 0.0
    decay = compute_age_decay(age)

    # Clean-sheet rate as a quality proxy; goals-conceded rate as the raw
    # lambda estimate. Age decay applied to both.
    cs_rate = (cs / apps) * decay
    gc_rate = (gc / apps) / decay if decay > 0 else gc / apps

    # Blend: lower cs_rate → weaker keeper → higher defence lambda.
    defence_score = gc_rate * (1.0 - cs_rate * 0.5)
    return max(0.5, float(defence_score))


def compute_team_discipline(team_df: pd.DataFrame) -> dict[str, float]:
    """
    Derive a team's disciplinary profile from career statistics.

    Each metric is computed as an appearance-weighted average across the squad,
    so players with more game time have proportionally more influence.

    Red-card normalisation is deferred to ``run_pipeline`` where the
    league-wide mean is available.

    Parameters
    ----------
    team_df : pd.DataFrame
        Rows for a single club from the master dataset.

    Returns
    -------
    dict
        Keys: ``RedRate_raw``, ``YellowRate``, ``FoulRate``.
    """
    active = team_df[team_df["Appearances"] > 0].copy()
    if active.empty:
        return {
            "RedRate":    PL_RED_CARD_RATE_PER_TEAM,
            "YellowRate": PL_YELLOW_CARD_RATE_PER_TEAM,
            "FoulRate":   14.0,
        }

    total_app = active["Appearances"].sum()
    weights   = active["Appearances"] / total_app

    def safe_rate(col: str) -> float:
        vals = (
            active[col].fillna(0) if col in active.columns
            else pd.Series(0, index=active.index)
        )
        return float((vals / active["Appearances"] * weights).sum())

    return {
        "RedRate_raw":  round(safe_rate("Red cards"),    6),
        "YellowRate":   round(safe_rate("Yellow cards"), 4),
        "FoulRate":     round(safe_rate("Fouls"),        4),
    }


# ---------------------------------------------------------------------------
# Player-level normalised scores  (for lineup sampling in MatchEngine)
# ---------------------------------------------------------------------------

def build_player_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-player normalised attack and defence scores for lineup sampling.

    Each score is normalised so that the appearance-weighted team average
    equals 1.0. This ensures ``E[lineup_factor] = 1.0`` inside MatchEngine,
    preserving the season-level expected-goals budget while adding realistic
    match-by-match variance.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataset with one row per player.

    Returns
    -------
    pd.DataFrame
        Columns: Club, Name, Position, Appearances,
        attack_score_raw, defense_score_raw,
        attack_score_norm, defense_score_norm.
    """
    records = []

    for club, group in df.groupby("Club"):
        active = group[group["Appearances"] > 0].copy()
        if active.empty:
            continue

        total_app = active["Appearances"].sum()
        weights   = active["Appearances"] / total_app

        atk_raws, def_raws = [], []

        for _, row in active.iterrows():
            apps = int(row["Appearances"])
            pos  = row.get("Position", "")
            age  = row.get("Age", np.nan)

            atk = compute_player_attack_raw(
                goals=row.get("Goals", 0),
                assists=row.get("Assists", 0),
                appearances=apps,
                position=pos,
                age=age,
            )
            def_ = compute_player_defense_raw(
                clean_sheets=row.get("Clean sheets", 0),
                goals_conceded=row.get("Goals conceded", 0),
                tackles=row.get("Tackles", 0),
                interceptions=row.get("Interceptions", 0),
                blocked_shots=row.get("Blocked shots", 0),
                appearances=apps,
                position=pos,
                age=age,
            )
            atk_raws.append(atk)
            def_raws.append(def_)

        atk_array = np.array(atk_raws)
        def_array = np.array(def_raws)

        # Appearance-weighted team means for normalisation.
        w      = weights.values
        atk_mu = float(np.dot(atk_array, w))
        def_mu = float(np.dot(def_array, w))

        for i, (_, row) in enumerate(active.iterrows()):
            records.append({
                "Club":               club,
                "Name":               row.get("Name", ""),
                "Position":           row.get("Position", ""),
                "Appearances":        int(row["Appearances"]),
                "attack_score_raw":   round(atk_array[i], 6),
                "defense_score_raw":  round(def_array[i], 6),
                "attack_score_norm":  round(atk_array[i] / atk_mu, 4) if atk_mu > 1e-9 else 1.0,
                "defense_score_norm": round(def_array[i] / def_mu, 4) if def_mu > 1e-9 else 1.0,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    csv_path: Path = DATA_PATH,
    output_team_stats: str    = "team_stats.csv",
    output_player_scores: str = "player_scores.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full feature-engineering pipeline from raw CSV to model inputs.

    Steps
    -----
    1. Load and inspect raw data.
    2. Compute appearance-weighted raw attack score per team.
    3. Rescale team attack scores to match the PL historical goal average.
    4. Compute goalkeeper-anchored defence score per team.
    5. Compute team disciplinary profile; normalise red-card rates
       against the league mean.
    6. Compute per-player normalised scores for lineup sampling.
    7. Export ``team_stats.csv`` and ``player_scores.csv``.

    Parameters
    ----------
    csv_path : Path
        Path to the raw player dataset CSV.
    output_team_stats : str
        Output filename for the team statistics table.
    output_player_scores : str
        Output filename for the player scores table.

    Returns
    -------
    tuple of pd.DataFrame
        ``(team_stats_df, player_scores_df)``
    """
    print(f"[1/6] Loading data: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"      {len(df)} players across {df['Club'].nunique()} clubs")

    # Step 2 — Raw attack scores
    print("[2/6] Computing attack lambdas (appearance-weighted)...")
    raw_attacks = {
        club: compute_team_attack(group)
        for club, group in df.groupby("Club")
    }

    # Rescale to PL historical average goals per match.
    league_raw_mean = np.mean(list(raw_attacks.values()))
    scale_factor = (
        PL_LEAGUE_AVG_GOALS_PER_MATCH / league_raw_mean
        if league_raw_mean > 0 else 14.0
    )
    print(f"      Raw mean: {league_raw_mean:.4f} | Scale factor: {scale_factor:.2f}")
    scaled_attacks = {club: raw * scale_factor for club, raw in raw_attacks.items()}

    # Step 3 — Defence scores
    print("[3/6] Computing defence lambdas (goalkeeper-anchored)...")
    league_avg_defense = float(np.mean([
        compute_team_defense(grp, league_avg_fallback=1.56)
        for _, grp in df.groupby("Club")
    ]))
    team_defenses = {
        club: compute_team_defense(grp, league_avg_defense)
        for club, grp in df.groupby("Club")
    }

    # Step 4 — Disciplinary profiles
    print("[4/6] Computing disciplinary profiles (RedRate, YellowRate, FoulRate)...")
    team_disciplines = {
        club: compute_team_discipline(grp)
        for club, grp in df.groupby("Club")
    }

    # Normalise raw red-card rates against the league mean, then rescale
    # to the PL target rate (Statathlon 2018).
    raw_reds   = [d["RedRate_raw"] for d in team_disciplines.values()]
    league_raw = float(np.mean(raw_reds)) if np.mean(raw_reds) > 0 else 1e-6
    for disc in team_disciplines.values():
        scaled = (disc["RedRate_raw"] / league_raw) * PL_RED_CARD_RATE_PER_TEAM
        disc["RedRate"] = round(float(np.clip(
            scaled,
            PL_RED_CARD_RATE_PER_TEAM * 0.40,  # floor: ~0.028
            PL_RED_CARD_RATE_PER_TEAM * 2.00,  # ceiling: ~0.141
        )), 6)

    # Step 5 — Assemble team stats table
    print("[5/6] Assembling team statistics table...")
    team_stats_rows = []
    for club in sorted(df["Club"].unique()):
        disc = team_disciplines.get(club, {
            "RedRate":    PL_RED_CARD_RATE_PER_TEAM,
            "YellowRate": PL_YELLOW_CARD_RATE_PER_TEAM,
            "FoulRate":   14.0,
        })
        team_stats_rows.append({
            "Club":       club,
            "AvgAttack":  round(scaled_attacks.get(club,  PL_LEAGUE_AVG_GOALS_PER_MATCH), 4),
            "AvgDefense": round(team_defenses.get(club,   league_avg_defense),              4),
            "RedRate":    disc["RedRate"],
            "YellowRate": disc["YellowRate"],
            "FoulRate":   disc["FoulRate"],
        })
    team_stats_df = (
        pd.DataFrame(team_stats_rows)
        .sort_values("AvgAttack", ascending=False)
    )

    # Step 6 — Player scores for lineup sampling
    print("[6/6] Computing per-player normalised scores (lineup simulation)...")
    player_scores_df = build_player_scores(df)

    # Export
    team_stats_df.to_csv(output_team_stats, index=False)
    player_scores_df.to_csv(output_player_scores, index=False)
    print(f"\n✅ Outputs written:")
    print(f"   → {output_team_stats}    ({len(team_stats_df)} teams)")
    print(f"   → {output_player_scores} ({len(player_scores_df)} players)")
    print("\n📊 Team Statistics Summary:")
    print(team_stats_df.to_string(index=False))

    return team_stats_df, player_scores_df


if __name__ == "__main__":
    run_pipeline()
