#!/usr/bin/env python3
"""
run_all_and_aggregate_stable.py
================================
Aggregates results from all universes, applies stability logic,
generates combined rankings and watchlists.

Stability features:
- Tracks persistence_days (how long pair has been qualifying)
- Implements Entry/Stay/Drop hysteresis
- Maintains rolling_class state
- Generates new_candidates and dropped_pairs reports
"""

import warnings
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

UNIVERSES = ["tech_core", "semis", "software", "financials", "healthcare", "discretionary"]

RESULTS_DIR = Path("results")
PERSISTENCE_FILE = RESULTS_DIR / "persistence_history.csv"

MIN_PERSISTENCE_DAYS = 5
DROP_CONSECUTIVE_DAYS = 3

STAY = {
    "corr_mean": 0.78,
    "coint_pvalue": 0.14,
    "hitrate": 0.65,
    "residual_corr": 0.55
}

DROP = {
    "corr_mean": 0.75,
    "coint_pvalue": 0.18,
    "hitrate": 0.50
}


# ============================================================================
# STABILITY TRACKING
# ============================================================================

def load_persistence_history():
    """Load historical persistence data."""
    if PERSISTENCE_FILE.exists():
        df = pd.read_csv(PERSISTENCE_FILE, parse_dates=["first_seen_date", "last_seen_date"])
        return df
    else:
        return pd.DataFrame(columns=[
            "pair_id", "ticker1", "ticker2", "universe",
            "first_seen_date", "last_seen_date", 
            "persistence_days", "days_since_last_seen",
            "rolling_class", "drop_counter"
        ])


def check_stay_criteria(row):
    """Check if pair meets STAY thresholds."""
    return (
        row.get("corr_mean", 0) >= STAY["corr_mean"] and
        row.get("coint_pvalue_best", 1) <= STAY["coint_pvalue"] and
        row.get("corr_hitrate_30d_6m", 0) >= STAY["hitrate"] and
        row.get("residual_corr_90", 0) >= STAY["residual_corr"]
    )


def check_drop_criteria(row):
    """Check if pair meets DROP thresholds."""
    return (
        row.get("corr_mean", 0) < DROP["corr_mean"] or
        row.get("coint_pvalue_best", 1) > DROP["coint_pvalue"] or
        row.get("corr_hitrate_30d_6m", 0) < DROP["hitrate"]
    )


def update_persistence(current_candidates, history_df):
    """
    Update persistence tracking for all pairs.
    
    Logic:
    - New pairs: first_seen_date = today, persistence_days = 1
    - Existing pairs meeting STAY: increment persistence_days
    - Existing pairs failing DROP check: increment drop_counter
    - Pairs not in current scan: increment days_since_last_seen
    """
    today = datetime.now().date()
    
    # Create pair_id for current candidates
    current_candidates["pair_id"] = (
        current_candidates[["ticker1", "ticker2"]]
        .apply(lambda x: "_".join(sorted([x["ticker1"], x["ticker2"]])), axis=1)
    )
    
    # Merge with history
    if history_df.empty:
        # First run - all pairs are new
        new_history = current_candidates[["pair_id", "ticker1", "ticker2", "universe"]].copy()
        new_history["first_seen_date"] = today
        new_history["last_seen_date"] = today
        new_history["persistence_days"] = 1
        new_history["days_since_last_seen"] = 0
        new_history["rolling_class"] = current_candidates["entry_class"]
        new_history["drop_counter"] = 0
        return new_history
    
    # Update existing history
    updated_rows = []
    
    for _, row in history_df.iterrows():
        pair_id = row["pair_id"]
        current_match = current_candidates[current_candidates["pair_id"] == pair_id]
        
        if not current_match.empty:
            # Pair still in scan
            current_row = current_match.iloc[0]
            
            # Check if meets STAY criteria
            meets_stay = check_stay_criteria(current_row)
            
            # Check if meets DROP criteria
            meets_drop = check_drop_criteria(current_row)
            
            if meets_stay:
                # Continue tracking
                row["last_seen_date"] = today
                row["persistence_days"] += 1
                row["days_since_last_seen"] = 0
                row["rolling_class"] = current_row["entry_class"]
                row["drop_counter"] = 0
            elif meets_drop:
                # Increment drop counter
                row["last_seen_date"] = today
                row["drop_counter"] += 1
                row["days_since_last_seen"] = 0
                
                if row["drop_counter"] >= DROP_CONSECUTIVE_DAYS:
                    row["rolling_class"] = "drop"
        else:
            # Pair not in current scan
            row["days_since_last_seen"] += 1
        
        updated_rows.append(row)
    
    # Add new pairs
    existing_pair_ids = set(history_df["pair_id"])
    new_pairs = current_candidates[~current_candidates["pair_id"].isin(existing_pair_ids)]
    
    for _, new_row in new_pairs.iterrows():
        updated_rows.append({
            "pair_id": new_row["pair_id"],
            "ticker1": new_row["ticker1"],
            "ticker2": new_row["ticker2"],
            "universe": new_row["universe"],
            "first_seen_date": today,
            "last_seen_date": today,
            "persistence_days": 1,
            "days_since_last_seen": 0,
            "rolling_class": new_row["entry_class"],
            "drop_counter": 0
        })
    
    return pd.DataFrame(updated_rows)


def save_persistence_history(history_df):
    """Save updated persistence history."""
    PERSISTENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(PERSISTENCE_FILE, index=False)
    print(f"✓ Saved persistence history: {PERSISTENCE_FILE}")


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_all_universes():
    """Combine candidates from all universes."""
    all_candidates = []
    
    for universe in UNIVERSES:
        cand_file = RESULTS_DIR / universe / f"{universe}_candidates.csv"
        if cand_file.exists():
            df = pd.read_csv(cand_file)
            all_candidates.append(df)
            print(f"Loaded {len(df)} candidates from {universe}")
    
    if not all_candidates:
        print("WARNING: No candidate files found!")
        return pd.DataFrame()
    
    combined = pd.concat(all_candidates, ignore_index=True)
    print(f"\nTotal combined candidates: {len(combined)}")
    
    return combined


def generate_stable_lists(combined_df, history_df):
    """
    Apply stability filters to generate final lists.
    
    Returns:
    - stable_df: Pairs meeting persistence + not dropped
    - new_candidates_df: Pairs that just qualified
    - dropped_df: Pairs that were dropped
    """
    # Merge with history
    combined_df["pair_id"] = (
        combined_df[["ticker1", "ticker2"]]
        .apply(lambda x: "_".join(sorted([x["ticker1"], x["ticker2"]])), axis=1)
    )
    
    merged = combined_df.merge(
        history_df[["pair_id", "first_seen_date", "persistence_days", "rolling_class"]],
        on="pair_id",
        how="left"
    )
    
    # Fill NaN for new pairs
    merged["persistence_days"] = merged["persistence_days"].fillna(1)
    merged["rolling_class"] = merged["rolling_class"].fillna(merged["entry_class"])
    
    # Stable pairs: persistence >= MIN_PERSISTENCE_DAYS and not dropped
    stable = merged[
        (merged["persistence_days"] >= MIN_PERSISTENCE_DAYS) &
        (merged["rolling_class"].isin(["A", "B+"]))
    ].copy()
    
    # New candidates: persistence == MIN_PERSISTENCE_DAYS (just qualified)
    new_candidates = merged[
        (merged["persistence_days"] == MIN_PERSISTENCE_DAYS) &
        (merged["rolling_class"].isin(["A", "B+"]))
    ].copy()
    
    # Dropped pairs: from history where rolling_class == "drop"
    dropped = history_df[history_df["rolling_class"] == "drop"].copy()
    
    return stable, new_candidates, dropped


def save_combined_outputs(stable_df, new_candidates_df, dropped_df, combined_raw):
    """Save final output files."""
    
    # 1. Combined scored pairs (all A/B+)
    scored_file = RESULTS_DIR / "combined_pairs_scored.csv"
    combined_scored = combined_raw.sort_values("global_score", ascending=False)
    combined_scored.to_csv(scored_file, index=False)
    print(f"✓ Saved combined scored pairs: {scored_file} ({len(combined_scored)} pairs)")
    
    # 2. TOP 150 watchlist (from stable pairs)
    watchlist = stable_df.sort_values("global_score", ascending=False).head(150)
    watchlist_file = RESULTS_DIR / "combined_top150_watchlist.csv"
    watchlist.to_csv(watchlist_file, index=False)
    print(f"✓ Saved TOP 150 watchlist: {watchlist_file}")
    
    # 3. New candidates
    new_file = RESULTS_DIR / "new_candidates.csv"
    new_candidates_df.to_csv(new_file, index=False)
    print(f"✓ Saved new candidates: {new_file} ({len(new_candidates_df)} pairs)")
    
    # 4. Dropped pairs
    dropped_file = RESULTS_DIR / "dropped_pairs.csv"
    dropped_df.to_csv(dropped_file, index=False)
    print(f"✓ Saved dropped pairs: {dropped_file} ({len(dropped_df)} pairs)")
    
    # 5. Stable lists per universe
    for universe in UNIVERSES:
        universe_stable = stable_df[stable_df["universe"] == universe].copy()
        stable_file = RESULTS_DIR / universe / f"{universe}_stable.csv"
        stable_file.parent.mkdir(parents=True, exist_ok=True)
        universe_stable.to_csv(stable_file, index=False)
        print(f"  ✓ {universe}: {len(universe_stable)} stable pairs")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("AGGREGATING & STABILIZING RESULTS")
    print("="*60)
    
    # 1. Aggregate all candidates
    combined = aggregate_all_universes()
    
    if combined.empty:
        print("No candidates to process.")
        return
    
    # 2. Load persistence history
    history = load_persistence_history()
    print(f"\nLoaded persistence history: {len(history)} tracked pairs")
    
    # 3. Update persistence
    updated_history = update_persistence(combined, history)
    save_persistence_history(updated_history)
    
    # 4. Generate stable lists
    stable, new_candidates, dropped = generate_stable_lists(combined, updated_history)
    
    print(f"\nStability summary:")
    print(f"  - Stable pairs (persistence >= {MIN_PERSISTENCE_DAYS}): {len(stable)}")
    print(f"  - New candidates: {len(new_candidates)}")
    print(f"  - Dropped pairs: {len(dropped)}")
    
    # 5. Save outputs
    save_combined_outputs(stable, new_candidates, dropped, combined)
    
    print("\n✓ Aggregation complete!")


if __name__ == "__main__":
    main()