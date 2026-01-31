#!/usr/bin/env python3
"""
Stabilization Layer for Pair Candidates
Implements entry/stay/drop logic with persistence tracking.
Supports multiple universes.
"""

import argparse
import os
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stabilize candidate pairs with entry/stay/drop logic"
    )
    parser.add_argument(
        "--universe",
        type=str,
        default=None,
        help="Universe name (e.g., tech_core, semis, software, etc.)",
    )
    parser.add_argument(
        "--in-csv",
        type=str,
        required=True,
        help="Path to today's candidates CSV",
    )
    parser.add_argument(
        "--state",
        type=str,
        required=True,
        help="Path to history state file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for stabilized results",
    )
    return parser.parse_args()


def load_or_create_history(filepath: str) -> pd.DataFrame:
    """Load history state or create empty DataFrame if doesn't exist."""
    if os.path.exists(filepath):
        print(f"Loading existing history from {filepath}")
        df = pd.read_csv(filepath)
        # Ensure required columns exist
        required_cols = [
            "ticker1", "ticker2", "rolling_class", "persistence_days",
            "first_seen_date", "last_seen_date", "drop_count"
        ]
        for col in required_cols:
            if col not in df.columns:
                if col == "persistence_days":
                    df[col] = 0
                elif col == "drop_count":
                    df[col] = 0
                elif col in ["first_seen_date", "last_seen_date"]:
                    df[col] = ""
                else:
                    df[col] = "none"
        return df
    else:
        print(f"No existing history found at {filepath}, creating new")
        return pd.DataFrame(columns=[
            "ticker1", "ticker2", "grade", "score_w",
            "corr_mean", "corr_60", "corr_90",
            "spearman_60", "spearman_90",
            "resid_corr_60", "resid_corr_90",
            "coint_pvalue_best", "corr_hitrate_30d_6m",
            "rolling_class", "persistence_days",
            "first_seen_date", "last_seen_date", "drop_count"
        ])


def check_entry_criteria(row: pd.Series) -> str:
    """Check if pair meets ENTRY criteria (A or B+)."""
    # Grade A criteria
    resid_max = max(row.get("resid_corr_60", 0), row.get("resid_corr_90", 0))
    
    if (
        row.get("corr_mean", 0) >= 0.82
        and row.get("coint_pvalue_best", 1.0) <= 0.05
        and row.get("corr_hitrate_30d_6m", 0) >= 0.70
        and resid_max >= 0.60
    ):
        return "A"
    
    # Grade B+ criteria
    corr_condition = (
        row.get("corr_mean", 0) >= 0.78 or 
        (row.get("corr_60", 0) >= 0.80 and row.get("corr_90", 0) >= 0.76)
    )
    
    if (
        corr_condition
        and row.get("coint_pvalue_best", 1.0) <= 0.12
        and row.get("spearman_60", 0) >= 0.75
        and row.get("spearman_90", 0) >= 0.75
        and resid_max >= 0.60
        and row.get("corr_hitrate_30d_6m", 0) >= 0.70
    ):
        return "B+"
    
    return "none"


def check_stay_criteria(row: pd.Series) -> bool:
    """Check if pair meets STAY criteria (relaxed thresholds)."""
    resid_max = max(row.get("resid_corr_60", 0), row.get("resid_corr_90", 0))
    
    return (
        row.get("corr_mean", 0) >= 0.78
        and row.get("coint_pvalue_best", 1.0) <= 0.14
        and row.get("corr_hitrate_30d_6m", 0) >= 0.65
        and resid_max >= 0.55
    )


def check_drop_criteria(row: pd.Series) -> bool:
    """Check if pair meets DROP criteria."""
    return (
        row.get("corr_mean", 1.0) < 0.75
        or row.get("coint_pvalue_best", 0) > 0.18
        or row.get("corr_hitrate_30d_6m", 1.0) < 0.50
    )


def determine_rolling_class(row: pd.Series) -> str:
    """Determine rolling class based on entry/stay/drop logic."""
    # First check ENTRY
    entry_grade = check_entry_criteria(row)
    if entry_grade in ["A", "B+"]:
        return entry_grade
    
    # Then check DROP
    if check_drop_criteria(row):
        return "drop"
    
    # Then check STAY
    if check_stay_criteria(row):
        return "stay"
    
    # Default to none if nothing matches
    return "none"


def update_persistence(
    current_class: str,
    previous_persistence: int,
    previous_drop_count: int,
    previous_class: str
) -> Tuple[int, int]:
    """Update persistence_days and drop_count based on rolling_class."""
    # If in A, B+, or stay - increment persistence, reset drop count
    if current_class in ["A", "B+", "stay"]:
        new_persistence = previous_persistence + 1
        new_drop_count = 0
    # If in drop - increment drop count, keep persistence for now
    elif current_class == "drop":
        new_drop_count = previous_drop_count + 1
        # After 3 consecutive drop days, reset persistence
        if new_drop_count >= 3:
            new_persistence = 0
        else:
            new_persistence = previous_persistence
    # If none - reset both
    else:
        new_persistence = 0
        new_drop_count = 0
    
    return new_persistence, new_drop_count


def stabilize_pairs(today_candidates: pd.DataFrame, history: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Main stabilization logic."""
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create pair key for matching
    today_candidates["pair_key"] = today_candidates.apply(
        lambda x: f"{min(x['ticker1'], x['ticker2'])}_{max(x['ticker1'], x['ticker2'])}", 
        axis=1
    )
    
    if not history.empty:
        history["pair_key"] = history.apply(
            lambda x: f"{min(x['ticker1'], x['ticker2'])}_{max(x['ticker1'], x['ticker2'])}", 
            axis=1
        )
    
    # Process each pair in today's candidates
    updated_pairs = []
    
    for _, today_row in today_candidates.iterrows():
        pair_key = today_row["pair_key"]
        
        # Check if pair exists in history
        if not history.empty and pair_key in history["pair_key"].values:
            hist_row = history[history["pair_key"] == pair_key].iloc[0]
            previous_persistence = hist_row["persistence_days"]
            previous_drop_count = hist_row.get("drop_count", 0)
            previous_class = hist_row["rolling_class"]
            first_seen = hist_row["first_seen_date"]
        else:
            previous_persistence = 0
            previous_drop_count = 0
            previous_class = "none"
            first_seen = today_date
        
        # Determine today's rolling class
        rolling_class = determine_rolling_class(today_row)
        
        # Update persistence and drop count
        persistence_days, drop_count = update_persistence(
            rolling_class,
            previous_persistence,
            previous_drop_count,
            previous_class
        )
        
        # Create updated row
        updated_row = {
            "ticker1": today_row["ticker1"],
            "ticker2": today_row["ticker2"],
            "grade": today_row.get("grade", ""),
            "score_w": today_row.get("score_w", 0),
            "corr_mean": today_row.get("corr_mean", np.nan),
            "corr_60": today_row.get("corr_60", np.nan),
            "corr_90": today_row.get("corr_90", np.nan),
            "spearman_60": today_row.get("spearman_60", np.nan),
            "spearman_90": today_row.get("spearman_90", np.nan),
            "resid_corr_60": today_row.get("resid_corr_60", np.nan),
            "resid_corr_90": today_row.get("resid_corr_90", np.nan),
            "coint_pvalue_best": today_row.get("coint_pvalue_best", np.nan),
            "corr_hitrate_30d_6m": today_row.get("corr_hitrate_30d_6m", np.nan),
            "rolling_class": rolling_class,
            "persistence_days": persistence_days,
            "first_seen_date": first_seen,
            "last_seen_date": today_date,
            "drop_count": drop_count,
            "pair_key": pair_key
        }
        
        updated_pairs.append(updated_row)
    
    # Also check for pairs in history that are not in today's scan
    if not history.empty:
        for _, hist_row in history.iterrows():
            pair_key = hist_row["pair_key"]
            if pair_key not in today_candidates["pair_key"].values:
                # Pair disappeared from scan - mark as drop
                drop_count = hist_row.get("drop_count", 0) + 1
                
                if drop_count >= 3:
                    persistence_days = 0
                else:
                    persistence_days = hist_row["persistence_days"]
                
                updated_row = {
                    "ticker1": hist_row["ticker1"],
                    "ticker2": hist_row["ticker2"],
                    "grade": hist_row.get("grade", ""),
                    "score_w": hist_row.get("score_w", 0),
                    "corr_mean": hist_row.get("corr_mean", np.nan),
                    "corr_60": hist_row.get("corr_60", np.nan),
                    "corr_90": hist_row.get("corr_90", np.nan),
                    "spearman_60": hist_row.get("spearman_60", np.nan),
                    "spearman_90": hist_row.get("spearman_90", np.nan),
                    "resid_corr_60": hist_row.get("resid_corr_60", np.nan),
                    "resid_corr_90": hist_row.get("resid_corr_90", np.nan),
                    "coint_pvalue_best": hist_row.get("coint_pvalue_best", np.nan),
                    "corr_hitrate_30d_6m": hist_row.get("corr_hitrate_30d_6m", np.nan),
                    "rolling_class": "drop",
                    "persistence_days": persistence_days,
                    "first_seen_date": hist_row["first_seen_date"],
                    "last_seen_date": hist_row["last_seen_date"],
                    "drop_count": drop_count,
                    "pair_key": pair_key
                }
                
                updated_pairs.append(updated_row)
    
    updated_df = pd.DataFrame(updated_pairs)
    
    # Separate into different categories
    
    # Stable pairs: rolling_class in {A, B+} AND persistence_days >= 5
    stable_pairs = updated_df[
        (updated_df["rolling_class"].isin(["A", "B+"])) &
        (updated_df["persistence_days"] >= 5)
    ].copy()
    
    # New candidates: rolling_class in {A, B+} AND persistence_days == 5
    new_candidates = updated_df[
        (updated_df["rolling_class"].isin(["A", "B+"])) &
        (updated_df["persistence_days"] == 5)
    ].copy()
    
    # Dropped pairs: drop_count >= 3
    dropped_pairs = updated_df[updated_df["drop_count"] >= 3].copy()
    
    # Clean up pair_key from output
    for df in [stable_pairs, new_candidates, dropped_pairs, updated_df]:
        if "pair_key" in df.columns:
            df.drop(columns=["pair_key"], inplace=True)
    
    return {
        "stable": stable_pairs,
        "new": new_candidates,
        "dropped": dropped_pairs,
        "history": updated_df
    }


def main():
    """Main execution function."""
    args = parse_args()
    
    universe_name = args.universe if args.universe else "unknown"
    print(f"\n{'='*60}")
    print(f"Stabilizing universe: {universe_name.upper()}")
    print(f"{'='*60}\n")
    
    # Load today's candidates
    if not os.path.exists(args.in_csv):
        print(f"ERROR: Input CSV not found: {args.in_csv}")
        # Create empty outputs to avoid workflow errors
        os.makedirs(args.out_dir, exist_ok=True)
        
        empty_df = pd.DataFrame(columns=[
            "ticker1", "ticker2", "grade", "score_w",
            "rolling_class", "persistence_days",
            "first_seen_date", "last_seen_date"
        ])
        
        if args.universe:
            empty_df.to_csv(os.path.join(args.out_dir, f"{args.universe}_stable.csv"), index=False)
            empty_df.to_csv(os.path.join(args.out_dir, f"{args.universe}_new_candidates.csv"), index=False)
            empty_df.to_csv(os.path.join(args.out_dir, f"{args.universe}_dropped_pairs.csv"), index=False)
        
        return
    
    today_candidates = pd.read_csv(args.in_csv)
    print(f"Loaded {len(today_candidates)} pairs from today's scan")
    
    # Load or create history
    history = load_or_create_history(args.state)
    print(f"History contains {len(history)} pairs")
    
    # Run stabilization
    results = stabilize_pairs(today_candidates, history)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Create state directory
    state_dir = os.path.dirname(args.state)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)
    
    # Determine file prefix
    prefix = args.universe if args.universe else "pairs"
    
    # Save stable pairs
    stable_path = os.path.join(args.out_dir, f"{prefix}_stable.csv")
    stable_cols = [
        "ticker1", "ticker2", "grade", "score_w",
        "rolling_class", "persistence_days",
        "first_seen_date", "last_seen_date"
    ]
    results["stable"][stable_cols].to_csv(stable_path, index=False)
    print(f"Saved {len(results['stable'])} stable pairs to {stable_path}")
    
    # Save new candidates
    new_path = os.path.join(args.out_dir, f"{prefix}_new_candidates.csv")
    results["new"][stable_cols].to_csv(new_path, index=False)
    print(f"Saved {len(results['new'])} new candidates to {new_path}")
    
    # Save dropped pairs
    dropped_path = os.path.join(args.out_dir, f"{prefix}_dropped_pairs.csv")
    dropped_cols = [
        "ticker1", "ticker2", "grade",
        "rolling_class", "persistence_days", "drop_count",
        "first_seen_date", "last_seen_date"
    ]
    results["dropped"][dropped_cols].to_csv(dropped_path, index=False)
    print(f"Saved {len(results['dropped'])} dropped pairs to {dropped_path}")
    
    # Update history state
    results["history"].to_csv(args.state, index=False)
    print(f"Updated history state at {args.state}")
    
    print(f"\nStabilization complete for {universe_name}!")
    print(f"  Stable pairs (ready for workflow 2): {len(results['stable'])}")
    print(f"  New candidates (persistence=5): {len(results['new'])}")
    print(f"  Dropped pairs (3+ consecutive drops): {len(results['dropped'])}")


if __name__ == "__main__":
    main()