#!/usr/bin/env python3
"""
Aggregate stable pairs from all universes into global watchlists.
"""

import argparse
import os
import pandas as pd


UNIVERSES = ["tech_core", "semis", "software", "financials", "healthcare", "discretionary"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate stable pairs from all universes"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base results directory (default: results)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory for combined files (default: results)",
    )
    return parser.parse_args()


def load_universe_file(universe: str, results_dir: str, filename: str) -> pd.DataFrame:
    """Load a file from a specific universe, return empty DataFrame if not found."""
    filepath = os.path.join(results_dir, universe, filename)
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["universe"] = universe
        return df
    else:
        print(f"  WARNING: {filepath} not found, skipping")
        return pd.DataFrame()


def main():
    """Main execution function."""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("AGGREGATING STABLE PAIRS FROM ALL UNIVERSES")
    print(f"{'='*60}\n")
    
    # Aggregate stable pairs
    print("Loading stable pairs from all universes...")
    stable_dfs = []
    for universe in UNIVERSES:
        df = load_universe_file(universe, args.results_dir, f"{universe}_stable.csv")
        if not df.empty:
            stable_dfs.append(df)
            print(f"  {universe}: {len(df)} pairs")
    
    if stable_dfs:
        combined_stable = pd.concat(stable_dfs, ignore_index=True)
        print(f"\nTotal stable pairs across all universes: {len(combined_stable)}")
    else:
        print("No stable pairs found in any universe")
        combined_stable = pd.DataFrame()
    
    # Aggregate new candidates
    print("\nLoading new candidates from all universes...")
    new_dfs = []
    for universe in UNIVERSES:
        df = load_universe_file(universe, args.results_dir, f"{universe}_new_candidates.csv")
        if not df.empty:
            new_dfs.append(df)
            print(f"  {universe}: {len(df)} pairs")
    
    if new_dfs:
        combined_new = pd.concat(new_dfs, ignore_index=True)
        print(f"\nTotal new candidates across all universes: {len(combined_new)}")
    else:
        print("No new candidates found in any universe")
        combined_new = pd.DataFrame()
    
    # Aggregate dropped pairs
    print("\nLoading dropped pairs from all universes...")
    dropped_dfs = []
    for universe in UNIVERSES:
        df = load_universe_file(universe, args.results_dir, f"{universe}_dropped_pairs.csv")
        if not df.empty:
            dropped_dfs.append(df)
            print(f"  {universe}: {len(df)} pairs")
    
    if dropped_dfs:
        combined_dropped = pd.concat(dropped_dfs, ignore_index=True)
        print(f"\nTotal dropped pairs across all universes: {len(combined_dropped)}")
    else:
        print("No dropped pairs found in any universe")
        combined_dropped = pd.DataFrame()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save combined stable pairs (scored)
    if not combined_stable.empty:
        # Sort by grade (A > B+), then score_w desc
        grade_order = {"A": 0, "B+": 1}
        combined_stable["grade_order"] = combined_stable["grade"].map(grade_order)
        combined_stable = combined_stable.sort_values(
            by=["grade_order", "score_w"],
            ascending=[True, False]
        )
        combined_stable = combined_stable.drop(columns=["grade_order"])
        
        scored_path = os.path.join(args.out_dir, "combined_pairs_scored.csv")
        combined_stable.to_csv(scored_path, index=False)
        print(f"\nSaved combined scored pairs to {scored_path}")
        
        # Create top 150 watchlist
        top150 = combined_stable.head(150)
        watchlist_path = os.path.join(args.out_dir, "combined_top150_watchlist.csv")
        top150.to_csv(watchlist_path, index=False)
        print(f"Saved top 150 watchlist to {watchlist_path}")
        
        print(f"\nWatchlist breakdown:")
        print(f"  Grade A: {len(top150[top150['grade'] == 'A'])}")
        print(f"  Grade B+: {len(top150[top150['grade'] == 'B+'])}")
        
        for universe in UNIVERSES:
            count = len(top150[top150["universe"] == universe])
            if count > 0:
                print(f"  {universe}: {count} pairs")
    else:
        print("\nNo stable pairs to aggregate")
    
    # Save combined new candidates
    if not combined_new.empty:
        new_path = os.path.join(args.out_dir, "combined_new_candidates.csv")
        combined_new.to_csv(new_path, index=False)
        print(f"\nSaved combined new candidates to {new_path}")
    
    # Save combined dropped pairs
    if not combined_dropped.empty:
        dropped_path = os.path.join(args.out_dir, "combined_dropped_pairs.csv")
        combined_dropped.to_csv(dropped_path, index=False)
        print(f"Saved combined dropped pairs to {dropped_path}")
    
    print("\n" + "="*60)
    print("AGGREGATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()