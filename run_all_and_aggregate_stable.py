#!/usr/bin/env python3
"""
run_all_and_aggregate_stable.py
================================
Po uruchomieniu screeningu dla wszystkich 6 universe:
  1. Agreguje wyniki z candidates.csv
  2. Tworzy combined_pairs_scored.csv (wszystkie A/B+)
  3. Tworzy combined_top150_watchlist.csv (top 150 par wg global_score)
  4. Tworzy new_candidates.csv (pary z persistence_days <= 5)
  5. Tworzy dropped_pairs.csv (pary ze stable_class = "drop")
"""

from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
UNIVERSES = ["tech_core", "semis", "software", "financials", "healthcare", "discretionary"]


def aggregate_all():
    """Agreguje wyniki ze wszystkich universe."""
    all_candidates = []
    all_metrics = []
    
    for univ in UNIVERSES:
        cand_file = RESULTS_DIR / univ / f"{univ}_candidates.csv"
        all_file = RESULTS_DIR / univ / f"{univ}_all_metrics.csv"
        
        if cand_file.exists():
            df_cand = pd.read_csv(cand_file)
            df_cand["universe"] = univ
            all_candidates.append(df_cand)
        
        if all_file.exists():
            df_all = pd.read_csv(all_file)
            df_all["universe"] = univ
            all_metrics.append(df_all)
    
    # Combine
    if all_candidates:
        df_combined_cand = pd.concat(all_candidates, ignore_index=True)
        df_combined_cand = df_combined_cand.sort_values("global_score", ascending=False).reset_index(drop=True)
    else:
        df_combined_cand = pd.DataFrame()
    
    if all_metrics:
        df_combined_all = pd.concat(all_metrics, ignore_index=True)
    else:
        df_combined_all = pd.DataFrame()
    
    return df_combined_cand, df_combined_all


def generate_stable_lists(df_cand: pd.DataFrame, df_all: pd.DataFrame):
    """
    Generuje:
      - combined_pairs_scored.csv (wszystkie A/B+)
      - combined_top150_watchlist.csv
      - new_candidates.csv (persistence_days <= 5)
      - dropped_pairs.csv (stable_class = "drop")
    """
    # 1. Combined pairs scored (wszystkie A/B+)
    out_file = RESULTS_DIR / "combined_pairs_scored.csv"
    df_cand.to_csv(out_file, index=False)
    print(f"[INFO] Saved combined_pairs_scored.csv ({len(df_cand)} pairs)")
    
    # 2. Top 150 watchlist
    df_top150 = df_cand.head(150)
    out_file = RESULTS_DIR / "combined_top150_watchlist.csv"
    df_top150.to_csv(out_file, index=False)
    print(f"[INFO] Saved combined_top150_watchlist.csv ({len(df_top150)} pairs)")
    
    # 3. New candidates (persistence_days <= 5)
    df_new = df_cand[df_cand["persistence_days"] <= 5].copy()
    out_file = RESULTS_DIR / "new_candidates.csv"
    df_new.to_csv(out_file, index=False)
    print(f"[INFO] Saved new_candidates.csv ({len(df_new)} pairs)")
    
    # 4. Dropped pairs (stable_class = "drop")
    df_dropped = df_all[df_all["stable_class"] == "drop"].copy()
    df_dropped = df_dropped.sort_values("global_score", ascending=False).reset_index(drop=True)
    out_file = RESULTS_DIR / "dropped_pairs.csv"
    df_dropped.to_csv(out_file, index=False)
    print(f"[INFO] Saved dropped_pairs.csv ({len(df_dropped)} pairs)")


def main():
    print("[INFO] Aggregating results from all universes...")
    df_cand, df_all = aggregate_all()
    
    if df_cand.empty:
        print("[WARNING] No candidates found across all universes.")
        return
    
    print(f"[INFO] Total candidates: {len(df_cand)}")
    generate_stable_lists(df_cand, df_all)
    print("[INFO] Aggregation complete.")


if __name__ == "__main__":
    main()