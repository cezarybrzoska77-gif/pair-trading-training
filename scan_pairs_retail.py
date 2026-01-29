#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_pairs_retail.py
Retail mode pairs scanner: correlation + cointegration (Engle-Granger).
Sector: Technology (XLK + VGT + FTEC combined).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")

# ============================================================================
# LOGGER SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_tickers_from_file(filepath: str) -> List[str]:
    """Load tickers from text file (one ticker per line)."""
    logger.info(f"Loading tickers from {filepath}")
    path = Path(filepath)
    if not path.exists():
        logger.error(f"Tickers file not found: {filepath}")
        sys.exit(1)
    
    with open(path, "r", encoding="utf-8") as f:
        tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
    
    tickers = sorted(list(set(tickers)))  # Remove duplicates
    logger.info(f"Loaded {len(tickers)} unique tickers")
    return tickers


def download_data_robust(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str],
    auto_adjust: bool
) -> pd.DataFrame:
    """
    Download price data via yfinance with robust error handling.
    Downloads tickers individually to handle delisted/invalid tickers gracefully.
    """
    logger.info(f"Downloading data for {len(tickers)} tickers from {start_date}")
    logger.info(f"Auto-adjust: {auto_adjust}")
    
    all_prices = []
    failed_tickers = []
    success_count = 0
    
    for i, ticker in enumerate(tickers, 1):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(tickers)} tickers")
        
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=auto_adjust
            )
            
            if df.empty:
                failed_tickers.append(ticker)
                continue
            
            # Extract Close or Adj Close
            if "Close" in df.columns:
                prices = df["Close"].copy()
            elif "Adj Close" in df.columns:
                prices = df["Adj Close"].copy()
            else:
                failed_tickers.append(ticker)
                continue
            
            # Rename to ticker symbol
            prices.name = ticker
            
            # Check for sufficient data
            valid_data = prices.dropna()
            if len(valid_data) < 200:
                logger.debug(f"{ticker}: insufficient data ({len(valid_data)} days)")
                failed_tickers.append(ticker)
                continue
            
            all_prices.append(prices)
            success_count += 1
            
        except Exception as e:
            logger.debug(f"{ticker}: download failed - {str(e)[:100]}")
            failed_tickers.append(ticker)
            continue
    
    if len(all_prices) == 0:
        logger.error("No data downloaded successfully")
        sys.exit(1)
    
    # Combine into DataFrame
    prices_df = pd.concat(all_prices, axis=1)
    
    logger.info(f"Successfully downloaded: {len(prices_df.columns)} tickers")
    if failed_tickers:
        logger.warning(f"Failed to download: {len(failed_tickers)} tickers")
        if len(failed_tickers) <= 10:
            logger.info(f"Failed tickers: {', '.join(failed_tickers)}")
    
    logger.info(f"Date range: {prices_df.index.min()} to {prices_df.index.max()}")
    logger.info(f"Sample tickers: {', '.join(list(prices_df.columns[:5]))}")
    
    return prices_df


def compute_returns(prices: pd.DataFrame, use_percent: bool) -> pd.DataFrame:
    """Compute log-returns or percent returns."""
    if use_percent:
        logger.info("Computing percent returns")
        returns = prices.pct_change()
    else:
        logger.info("Computing log-returns")
        returns = np.log(prices / prices.shift(1))
    
    # Replace inf with NaN
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    return returns


def compute_correlation_windows(
    returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window_60: int = 60,
    window_90: int = 90
) -> Tuple[float, float, int, int]:
    """
    Compute rolling correlation for last 60 and 90 days.
    Returns: (corr_60, corr_90, obs_60, obs_90)
    Requires >=60% of observations in window.
    """
    ret_a = returns[ticker_a]
    ret_b = returns[ticker_b]
    
    # Last 60 days
    data_60 = pd.concat([ret_a.tail(window_60), ret_b.tail(window_60)], axis=1).dropna()
    obs_60 = len(data_60)
    if obs_60 >= int(0.6 * window_60):
        corr_60 = data_60.corr().iloc[0, 1]
    else:
        corr_60 = np.nan
    
    # Last 90 days
    data_90 = pd.concat([ret_a.tail(window_90), ret_b.tail(window_90)], axis=1).dropna()
    obs_90 = len(data_90)
    if obs_90 >= int(0.6 * window_90):
        corr_90 = data_90.corr().iloc[0, 1]
    else:
        corr_90 = np.nan
    
    return corr_60, corr_90, obs_60, obs_90


def compute_cointegration_best(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    lookbacks: List[int]
) -> Tuple[float, float, int]:
    """
    Run Engle-Granger cointegration test for multiple lookbacks.
    Returns: (best_stat, best_pvalue, best_lookback)
    """
    price_a = prices[ticker_a]
    price_b = prices[ticker_b]
    
    best_pvalue = 1.0
    best_stat = np.nan
    best_lookback = lookbacks[0]
    
    for lb in lookbacks:
        data = pd.concat([price_a.tail(lb), price_b.tail(lb)], axis=1).dropna()
        if len(data) < int(0.6 * lb):
            continue
        
        try:
            stat, pvalue, _ = coint(data.iloc[:, 0], data.iloc[:, 1])
            if pvalue < best_pvalue:
                best_pvalue = pvalue
                best_stat = stat
                best_lookback = lb
        except Exception:
            continue
    
    return best_stat, best_pvalue, best_lookback


def compute_sample_size(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    max_lookback: int
) -> int:
    """Count common observations in the longest lookback window."""
    price_a = prices[ticker_a]
    price_b = prices[ticker_b]
    data = pd.concat([price_a.tail(max_lookback), price_b.tail(max_lookback)], axis=1).dropna()
    return len(data)


def assign_grade(pvalue: float) -> str:
    """Assign grade based on cointegration p-value."""
    if np.isnan(pvalue):
        return "F"
    elif pvalue <= 0.05:
        return "A"
    elif pvalue <= 0.10:
        return "B"
    else:
        return "C"


# ============================================================================
# MAIN SCANNER
# ============================================================================

def scan_pairs(
    tickers: List[str],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    coint_lookbacks: List[int],
    min_sample: int
) -> pd.DataFrame:
    """
    Scan all pairs and compute correlation + cointegration metrics.
    """
    n = len(tickers)
    total_pairs = n * (n - 1) // 2
    logger.info(f"Scanning {total_pairs} pairs from {n} tickers")
    
    results = []
    max_lookback = max(coint_lookbacks)
    
    processed = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = tickers[i]
            b = tickers[j]
            
            processed += 1
            if processed % 500 == 0:
                logger.info(f"Progress: {processed}/{total_pairs} pairs ({100*processed/total_pairs:.1f}%)")
            
            # Correlation windows
            try:
                corr_60, corr_90, obs_60, obs_90 = compute_correlation_windows(
                    returns, a, b
                )
            except Exception as e:
                logger.debug(f"Correlation error for {a}-{b}: {e}")
                continue
            
            if np.isnan(corr_60) or np.isnan(corr_90):
                continue
            
            corr_mean = (corr_60 + corr_90) / 2.0
            
            # Cointegration
            try:
                coint_stat, coint_pvalue, coint_lb = compute_cointegration_best(
                    prices, a, b, coint_lookbacks
                )
            except Exception as e:
                logger.debug(f"Cointegration error for {a}-{b}: {e}")
                coint_stat = np.nan
                coint_pvalue = 1.0
                coint_lb = coint_lookbacks[0]
            
            # Sample size
            try:
                sample = compute_sample_size(prices, a, b, max_lookback)
            except Exception:
                sample = 0
            
            # Grade
            grade = assign_grade(coint_pvalue)
            
            results.append({
                "a": a,
                "b": b,
                "corr_60": round(corr_60, 4),
                "corr_90": round(corr_90, 4),
                "corr_obs_60": obs_60,
                "corr_obs_90": obs_90,
                "corr_mean": round(corr_mean, 4),
                "coint_stat_best": round(coint_stat, 4) if not np.isnan(coint_stat) else np.nan,
                "coint_pvalue_best": round(coint_pvalue, 6),
                "coint_lookback_best": coint_lb,
                "sample": sample,
                "grade": grade
            })
    
    df = pd.DataFrame(results)
    logger.info(f"Computed metrics for {len(df)} pairs")
    return df


def apply_filters(
    df: pd.DataFrame,
    min_sample: int
) -> pd.DataFrame:
    """
    Apply retail mode filters:
    - Correlation: (mean >= 0.82) OR (corr_60 >= 0.84 AND corr_90 >= 0.80)
    - Cointegration: p-value <= 0.10
    - Sample: >= min_sample
    """
    logger.info("Applying filters...")
    
    if len(df) == 0:
        logger.warning("Empty dataframe received, returning empty filtered result")
        return df
    
    corr_pass = (
        (df["corr_mean"] >= 0.82) |
        ((df["corr_60"] >= 0.84) & (df["corr_90"] >= 0.80))
    )
    
    coint_pass = df["coint_pvalue_best"] <= 0.10
    sample_pass = df["sample"] >= min_sample
    
    mask = corr_pass & coint_pass & sample_pass
    filtered = df[mask].copy()
    
    logger.info(f"Pairs after filters: {len(filtered)} / {len(df)}")
    return filtered


def sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by: grade (A before B), coint_pvalue (asc), corr_90 (desc), corr_60 (desc).
    """
    if len(df) == 0:
        return df
    
    df = df.sort_values(
        by=["grade", "coint_pvalue_best", "corr_90", "corr_60"],
        ascending=[True, True, False, False]
    ).reset_index(drop=True)
    
    return df


# ============================================================================
# CLI & MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Retail mode pairs scanner: correlation + cointegration"
    )
    
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="List of tickers (space-separated)"
    )
    parser.add_argument(
        "--tickers-file",
        default="data/tickers_tech.txt",
        help="Path to file with tickers (default: data/tickers_tech.txt)"
    )
    parser.add_argument(
        "--start-date",
        default="2018-01-01",
        help="Start date (default: 2018-01-01)"
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date (optional)"
    )
    parser.add_argument(
        "--auto-adjust",
        action="store_true",
        default=True,
        help="Use auto-adjust for prices (default: ON)"
    )
    parser.add_argument(
        "--no-auto-adjust",
        dest="auto_adjust",
        action="store_false",
        help="Disable auto-adjust"
    )
    parser.add_argument(
        "--use-percent-returns",
        action="store_true",
        help="Use percent returns instead of log-returns"
    )
    parser.add_argument(
        "--coint-lookbacks",
        default="240,300",
        help="Cointegration lookback windows (comma-separated, default: 240,300)"
    )
    parser.add_argument(
        "--min-sample",
        type=int,
        default=200,
        help="Minimum sample size (default: 200)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Top K candidates to save (default: 50)"
    )
    parser.add_argument(
        "--out-csv",
        default="results/tech_pairs_candidates.csv",
        help="Output CSV path (default: results/tech_pairs_candidates.csv)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("TECH PAIRS SCANNER - RETAIL MODE")
    logger.info("=" * 60)
    
    # Load tickers
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        logger.info(f"Using {len(tickers)} tickers from CLI")
    else:
        tickers = load_tickers_from_file(args.tickers_file)
    
    if len(tickers) < 2:
        logger.error("Need at least 2 tickers to scan pairs")
        sys.exit(1)
    
    # Parse cointegration lookbacks
    coint_lookbacks = [int(x.strip()) for x in args.coint_lookbacks.split(",")]
    logger.info(f"Cointegration lookbacks: {coint_lookbacks}")
    
    # Download data (robust method)
    prices = download_data_robust(tickers, args.start_date, args.end_date, args.auto_adjust)
    
    if len(prices.columns) < 2:
        logger.error(f"Insufficient tickers with data: {len(prices.columns)} (need at least 2)")
        sys.exit(1)
    
    # Compute returns
    returns = compute_returns(prices, args.use_percent_returns)
    
    # Get tickers with valid data
    valid_tickers = sorted(prices.columns.tolist())
    logger.info(f"Valid tickers for scanning: {len(valid_tickers)}")
    
    # Scan pairs
    all_metrics = scan_pairs(valid_tickers, prices, returns, coint_lookbacks, args.min_sample)
    
    # Create output directory
    out_all = Path(args.out_csv).parent / "tech_pairs_all_metrics.csv"
    out_all.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle empty results
    if len(all_metrics) == 0:
        logger.warning("No pairs computed. Creating empty output files.")
        
        # Create empty DataFrames with proper columns
        empty_cols = ["a", "b", "corr_60", "corr_90", "corr_obs_60", "corr_obs_90", 
                      "corr_mean", "coint_stat_best", "coint_pvalue_best", 
                      "coint_lookback_best", "sample", "grade"]
        empty_df = pd.DataFrame(columns=empty_cols)
        
        empty_df.to_csv(out_all, index=False)
        empty_df.to_csv(args.out_csv, index=False)
        
        logger.info(f"Saved empty files: {out_all}, {args.out_csv}")
        logger.info("Process completed (0 candidates)")
        return
    
    # Save all metrics
    try:
        all_metrics.to_csv(out_all, index=False)
        logger.info(f"Saved all metrics: {out_all}")
    except Exception as e:
        logger.error(f"Failed to save all metrics: {e}")
        sys.exit(1)
    
    # Apply filters
    candidates = apply_filters(all_metrics, args.min_sample)
    
    if len(candidates) == 0:
        logger.warning("No pairs passed filters!")
        # Create empty CSV with correct columns
        candidates.to_csv(args.out_csv, index=False)
        logger.info(f"Saved empty candidates CSV: {args.out_csv}")
        logger.info("Process completed (0 candidates)")
        return
    
    # Sort candidates
    candidates = sort_candidates(candidates)
    
    # Limit to top K
    if len(candidates) > args.topk:
        logger.info(f"Limiting to top {args.topk} candidates")
        candidates = candidates.head(args.topk)
    
    # Save candidates
    try:
        candidates.to_csv(args.out_csv, index=False)
        logger.info(f"Saved {len(candidates)} candidates: {args.out_csv}")
    except Exception as e:
        logger.error(f"Failed to save candidates: {e}")
        sys.exit(1)
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total pairs scanned: {len(all_metrics)}")
    logger.info(f"Pairs passing filters: {len(candidates)}")
    if len(candidates) > 0:
        grade_counts = candidates["grade"].value_counts()
        for grade in sorted(grade_counts.index):
            logger.info(f"  Grade {grade}: {grade_counts[grade]}")
        logger.info("\nTop 5 candidates:")
        top5 = candidates.head(5)[["a", "b", "corr_mean", "coint_pvalue_best", "grade"]]
        print(top5.to_string(index=False))
    logger.info("=" * 60)
    logger.info("Process completed successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)