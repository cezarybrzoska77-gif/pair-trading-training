#!/usr/bin/env python3
"""
Run All Universes and Aggregate Results
Orchestrates scanning of all 6 universes and creates combined rankings.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# Universe configurations
UNIVERSES = {
    'tech_core': {'residual_index': 'QQQ'},
    'semis': {'residual_index': 'SOXX'},
    'software': {'residual_index': 'IGV'},
    'financials': {'residual_index': 'XLF'},
    'healthcare': {'residual_index': 'XLV'},
    'discretionary': {'residual_index': 'XLY'},
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run scanner for all universes and aggregate results'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2018-01-01',
        help='Start date for data download (default: 2018-01-01)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for data download (default: today)'
    )
    parser.add_argument(
        '--auto-adjust',
        action='store_true',
        default=True,
        help='Use auto-adjusted close prices (default: True)'
    )
    parser.add_argument(
        '--no-auto-adjust',
        action='store_false',
        dest='auto_adjust',
        help='Use raw close prices'
    )
    parser.add_argument(
        '--use-percent-returns',
        action='store_true',
        default=False,
        help='Use percent returns instead of log returns'
    )
    parser.add_argument(
        '--coint-lookbacks',
        type=str,
        default='240,300',
        help='Comma-separated cointegration lookback periods (default: 240,300)'
    )
    parser.add_argument(
        '--min-sample',
        type=int,
        default=200,
        help='Minimum sample size for pair (default: 200)'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        default=False,
        help='Use strict filtering (A-grade only)'
    )
    parser.add_argument(
        '--allow-grade-b',
        action='store_true',
        default=False,
        help='Allow B-grade pairs'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=100,
        help='Keep top K pairs per universe (default: 100)'
    )
    parser.add_argument(
        '--topn',
        type=int,
        default=150,
        help='Top N pairs for final watchlist (default: 150)'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    parser.add_argument(
        '--skip-scan',
        action='store_true',
        default=False,
        help='Skip scanning, only aggregate existing results'
    )
    
    return parser.parse_args()


def run_scanner(universe: str, args: argparse.Namespace) -> bool:
    """Run scanner for a single universe."""
    logger.info("=" * 60)
    logger.info(f"Running scanner for universe: {universe}")
    logger.info("=" * 60)
    
    # Build command
    cmd = [
        sys.executable,
        'scan_pairs_multi.py',
        '--universe', universe,
        '--start-date', args.start_date,
        '--coint-lookbacks', args.coint_lookbacks,
        '--min-sample', str(args.min_sample),
        '--topk', str(args.topk),
        '--out-dir', args.out_dir,
    ]
    
    if args.end_date:
        cmd.extend(['--end-date', args.end_date])
    
    if args.auto_adjust:
        cmd.append('--auto-adjust')
    else:
        cmd.append('--no-auto-adjust')
    
    if args.use_percent_returns:
        cmd.append('--use-percent-returns')
    
    if args.strict:
        cmd.append('--strict')
    
    if args.allow_grade_b:
        cmd.append('--allow-grade-b')
    
    # Run command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"✓ Successfully completed scan for {universe}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Scanner failed for {universe}: {e}")
        return False


def aggregate_results(universes: List[str], out_dir: str, topn: int) -> None:
    """Aggregate results from all universes."""
    logger.info("=" * 60)
    logger.info("Aggregating results from all universes")
    logger.info("=" * 60)
    
    all_candidates = []
    
    for universe in universes:
        candidates_file = Path(out_dir) / universe / f"{universe}_candidates.csv"
        
        if candidates_file.exists():
            try:
                df = pd.read_csv(candidates_file)
                logger.info(f"Loaded {len(df)} candidates from {universe}")
                all_candidates.append(df)
            except Exception as e:
                logger.warning(f"Error reading {candidates_file}: {e}")
        else:
            logger.warning(f"Candidates file not found: {candidates_file}")
    
    if not all_candidates:
        logger.error("No candidate files found to aggregate!")
        return
    
    # Combine all candidates
    combined = pd.concat(all_candidates, ignore_index=True)
    logger.info(f"Combined total: {len(combined)} pairs from {len(all_candidates)} universes")
    
    # Sort by score
    combined_sorted = combined.sort_values('score_w', ascending=False).reset_index(drop=True)
    
    # Add global rank
    combined_sorted.insert(0, 'rank', range(1, len(combined_sorted) + 1))
    
    # Save combined scored pairs
    combined_file = Path(out_dir) / 'combined_pairs_scored.csv'
    combined_sorted.to_csv(combined_file, index=False)
    logger.info(f"Saved combined rankings to: {combined_file}")
    
    # Create top N watchlist
    top_watchlist = combined_sorted.head(topn).copy()
    watchlist_file = Path(out_dir) / f'combined_top{topn}_watchlist.csv'
    top_watchlist.to_csv(watchlist_file, index=False)
    logger.info(f"Saved TOP {topn} watchlist to: {watchlist_file}")
    
    # Summary statistics
    logger.info("=" * 60)
    logger.info("AGGREGATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total pairs: {len(combined_sorted)}")
    logger.info(f"Top watchlist size: {len(top_watchlist)}")
    logger.info(f"Score range: {combined_sorted['score_w'].min():.3f} - {combined_sorted['score_w'].max():.3f}")
    
    # By universe
    logger.info("\nPairs by universe:")
    universe_counts = combined_sorted['universe'].value_counts().sort_index()
    for univ, count in universe_counts.items():
        logger.info(f"  {univ}: {count}")
    
    # By grade
    logger.info("\nPairs by grade:")
    grade_counts = combined_sorted['grade'].value_counts()
    for grade, count in grade_counts.items():
        logger.info(f"  {grade}: {count}")
    
    # Top universes in watchlist
    logger.info(f"\nTop {topn} watchlist by universe:")
    watchlist_univ = top_watchlist['universe'].value_counts().sort_index()
    for univ, count in watchlist_univ.items():
        logger.info(f"  {univ}: {count}")
    
    logger.info("=" * 60)


def main():
    """Main execution function."""
    args = parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    universes = list(UNIVERSES.keys())
    
    # Run scanners for all universes
    if not args.skip_scan:
        logger.info(f"Starting scans for {len(universes)} universes...")
        
        success_count = 0
        failed_universes = []
        
        for universe in universes:
            success = run_scanner(universe, args)
            if success:
                success_count += 1
            else:
                failed_universes.append(universe)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Scanning complete: {success_count}/{len(universes)} succeeded")
        if failed_universes:
            logger.warning(f"Failed universes: {', '.join(failed_universes)}")
        logger.info("=" * 60)
        logger.info("")
    else:
        logger.info("Skipping scans (--skip-scan enabled)")
    
    # Aggregate results
    aggregate_results(universes, args.out_dir, args.topn)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL OPERATIONS COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()