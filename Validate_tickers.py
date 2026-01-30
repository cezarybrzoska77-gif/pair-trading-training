#!/usr/bin/env python3
"""
Validate ticker files for duplicates, format issues, and basic sanity checks.
"""

import sys
from pathlib import Path
from collections import Counter

UNIVERSES = [
    'tech_core',
    'semis', 
    'software',
    'financials',
    'healthcare',
    'discretionary'
]

def validate_ticker_file(filepath: Path) -> tuple[bool, list[str]]:
    """Validate a single ticker file."""
    issues = []
    
    if not filepath.exists():
        return False, [f"File not found: {filepath}"]
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse tickers (skip comments and empty lines)
    tickers = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if line and not line.startswith('#'):
            tickers.append(line)
            
            # Check format
            if ' ' in line or '\t' in line:
                issues.append(f"Line {i}: Whitespace in ticker '{line}'")
            if not line.isupper():
                issues.append(f"Line {i}: Ticker not uppercase '{line}'")
            if len(line) > 10:
                issues.append(f"Line {i}: Ticker suspiciously long '{line}'")
    
    # Check for duplicates
    ticker_counts = Counter(tickers)
    duplicates = [t for t, count in ticker_counts.items() if count > 1]
    
    if duplicates:
        issues.append(f"Duplicate tickers found: {', '.join(duplicates)}")
    
    # Check minimum count
    if len(tickers) < 20:
        issues.append(f"Only {len(tickers)} tickers (minimum 20 recommended)")
    
    return len(issues) == 0, issues

def main():
    """Validate all ticker files."""
    print("=" * 60)
    print("TICKER FILE VALIDATION")
    print("=" * 60)
    
    data_dir = Path('data')
    
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        print("  Expected structure: data/tickers_*.txt")
        return 1
    
    all_valid = True
    ticker_stats = []
    
    for universe in UNIVERSES:
        filepath = data_dir / f'tickers_{universe}.txt'
        
        valid, issues = validate_ticker_file(filepath)
        
        # Count tickers
        if filepath.exists():
            with open(filepath, 'r') as f:
                tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            ticker_count = len(tickers)
            ticker_stats.append((universe, ticker_count))
        else:
            ticker_count = 0
            ticker_stats.append((universe, 0))
        
        # Print results
        status = "✓" if valid else "✗"
        print(f"\n{status} {universe:20s} ({ticker_count:3d} tickers)")
        
        if not valid:
            all_valid = False
            for issue in issues:
                print(f"  - {issue}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_tickers = sum(count for _, count in ticker_stats)
    print(f"Total tickers across all universes: {total_tickers}")
    
    for universe, count in ticker_stats:
        print(f"  {universe:20s}: {count:3d}")
    
    print("=" * 60)
    
    if all_valid:
        print("✓ All ticker files valid!")
        return 0
    else:
        print("✗ Some validation issues found (see above)")
        return 1

if __name__ == '__main__':
    sys.exit(main())