#!/usr/bin/env python3
"""
Quick import test for scan_pairs_multi.py dependencies
Run this before deploying to verify all dependencies are available.
"""

import sys

def test_imports():
    """Test all critical imports."""
    tests = []
    
    # Test numpy
    try:
        import numpy as np
        tests.append(("numpy", "✓", np.__version__))
    except ImportError as e:
        tests.append(("numpy", "✗", str(e)))
    
    # Test pandas
    try:
        import pandas as pd
        tests.append(("pandas", "✓", pd.__version__))
    except ImportError as e:
        tests.append(("pandas", "✗", str(e)))
    
    # Test scipy
    try:
        import scipy
        from scipy import stats
        tests.append(("scipy", "✓", scipy.__version__))
    except ImportError as e:
        tests.append(("scipy", "✗", str(e)))
    
    # Test statsmodels (CRITICAL - check coint function)
    try:
        import statsmodels
        from statsmodels.tsa.stattools import coint
        tests.append(("statsmodels", "✓", f"{statsmodels.__version__} (coint available)"))
    except ImportError as e:
        tests.append(("statsmodels", "✗", str(e)))
    
    # Test yfinance
    try:
        import yfinance as yf
        tests.append(("yfinance", "✓", yf.__version__))
    except ImportError as e:
        tests.append(("yfinance", "✗", str(e)))
    
    # Test sklearn
    try:
        import sklearn
        from sklearn.linear_model import LinearRegression
        tests.append(("scikit-learn", "✓", sklearn.__version__))
    except ImportError as e:
        tests.append(("scikit-learn", "✗", str(e)))
    
    # Print results
    print("=" * 60)
    print("DEPENDENCY CHECK")
    print("=" * 60)
    
    all_ok = True
    for name, status, info in tests:
        print(f"{status} {name:20s} {info}")
        if status == "✗":
            all_ok = False
    
    print("=" * 60)
    
    if all_ok:
        print("✓ All dependencies available!")
        return 0
    else:
        print("✗ Some dependencies missing. Run: pip install -r requirements.txt")
        return 1

if __name__ == '__main__':
    sys.exit(test_imports())