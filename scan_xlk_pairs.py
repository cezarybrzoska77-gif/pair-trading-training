#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_xlk_pairs.py (wersja rozszerzona)
--------------------------------------
Skanuje spółki (domyślnie XLK) pod kątem par z filtrami:
1) Korelacja Pearsona (returns) w oknach 60 i 90 dni (oba ≥ corr_threshold),
2) Kointegracja (Engle–Granger) p-value ≤ pvalue_max,
3) Half-life spreadu ≤ half_life_max,
4) Stabilność beta: |beta_90 - beta_60| / |beta_90| ≤ beta_stability_max,
5) Volatility: σ/|mean| (z okna Z) ≤ vol_ratio_max,
6) ADF p-value ≤ adf_pvalue_max,
7) Z-score (okno zwindow): sygnał wejścia przy |Z| ≥ z_enter, wyjścia przy |Z| ≤ z_exit.

Zapisuje:
1) xlk_pairs_candidates.csv  -> pary po filtrach,
2) xlk_pairs_all_metrics.csv -> pełne metryki (diagnostyka).

Użycie (lokalnie):
    python scan_xlk_pairs.py --start-date 2018-01-01 \
      --corr-threshold 0.85 --pvalue-max 0.03 \
      --half-life-max 15 --beta-stability-max 0.15 \
      --adf-pvalue-max 0.05 --vol-ratio-max 4.0 \
      --z-window 60 --z-enter 2.0 --z-exit 0.5 \
      --out-csv results/xlk_pairs_candidates.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import yfinance as yf


# ======================
# USTAWIENIA DOMYŚLNE
# ======================

DEFAULT_START_DATE = "2018-01-01"
DEFAULT_OUT_CSV = "results/xlk_pairs_candidates.csv"

# Fallback: reprezentanci XLK (możesz rozszerzyć listę)
DEFAULT_XLK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "CRM", "CSCO", "ACN", "ADBE",
    "TXN", "QCOM", "AMD", "INTU", "ORCL", "AMAT", "NOW", "ADI", "MU", "IBM",
    "LRCX", "PANW", "ANSS", "CDNS", "SNPS", "MSI", "FTNT", "MCHP", "KLAC", "NXPI",
    "HPQ", "DELL", "APH", "TEL", "GLW", "FICO", "ROP", "HPE", "CTSH", "PAYX"
]


# ======================
# POMOCNICZE
# ======================

def print_step(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def parse_tickers_from_cli(cli_list: List[str] | None) -> List[str]:
    """Zwraca listę unikalnych tickerów z CLI."""
    if not cli_list:
        return []
    clean = [t.strip().upper() for t in cli_list if t and t.strip()]
    seen = set()
    uniq = []
    for t in clean:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def parse_tickers_from_file(path: str) -> List[str]:
    """Czyta tickery z pliku (jeden ticker na linię)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku z tickerami: {path}")
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if t and not t.startswith("#"):
                tickers.append(t)
    return parse_tickers_from_cli(tickers)


def resolve_universe(tickers_cli: List[str], tickers_file: str | None) -> List[str]:
    """Kolejność: --tickers > --tickers-file > fallback XLK."""
    if tickers_cli:
        return tickers_cli
    if tickers_file:
        return parse_tickers_from_file(tickers_file)
    return DEFAULT_XLK_TICKERS.copy()


def extract_prices(
    df: pd.DataFrame,
    field_preferred: str = "Adj Close",
    field_fallback: str = "Close",
    tickers: Iterable[str] | None = None
) -> pd.DataFrame:
    """
    Wyciąga macierz cen (wiersze = daty, kolumny = tickery).
    Obsługuje MultiIndex [field, ticker] / [ticker, field] i SingleIndex.
    """
    if df is None or df.empty:
        raise RuntimeError("Brak danych (DataFrame pusty).")

    cols = df.columns

    # MultiIndex: [field, ticker]
    if isinstance(cols, pd.MultiIndex) and field_preferred in cols.get_level_values(0):
        out = df[field_preferred].copy()
        if out.empty and field_fallback in cols.get_level_values(0):
            out = df[field_fallback].copy()
        if out.empty:
            raise KeyError(f"Brak kolumn '{field_preferred}' i '{field_fallback}'.")
        return out

    # MultiIndex: [ticker, field]
    if isinstance(cols, pd.MultiIndex) and (
        field_preferred in cols.get_level_values(-1) or field_fallback in cols.get_level_values(-1)
    ):
        try:
            out = df.xs(field_preferred, axis=1, level=-1)
        except KeyError:
            if field_fallback in cols.get_level_values(-1):
                out = df.xs(field_fallback, axis=1, level=-1)
            else:
                raise KeyError(f"Brak kolumn '{field_preferred}' i '{field_fallback}'.")
        return out

    # SingleIndex
    if field_preferred in cols:
        name = list(tickers)[0] if tickers else field_preferred
        return df[[field_preferred]].rename(columns={field_preferred: name})
    if field_fallback in cols:
        name = list(tickers)[0] if tickers else field_fallback
        return df[[field_fallback]].rename(columns={field_fallback: name})

    raise KeyError(f"Nie znaleziono '{field_preferred}' ani '{field_fallback}'. Kolumny: {list(map(str, cols))}")


def download_prices(
    tickers: List[str],
    start_date: str,
    end_date: str | None = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Pobiera dane przez yfinance i zwraca macierz cen (Adj Close preferowane)."""
    print_step(f"Pobieranie danych ({len(tickers)} tickerów) od {start_date}{' do ' + end_date if end_date else ''}…")
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column"
    )
    if data is None or data.empty:
        raise RuntimeError("Brak danych z yfinance. Sprawdź tickery, daty lub sieć.")

    if auto_adjust:
        prices = extract_prices(data, field_preferred="Close", field_fallback="Close", tickers=tickers)
    else:
        prices = extract_prices(data, field_preferred="Adj Close", field_fallback="Close", tickers=tickers)

    # Czyszczenie
    prices = prices.dropna(how="all")
    if prices.empty:
        raise RuntimeError("Po czyszczeniu brak danych cenowych.")
    min_non_na = max(50, int(0.4 * len(prices)))  # min 50 dni i >=40% dostępnych obserwacji
    prices = prices.dropna(axis=1, thresh=min_non_na).sort_index(axis=1)

    print_step(f"Dane OK: kształt = {prices.shape}")
    return prices


def rolling_corr_last_window(prices: pd.DataFrame, a: str, b: str, lookback: int) -> Tuple[float, int]:
    """Korelacja Pearsona dziennych stóp zwrotu dla ostatniego okna."""
    pa = prices[a].dropna()
    pb = prices[b].dropna()
    common = pa.index.intersection(pb.index)
    if len(common) < 5:
        return (np.nan, 0)
    pa = pa.loc[common].tail(lookback)
    pb = pb.loc[common].tail(lookback)
    ra = pa.pct_change().dropna()
