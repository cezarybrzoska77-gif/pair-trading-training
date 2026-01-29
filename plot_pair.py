#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_pair.py
------------
Rysuje wykresy dla pary (domyślnie KLAC–LRCX):

- Spread = A - (alpha + beta * B) + markery sygnałów (Long/Short)
- Z-score (20/40/60) z liniami ±2 i ±1.5

Zapisy:
- results/<a>_<b>_spread.png
- results/<a>_<b>_zscore.png
- results/<a>_<b>_combo.png

Uruchomienie (lokalnie):
    python plot_pair.py --a KLAC --b LRCX --start-date 2018-01-01 --out-dir results --auto-adjust

Uwaga:
- Backend Matplotlib wymuszony na 'Agg', więc skrypt działa w środowiskach headless (np. GitHub Actions).
"""

# --- WAŻNE: backend do środowisk bez ekranu (CI) ---
import matplotlib
matplotlib.use("Agg")  # to musi być przed "import matplotlib.pyplot as plt"

import argparse
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


# ---------- Pomocnicze ----------

def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def extract_prices(df: pd.DataFrame,
                   field_preferred: str = "Adj Close",
                   field_fallback: str = "Close",
                   tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Odporna ekstrakcja macierzy cen (daty x tickery) dla różnych układów kolumn zwracanych przez yfinance:
    - MultiIndex [field, ticker]
    - MultiIndex [ticker, field]
    - SingleIndex
    """
    if df is None or df.empty:
        raise RuntimeError("Brak danych (DataFrame pusty).")

    cols = df.columns

    # MultiIndex: [field, ticker]
    if isinstance(cols, pd.MultiIndex) and field_preferred in cols.get_level_values(0):
        out = df[field_preferred]
        if out.empty and field_fallback in cols.get_level_values(0):
            out = df[field_fallback]
        if out.empty:
            raise KeyError(f"Nie ma '{field_preferred}' ani '{field_fallback}'.")
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
                raise KeyError(f"Nie ma '{field_preferred}' ani '{field_fallback}'.")
        return out

    # SingleIndex
    if field_preferred in cols:
        name = list(tickers)[0] if tickers else field_preferred
        return df[[field_preferred]].rename(columns={field_preferred: name})
    if field_fallback in cols:
        name = list(tickers)[0] if tickers else field_fallback
        return df[[field_fallback]].rename(columns={field_fallback: name})

    raise KeyError(f"Nie znaleziono '{field_preferred}' ani '{field_fallback}'. Kolumny={list(map(str, cols))}")


def download_prices(tickers: list[str],
                    start_date: str,
                    end_date: str | None = None,
                    auto_adjust: bool = False,
                    retries: int = 3,
                    pause_sec: float = 1.5) -> pd.DataFrame:
    """
    Pobiera dane z yfinance z kilkoma próbami (retry).
    Zwraca DataFrame z cenami (Adj Close preferowane, Close fallback).
    """
    info(f"Pobieranie danych: {tickers} od {start_date}{' do ' + end_date if end_date else ''} …")
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                auto_adjust=auto_adjust,   # jeśli True -> Close jest już skorygowany
                progress=False,
                group_by="column",
                threads=True
            )
            if df is not None and not df.empty:
                break
        except Exception as e:
            last_exc = e
        info(f"… próba {attempt}/{retries} nieudana, ponawiam po {pause_sec}s …")
        time.sleep(pause_sec)
    else:
        raise RuntimeError(f"Brak danych z yfinance po {retries} próbach. Ostatni błąd: {last_exc}")

    if auto_adjust:
        prices = extract_prices(df, field_preferred="Close", field_fallback="Close", tickers=tickers)
    else:
        prices = extract_prices(df, field_preferred="Adj Close", field_fallback="Close", tickers=tickers)

    # proste czyszczenie
    prices = prices.dropna(how="all")
    min_non_na = max(30, int(0.4 * len(prices)))
    prices = prices.dropna(axis=1, thresh=min_non_na).sort_index(axis=1)
    if prices.empty:
        raise RuntimeError("Po czyszczeniu brak danych cenowych.")
    info(f"Dane OK: {prices.shape}")
    return prices


def hedge_and_spread(a: pd.Series, b: pd.Series) -> tuple[float, float, pd.Series]:
    """
    OLS: a ~ const + beta * b  → zwraca (alpha, beta, spread = a - (alpha + beta * b))
    """
    df = pd.concat([a, b], axis=1).dropna()
    df.columns = ["y", "x"]
    if len(df) < 30:
        raise RuntimeError("Za mało danych do OLS (min 30 obserwacji).")
    X = sm.add_constant(df["x"])
    model = sm.OLS(df["y"], X).fit()
    alpha = float(model.params.get("const", 0.0))
    beta = float(model.params.get("x", np.nan))
    spread = df["y"] - (alpha + beta * df["x"])
    spread.name = "spread"
    return alpha, beta, spread


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


# ---------- Rysowanie ----------

def format_axes_date(ax):
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_horizontalalignment("center")


def plot_spread(ax, spread: pd.Series, long_sig: pd.Series, short_sig: pd.Series, title: str):
    ax.plot(spread.index, spread.values, color="#1f77b4", label="Spread")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)

    # Markery sygnałów
    if long_sig.any():
        ax.scatter(spread.index[long_sig], spread[long_sig], marker="^", color="#2ca02c",
                   s=25, label="Long signal", zorder=3)
    if short_sig.any():
        ax.scatter(spread.index[short_sig], spread[short_sig], marker="v", color="#d62728",
                   s=25, label="Short signal", zorder=3)

    ax.set_title(title)
    ax.set_ylabel("Spread")
    ax.legend(loc="best")
    format_axes_date(ax)


def plot_zscores(ax, z20: pd.Series, z40: pd.Series, z60: pd.Series, title: str):
