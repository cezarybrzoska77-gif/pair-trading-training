#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_xlk.py
-----------
Skrypt do pobierania danych giełdowych dla wszechświata spółek (domyślnie XLK),
odpornego wyciągania cen skorygowanych (Adj Close) oraz prostego skanowania par
na podstawie korelacji dziennych stóp zwrotu.

Użycie:
    python scan_xlk.py --start-date 2018-01-01
    python scan_xlk.py --tickers MSFT AAPL NVDA
    python scan_xlk.py --tickers MSFT AAPL --auto-adjust --out-prices-csv prices.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable, List

import numpy as np
import pandas as pd
import yfinance as yf


# ======================
# USTAWIENIA DOMYŚLNE
# ======================

DEFAULT_START_DATE = "2015-01-01"

# Najwięksi reprezentanci XLK (fallback, gdy nie podasz własnych tickerów).
# Lista celowo ograniczona do kilku(nastu) najpłynniejszych; możesz rozszerzyć wedle potrzeb.
DEFAULT_XLK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META", "CRM", "CSCO", "ACN", "ADBE",
    "TXN", "QCOM", "AMD", "INTU", "ORCL", "AMAT", "NOW", "ADI", "MU", "IBM"
]


# ======================
# PASEK POSTĘPU (ASCII)
# ======================

def print_progress(step: int, total_steps: int, width: int = 55, prefix: str = "") -> None:
    """
    Prost y pasek postępu z procentami i gwiazdkami,
    drukuje nową linię dla każdego kroku (jak w podanym logu).
    """
    pct = int((step / total_steps) * 100)
    filled = int((pct / 100) * width)
    bar = "[" + "*" * filled + " " * (width - filled) + "]"
    # Format podobny do przykładu: procent "w środku" i komentarz z prawej
    line = f"{bar}  {pct:>3}%"
    if prefix:
        line += f" {prefix}"
    print(line, flush=True)


# ======================
# EKSTRAKCJA CEN
# ======================

def extract_prices(
    df: pd.DataFrame,
    field_preferred: str = "Adj Close",
    field_fallback: str = "Close",
    tickers: Iterable[str] | None = None
) -> pd.DataFrame:
    """
    Wyciąga macierz cen (wiersze = daty, kolumny = tickery) dla żądanego pola.
    Obsługuje różne układy kolumn zwracane przez yfinance (MultiIndex/SingleIndex).
    Ma fallback do 'Close' gdy 'Adj Close' nie występuje.
    """
    if df is None or df.empty:
        raise RuntimeError("Brak danych wejściowych (DataFrame jest pusty).")

    cols = df.columns

    # MultiIndex przypadek: [field, ticker]
    if isinstance(cols, pd.MultiIndex) and field_preferred in cols.get_level_values(0):
        out = df[field_preferred].copy()
        if out.empty and field_fallback in cols.get_level_values(0):
            out = df[field_fallback].copy()
        if out.empty:
            raise KeyError(f"Nie znaleziono kolumn '{field_preferred}' ani '{field_fallback}'.")
        return out

    # MultiIndex przypadek: [ticker, field]
    if isinstance(cols, pd.MultiIndex) and field_preferred in cols.get_level_values(-1):
        try:
            out = df.xs(field_preferred, axis=1, level=-1)
        except KeyError:
            if field_fallback in cols.get_level_values(-1):
                out = df.xs(field_fallback, axis=1, level=-1)
            else:
                raise KeyError(f"Nie znaleziono kolumn '{field_preferred}' ani '{field_fallback}'.")
        return out

    # SingleIndex przypadek (pojedynczy ticker albo nietypowy układ)
    if field_preferred in cols:
        name = list(tickers)[0] if tickers else field_preferred
        return df[[field_preferred]].rename(columns={field_preferred: name})
    if field_fallback in cols:
        name = list(tickers)[0] if tickers else field_fallback
        return df[[field_fallback]].rename(columns={field_fallback: name})

    # Brak spodziewanych pól
    raise KeyError(
        f"Nie znaleziono kolumn '{field_preferred}' ani '{field_fallback}'. "
        f"Dostępne kolumny: {list(map(str, cols))}"
    )


# ======================
# NARZĘDZIA POMOCNICZE
# ======================

def parse_tickers_from_cli(cli_list: List[str] | None) -> List[str]:
    """Zwraca listę tickerów z CLI (usunie duplikaty, spacje, puste)."""
    if not cli_list:
        return []
    clean = [t.strip().upper() for t in cli_list if t and t.strip()]
    # unikalne z zachowaniem kolejności
    seen = set()
    uniq = []
    for t in clean:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def resolve_universe(tickers_cli: List[str]) -> List[str]:
    """
    Zwraca finalną listę tickerów do pobrania:
    - jeśli podano w CLI -> użyj,
    - w przeciwnym razie -> fallback do listy największych spółek XLK.
    """
    if tickers_cli:
        return tickers_cli
    return DEFAULT_XLK_TICKERS.copy()


def compute_top_correlated_pairs(prices: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Liczy dzienne zwroty, korelacje i zwraca top N skorelowanych par (bez duplikatów).
    """
    if prices is None or prices.empty:
        return pd.DataFrame(columns=["ticker_i", "ticker_j", "corr"])

    returns = prices.pct_change().dropna(how="all")
    if returns.empty:
        return pd.DataFrame(columns=["ticker_i", "ticker_j", "corr"])

    corr = returns.corr()
    # bierzemy tylko górny trójkąt (bez diagonalnych 1.0)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_vals = corr.where(mask).stack().sort_values(ascending=False)

    top = corr_vals.head(top_n)
    if top.empty:
        return pd.DataFrame(columns=["ticker_i", "ticker_j", "corr"])

    out = top.reset_index()
    out.columns = ["ticker_i", "ticker_j", "corr"]
    return out


# ======================
# GŁÓWNA LOGIKA
# ======================

def main() -> int:
    total_steps = 10
    step = 0

    parser = argparse.ArgumentParser(
        description="Pobieranie danych i prosty skan par dla spółek (domyślnie XLK)."
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Lista tickerów rozdzielona spacjami, np. --tickers MSFT AAPL NVDA"
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Data początkowa (YYYY-MM-DD), domyślnie {DEFAULT_START_DATE}"
    )
    parser.add_argument(
        "--auto-adjust",
        action="store_true",
        help="Jeśli ustawione, używa auto_adjust=True (Close będzie już skorygowany)."
    )
    parser.add_argument(
        "--out-prices-csv",
        default=None,
        help="Ścieżka do CSV z zapisanymi cenami (opcjonalnie)."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Ile najwyżej skorelowanych par wypisać (domyślnie 20)."
    )

    args = parser.parse_args()

    step += 1
    print_progress(step, total_steps, prefix="Parsowanie argumentów")

    # Ustalenie wszechświata
    step += 1
    tickers_cli = parse_tickers_from_cli(args.tickers)
    universe = resolve_universe(tickers_cli)
    if not universe:
        print("Błąd: nie określono tickerów, a fallback XS jest pusty.", file=sys.stderr)
        return 2
    print_progress(step, total_steps, prefix=f"Ustalanie wszechświata ({len(universe)} tickerów)")

    # Walidacja tickerów
    step += 1
    bad = [t for t in universe if not t.isalnum()]
    if bad:
        print(f"Ostrzeżenie: wykryto nietypowe tickery: {bad}", file=sys.stderr)
    print_progress(step, total_steps, prefix="Walidacja tickerów")

    # Pobieranie danych
    step += 1
    print("Pobieranie danych...")
    try:
        data = yf.download(
            tickers=universe,
            start=args.start_date,
            auto_adjust=args.auto_adjust,  # jeśli True, 'Close' będzie już skorygowany
            progress=False,
            group_by="column"  # stabilny układ: [field, ticker]
        )
    except Exception as e:
        print_progress(step, total_steps, prefix="Pobieranie danych – BŁĄD")
        print(f"Błąd pobierania danych z yfinance: {e}", file=sys.stderr)
        return 3
    print_progress(step, total_steps, prefix="Pobieranie danych – OK")

    # Ekstrakcja cen (Adj Close -> Close fallback)
    step += 1
    try:
        if args.auto_adjust:
            # W tym trybie 'Close' jest już skorygowany
            prices = extract_prices(data, field_preferred="Close", field_fallback="Close", tickers=universe)
        else:
            # Standard: preferuj 'Adj Close', w razie braku fallback do 'Close'
            prices = extract_prices(data, field_preferred="Adj Close", field_fallback="Close", tickers=universe)
    except Exception as e:
        print_progress(step, total_steps, prefix="Ekstrakcja cen – BŁĄD")
        print(f"Błąd ekstrakcji cen: {e}", file=sys.stderr)
        return 4
    print_progress(step, total_steps, prefix="Ekstrakcja cen – OK")

    # Czyszczenie cen
    step += 1
    prices = prices.sort_index(axis=1)
    prices = prices.dropna(how="all")
    # Opcjonalnie: usuń kolumny z bardzo małą ilością danych
    min_non_na = max(10, int(0.2 * len(prices)))  # wymagaj min 10 obserwacji i >=20% próby
    prices = prices.dropna(axis=1, thresh=min_non_na)
    if prices.empty:
        print_progress(step, total_steps, prefix="Czyszczenie – BRAK DANYCH")
        print("Po czyszczeniu brak dostępnych cen. Sprawdź tickery i zakres dat.", file=sys.stderr)
        return 5
    print_progress(step, total_steps, prefix=f"Czyszczenie – OK (kształt: {prices.shape})")

    # Zapis CSV (opcjonalnie)
    step += 1
    if args.out_prices_csv:
        try:
            prices.to_csv(args.out_prices_csv, index=True)
            print_progress(step, total_steps, prefix=f"Zapis CSV – {args.out_prices_csv}")
        except Exception as e:
            print_progress(step, total_steps, prefix="Zapis CSV – BŁĄD")
            print(f"Nie udało się zapisać CSV: {e}", file=sys.stderr)
            return 6
    else:
        print_progress(step, total_steps, prefix="Zapis CSV – pominięty")

    # Obliczenia statystyczne (korelacje)
    step += 1
    top_pairs_df = compute_top_correlated_pairs(prices, top_n=args.top_n)
    print_progress(step, total_steps, prefix="Korelacje – OK")

    # Raport
    step += 1
    print("\n=== PODSUMOWANIE ===")
    print(f"Zakres dat: {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"Liczba spółek po filtrach: {prices.shape[1]}")
    print(f"Liczba obserwacji: {prices.shape[0]}")

    if not top_pairs_df.empty:
        print("\nTop skorelowane pary (dzienny return, Pearson):")
        # ładny wydruk
        for _, row in top_pairs_df.iterrows():
            print(f"  {row['ticker_i']:>6s} — {row['ticker_j']:<6s} | corr = {row['corr']:.4f}")
    else:
        print("\nBrak par do wyświetlenia (niewystarczające dane lub brak korelacji).")

    print_progress(step, total_steps, prefix="Raport – OK")

    # Gotowe
    step += 1
    print_progress(step, total_steps, prefix="Zakończono")
    return 0


if __name__ == "__main__":
    # Niewielkie opóźnienie wyłącznie dla czytelności paska postępu (opcjonalne)
    t0 = time.time()
    try:
        code = main()
    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika.", file=sys.stderr)
        code = 130
    dt = time.time() - t0
    print(f"\nCzas wykonania: {dt:.2f} s")
    sys.exit(code)