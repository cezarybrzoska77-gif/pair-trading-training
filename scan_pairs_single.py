#!/usr/bin/env python3
import argparse
import os
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
import warnings

from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Single-basket pair scanner (Discretionary / XLY)")
    p.add_argument("--tickers-file", default="data/tickers_discretionary.txt")
    p.add_argument("--start-date", default="2018-01-01")
    p.add_argument("--auto-adjust", dest="auto_adjust", action="store_true", default=True)
    p.add_argument("--no-auto-adjust", dest="auto_adjust", action="store_false")
    p.add_argument("--use-percent-returns", action="store_true", default=False)
    p.add_argument("--winsorize", action="store_true", default=True)
    p.add_argument("--coint-lookbacks", default="240,300")
    p.add_argument("--min-sample", type=int, default=200)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--out-dir", default="results/discretionary")
    return p.parse_args()


def load_tickers(path):
    with open(path, "r") as f:
        tickers = [x.strip().upper() for x in f if x.strip()]
    return list(dict.fromkeys(tickers))


def download_prices(tickers, start, auto_adjust):
    data = yf.download(
        tickers,
        start=start,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    prices = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") in data.columns:
                s = data[(t, "Close")].dropna()
                if len(s) > 0:
                    prices[t] = s.rename(t)
    else:
        s = data["Close"].dropna()
        if len(s) > 0:
            prices[tickers[0]] = s.rename(tickers[0])

    if not prices:
        raise RuntimeError("No price data downloaded.")

    return pd.concat(prices.values(), axis=1)


def compute_returns(prices, percent=False):
    if percent:
        rets = prices.pct_change()
    else:
        rets = np.log(prices).diff()
    return rets.dropna(how="all")


def winsorize_df(df, lower=0.01, upper=0.99):
    ql = df.quantile(lower)
    qu = df.quantile(upper)
    return df.clip(lower=ql, upper=qu, axis=1)


def residuals_vs_index(price_series, index_series):
    df = pd.concat([price_series, index_series], axis=1).dropna()
    y = df.iloc[:, 0]
    x = sm.add_constant(df.iloc[:, 1])
    model = sm.OLS(y, x).fit()
    return model.resid


def rolling_spearman(a, b, window):
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < window:
        return np.nan
    r = (
        df.rank()
        .rolling(window)
        .corr()
        .iloc[-1, 0]
    )
    return r


def corr_hitrate_30d_6m(a, b):
    lookback_6m = 126
    roll = a.rolling(30).corr(b)
    recent = roll.dropna().iloc[-lookback_6m:]
    if len(recent) == 0:
        return np.nan
    return (recent >= 0.80).mean()


def engle_granger_best(a, b, lookbacks):
    best = {"pvalue": np.nan, "stat": np.nan, "lb": np.nan}
    for lb in lookbacks:
        if len(a) < lb or len(b) < lb:
            continue
        x = a.iloc[-lb:]
        y = b.iloc[-lb:]
        try:
            stat, pval, _ = coint(x, y)
            if np.isnan(best["pvalue"]) or pval < best["pvalue"]:
                best = {"pvalue": pval, "stat": stat, "lb": lb}
        except Exception:
            continue
    return best["pvalue"], best["stat"], best["lb"]


def score_pair(row):
    corr_mean = np.clip(row["corr_mean"], 0, 1)
    spear_mean = np.clip((row["spearman_60"] + row["spearman_90"]) / 2, 0, 1)
    resid = np.clip(row["resid_corr_max"], 0, 1)
    hitrate = np.clip(row["corr_hitrate_30d_6m"], 0, 1)

    if pd.isna(row["coint_pvalue_best"]):
        coint_score = 0.0
    else:
        coint_score = 1.0 - min(row["coint_pvalue_best"] / 0.20, 1.0)

    score = (
        0.30 * corr_mean
        + 0.15 * spear_mean
        + 0.20 * resid
        + 0.20 * coint_score
        + 0.10 * hitrate
    )

    if row["grade"] == "A":
        score += 0.03
    elif row["grade"] == "B+":
        score += 0.01

    return min(score, 1.0)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tickers = load_tickers(args.tickers_file)
    tickers = list(dict.fromkeys(tickers + ["XLY"]))

    prices = download_prices(tickers, args.start_date, args.auto_adjust)
    returns = compute_returns(prices, percent=args.use_percent_returns)

    if args.winsorize:
        returns = winsorize_df(returns)

    lookbacks = [int(x) for x in args.coint_lookbacks.split(",")]

    valid_tickers = [t for t in returns.columns if t != "XLY"]

    results = []

    for a, b in itertools.combinations(valid_tickers, 2):
        ra, rb = returns[a], returns[b]
        df = pd.concat([ra, rb], axis=1).dropna()
        sample = len(df)
        if sample < args.min_sample:
            continue

        corr_60 = df[a].rolling(60).corr(df[b]).iloc[-1]
        corr_90 = df[a].rolling(90).corr(df[b]).iloc[-1]
        corr_obs_60 = df[a].rolling(60).count().iloc[-1]
        corr_obs_90 = df[a].rolling(90).count().iloc[-1]
        corr_mean = np.nanmean([corr_60, corr_90])

        spearman_60 = rolling_spearman(df[a], df[b], 60)
        spearman_90 = rolling_spearman(df[a], df[b], 90)

        resid_a = residuals_vs_index(prices[a], prices["XLY"])
        resid_b = residuals_vs_index(prices[b], prices["XLY"])
        resid_df = pd.concat([resid_a, resid_b], axis=1).dropna()

        resid_corr_60 = resid_df.iloc[:, 0].rolling(60).corr(resid_df.iloc[:, 1]).iloc[-1]
        resid_corr_90 = resid_df.iloc[:, 0].rolling(90).corr(resid_df.iloc[:, 1]).iloc[-1]
        resid_corr_max = np.nanmax([resid_corr_60, resid_corr_90])

        hitrate = corr_hitrate_30d_6m(df[a], df[b])

        coint_p, coint_stat, coint_lb = engle_granger_best(
            prices[a].dropna(), prices[b].dropna(), lookbacks
        )

        grade = None
        if (
            corr_mean >= 0.82
            and coint_p <= 0.05
            and hitrate >= 0.70
            and resid_corr_max >= 0.60
        ):
            grade = "A"
        elif (
            (
                corr_mean >= 0.78
                or (corr_60 >= 0.80 and corr_90 >= 0.76)
            )
            and coint_p <= 0.12
            and spearman_60 >= 0.75
            and spearman_90 >= 0.75
            and resid_corr_max >= 0.60
            and hitrate >= 0.70
        ):
            grade = "B+"

        results.append({
            "pair": f"{a}/{b}",
            "a": a,
            "b": b,
            "sample": sample,
            "corr_60": corr_60,
            "corr_90": corr_90,
            "corr_obs_60": corr_obs_60,
            "corr_obs_90": corr_obs_90,
            "corr_mean": corr_mean,
            "spearman_60": spearman_60,
            "spearman_90": spearman_90,
            "resid_corr_60": resid_corr_60,
            "resid_corr_90": resid_corr_90,
            "resid_corr_max": resid_corr_max,
            "corr_hitrate_30d_6m": hitrate,
            "coint_pvalue_best": coint_p,
            "coint_stat_best": coint_stat,
            "coint_lookback_best": coint_lb,
            "grade": grade,
        })

    df_all = pd.DataFrame(results)
    df_all["score_w"] = df_all.apply(score_pair, axis=1)

    all_path = os.path.join(args.out_dir, "discretionary_all_metrics.csv")
    df_all.to_csv(all_path, index=False)

    df_cand = df_all[df_all["grade"].isin(["A", "B+"])]
    df_cand = df_cand.sort_values(
        by=["grade", "coint_pvalue_best", "corr_90", "corr_60"],
        ascending=[True, True, False, False],
    ).head(args.topk)

    cand_path = os.path.join(args.out_dir, "discretionary_candidates.csv")
    df_cand.to_csv(cand_path, index=False)

    print(f"Saved:\n- {all_path}\n- {cand_path}")


if __name__ == "__main__":
    main()
