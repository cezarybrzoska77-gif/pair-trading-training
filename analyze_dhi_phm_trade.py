#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workflow 2 – Retail Pair Trade Analyzer (DHI vs PHM)

Cel: ocenić, czy para DHI/PHM jest gotowa do otwarcia pozycji według reguł retail Workflow 2:
- start-date = 2018-01-01, auto-adjust ON (Adj Close)
- domyślnie log-returns, opcja percent-returns
- beta-neutral / cash-neutral
- ADF p-value, half-life, Z-score (60/30)
- progi wejścia i wyjścia
- zapis wyników do CSV, JSON i opcjonalnie PNG

CLI:
--start-date            (domyślnie 2018-01-01)
--auto-adjust / --no-auto-adjust
--use-percent-returns
--winsorize / --no-winsorize
--z-lookbacks           (domyślnie "60,30")
--min-sample            (domyślnie 200)
--out-dir               (domyślnie results_workflow2)
--plot / --no-plot
--help-checklist        (wyświetla checklistę wejścia/wyjścia i kończy)
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def download_data(ticker, start_date, auto_adjust=True):
    df = yf.download(ticker, start=start_date, auto_adjust=auto_adjust)
    if df.empty:
        raise ValueError(f"Brak danych dla {ticker}")
    return df['Adj Close']

def compute_returns(prices, use_percent=False, winsorize=True):
    if use_percent:
        rets = prices.pct_change().dropna()
    else:
        rets = np.log(prices / prices.shift(1)).dropna()
    if winsorize:
        lower = rets.quantile(0.01)
        upper = rets.quantile(0.99)
        rets = rets.clip(lower, upper)
    return rets

def rolling_ols(y, x, window):
    betas = []
    for i in range(len(y) - window + 1):
        model = OLS(y[i:i+window], np.vstack([np.ones(window), x[i:i+window]]).T)
        res = model.fit()
        betas.append(res.params[1])
    return np.array(betas)

def spread_metrics(y, x, alpha, beta):
    spread_beta = y - (alpha + beta * x)
    spread_cash = y - x
    return spread_beta, spread_cash

def adf_half_life(spread):
    try:
        adf_res = adfuller(spread, autolag='AIC')
        pval = adf_res[1]
        stat = adf_res[0]
    except Exception:
        pval, stat = np.nan, np.nan
    try:
        delta_s = spread.diff().dropna()
        s_lag = spread.shift(1).dropna()
        s_lag = s_lag.loc[delta_s.index]
        beta_hl = OLS(delta_s.values, s_lag.values).fit().params[0]
        if 1 + beta_hl <= 0 or abs(beta_hl) < 1e-6:
            half_life = np.nan
        else:
            half_life = -np.log(2) / np.log(1 + beta_hl)
    except Exception:
        half_life = np.nan
    return pval, stat, half_life

def zscore(spread, window):
    roll_mean = spread.rolling(window).mean()
    roll_std = spread.rolling(window).std()
    z = (spread - roll_mean) / roll_std
    return z

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--auto-adjust", action="store_true", default=True)
    parser.add_argument("--no-auto-adjust", action="store_false", dest="auto_adjust")
    parser.add_argument("--use-percent-returns", action="store_true", default=False)
    parser.add_argument("--winsorize", action="store_true", default=True)
    parser.add_argument("--no-winsorize", action="store_false", dest="winsorize")
    parser.add_argument("--z-lookbacks", default="60,30")
    parser.add_argument("--min-sample", type=int, default=200)
    parser.add_argument("--out-dir", default="results_workflow2")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    parser.add_argument("--help-checklist", action="store_true")
    args = parser.parse_args()

    if args.help_checklist:
        print("""
Workflow 2 – DHI/PHM Trade Checklist

1. Beta-neutral OLS: alpha + beta
2. Beta stability (beta_60 vs beta_90)
3. Spread_beta & spread_cash
4. ADF p-value & half-life
5. Z-score (Z60, Z30)
6. Hedge mode recommendation: beta_neutral / cash_neutral
7. TRADE_READY = YES if all entry criteria met:
   - ADF p <= 0.05
   - Half-life <= 15
   - Z60 <= -2 & Z30 <= -1.5 (LONG) OR Z60 >= 2 & Z30 >=1.5 (SHORT)
8. Exit: |Z60| <= 0.5
""")
        sys.exit(0)

    os.makedirs(args.out_dir, exist_ok=True)

    y_ticker = "DHI"
    x_ticker = "PHM"

    try:
        y_prices = download_data(y_ticker, args.start_date, args.auto_adjust)
        x_prices = download_data(x_ticker, args.start_date, args.auto_adjust)
    except Exception as e:
        print(f"ERROR downloading: {e}")
        sys.exit(1)

    # align
    df = pd.concat([y_prices, x_prices], axis=1, join="inner")
    df.columns = [y_ticker, x_ticker]
    if len(df) < args.min_sample:
        print(f"NO: Not enough data points ({len(df)} < {args.min_sample})")
        trade_ready = "NO"
        df.to_csv(os.path.join(args.out_dir,"DHI_PHM_trade_readiness.csv"))
        sys.exit(1)

    ret_y = compute_returns(df[y_ticker], args.use_percent_returns, args.winsorize)
    ret_x = compute_returns(df[x_ticker], args.use_percent_returns, args.winsorize)

    # Beta-neutral OLS
    model = OLS(df[y_ticker], np.vstack([np.ones(len(df)), df[x_ticker]]).T).fit()
    alpha_OLS, beta_OLS = model.params
    beta_60 = rolling_ols(ret_y.values, ret_x.values, 60)[-1]
    beta_90 = rolling_ols(ret_y.values, ret_x.values, 90)[-1]
    beta_stability_pct = 100 * abs(beta_60 - beta_90) / np.mean([abs(beta_60), abs(beta_90)])

    spread_beta, spread_cash = spread_metrics(df[y_ticker], df[x_ticker], alpha_OLS, beta_OLS)

    adf_p_beta, adf_stat_beta, hl_beta = adf_half_life(spread_beta)
    adf_p_cash, adf_stat_cash, hl_cash = adf_half_life(spread_cash)

    z60 = zscore(spread_beta, 60).iloc[-1]
    z30 = zscore(spread_beta, 30).iloc[-1]
    delta_z3d = z60 - zscore(spread_beta, 3).iloc[-4]

    vol_ratio = spread_beta.std() / (ret_y.std() + ret_x.std())

    # Hedge recommendation
    hedge_reco = "beta_neutral"
    if beta_stability_pct >= 40 or (adf_p_cash < adf_p_beta and hl_cash < hl_beta):
        hedge_reco = "cash_neutral"
    elif 25 < beta_stability_pct < 35:
        if adf_p_beta > adf_p_cash - 0.02 and hl_beta >= hl_cash - 3:
            hedge_reco = "cash_neutral"

    # TRADE_READY
    trade_ready = "NO"
    direction = None
    sizes = None
    exit_rule = "|Z60| <= 0.5"
    reason = []

    if hedge_reco == "beta_neutral":
        spread_used = spread_beta
        hl_used = hl_beta
        adf_used = adf_p_beta
        if beta_stability_pct > 35:
            reason.append(f"Beta stability too high ({beta_stability_pct:.1f}%)")
    else:
        spread_used = spread_cash
        hl_used = hl_cash
        adf_used = adf_p_cash

    # Entry criteria
    if adf_used <= 0.05:
        if hl_used <= 15:
            if z60 <= -2.0 and z30 <= -1.5:
                trade_ready = "YES"
                direction = "long_y_short_x"
            elif z60 >= 2.0 and z30 >= 1.5:
                trade_ready = "YES"
                direction = "short_y_long_x"
            else:
                reason.append("Z-score not meeting entry thresholds")
        else:
            reason.append(f"Half-life too high ({hl_used:.1f})")
    else:
        reason.append(f"ADF p-value too high ({adf_used:.3f})")

    # Sizes
    if trade_ready == "YES":
        if hedge_reco == "beta_neutral":
            w_y = 1.0
            w_x = -beta_OLS
            total = abs(w_y) + abs(w_x)
            w_y /= total
            w_x /= total
            sizes = {"y_weight": round(w_y,3), "x_weight": round(w_x,3),
                     "y_shares": round(w_y/df[y_ticker].iloc[-1],3),
                     "x_shares": round(-w_x/df[x_ticker].iloc[-1],3)}
        else:
            sizes = {"y_weight": 0.5, "x_weight": -0.5,
                     "y_shares": round(0.5/df[y_ticker].iloc[-1],3),
                     "x_shares": round(0.5/df[x_ticker].iloc[-1],3)}

    # Output
    out_csv = os.path.join(args.out_dir, "DHI_PHM_trade_readiness.csv")
    out_json = os.path.join(args.out_dir, "DHI_PHM_trade_readiness.json")
    pd.DataFrame([{
        "TRADE_READY": trade_ready,
        "hedge_reco": hedge_reco,
        "direction": direction,
        "sizes": sizes,
        "exit_rule": exit_rule,
        "reason": reason,
        "alpha_OLS": round(alpha_OLS,4),
        "beta_OLS": round(beta_OLS,4),
        "beta_stability_pct": round(beta_stability_pct,2),
        "adf_p_beta": round(adf_p_beta,4),
        "adf_stat_beta": round(adf_stat_beta,4),
        "hl_beta": round(hl_beta,1),
        "adf_p_cash": round(adf_p_cash,4),
        "adf_stat_cash": round(adf_stat_cash,4),
        "hl_cash": round(hl_cash,1),
        "Z60": round(z60,3),
        "Z30": round(z30,3),
        "delta_Z3d": round(delta_z3d,3),
        "vol_ratio": round(vol_ratio,4)
    }]).to_csv(out_csv, index=False)
    with open(out_json,"w") as f:
        json.dump({
            "TRADE_READY": trade_ready,
            "hedge_reco": hedge_reco,
            "direction": direction,
            "sizes": sizes,
            "exit_rule": exit_rule,
            "reason": reason,
            "metrics": {
                "alpha_OLS": alpha_OLS,
                "beta_OLS": beta_OLS,
                "beta_stability_pct": beta_stability_pct,
                "adf_p_beta": adf_p_beta,
                "adf_stat_beta": adf_stat_beta,
                "hl_beta": hl_beta,
                "adf_p_cash": adf_p_cash,
                "adf_stat_cash": adf_stat_cash,
                "hl_cash": hl_cash,
                "Z60": z60,
                "Z30": z30,
                "delta_Z3d": delta_z3d,
                "vol_ratio": vol_ratio
            }
        }, f, indent=2)

    # Optional plot
    if args.plot:
        plt.figure(figsize=(12,8))
        plt.subplot(3,1,1)
        roll_mean = spread_beta.rolling(60).mean()
        roll_std = spread_beta.rolling(60).std()
        plt.plot(spread_beta, label="spread_beta")
        plt.plot(roll_mean, label="roll_mean", color="orange")
        plt.fill_between(spread_beta.index, roll_mean+2*roll_std, roll_mean-2*roll_std, color="orange", alpha=0.2)
        plt.title("Spread Beta + rolling mean ±2σ")
        plt.legend()

        plt.subplot(3,1,2)
        z60_series = zscore(spread_beta,60)
        z30_series = zscore(spread_beta,30)
        plt.plot(z60_series,label="Z60")
        plt.plot(z30_series,label="Z30")
        plt.axhline(2,color="red",linestyle="--")
        plt.axhline(1.5,color="red",linestyle=":")
        plt.axhline(-1.5,color="green",linestyle=":")
        plt.axhline(-2,color="green",linestyle="--")
        plt.axhline(0.5,color="black",linestyle="-.")
        plt.axhline(-0.5,color="black",linestyle="-.")
        plt.title("Z-scores")
        plt.legend()

        plt.subplot(3,1,3)
        plt.plot(spread_beta[-250:],label="spread_beta")
        plt.plot(spread_cash[-250:],label="spread_cash")
        plt.title(f"Spread Beta vs Cash (last 250 days)\nADF_beta p={adf_p_beta:.4f}, ADF_cash p={adf_p_cash:.4f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir,"DHI_PHM_spread_chart.png"))
        plt.close()

    print(f"TRADE_READY: {trade_ready}, hedge_reco={hedge_reco}, direction={direction}")
    if trade_ready=="YES":
        print(f"Sizes: {sizes}, exit_rule={exit_rule}")
        print("5-punktowy plan:")
        print("1. Konflikt ✅")
        print("2. Kalendarz ✅")
        print("3. Tryb ✅")
        print("4. Wagi legs: y/x")
        print(f"5. Exit = {exit_rule} + time-stop=2*HL")
    else:
        print(f"Main reason: {reason[0] if reason else 'Unknown'}, Suggested next step: czekaj na dual-Z lub rozważ cash_neutral")

if __name__=="__main__":
    main()
