"""Stage 4: option strategies 91–100 on monthly RTS-index options (MOEX settle prices).

Position size: 1 underlying RTS futures notional per unit of capital; P&L in index points / entry futures price.
python -m backtest.options [id ...]
"""
import re
import sys
from functools import lru_cache
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd

from . import data
from .engine import Result, metrics, money_returns
from .futures import chain, continuous
from .run import FINALIST_SHARPE, PERIODS, ROOT, names

CODE = re.compile(r"^RI(\d+)B([A-X])(\d)$")  # monthly series only; weekly codes carry an extra letter
COST_PTS = 20.0           # per option leg per side: fee + half spread, in index points (assumption)
COST_PCT = 0.03           # ...or 3% of the premium, whichever is larger
FUT_COST = 0.001          # futures leg, per side, of notional
MIN_DTE, MAX_DTE = 20, 50  # calendar days to expiry when opening a monthly cycle


def ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def black76(F, K, T, vol, call):
    if T <= 0 or vol <= 0:
        return max(F - K, 0) if call else max(K - F, 0)
    d1 = (log(F / K) + 0.5 * vol * vol * T) / (vol * sqrt(T))
    d2 = d1 - vol * sqrt(T)
    return F * ncdf(d1) - K * ncdf(d2) if call else K * ncdf(-d2) - F * ncdf(-d1)


def implied_vol(price, F, K, T, call):
    lo, hi = 0.01, 3.0
    intrinsic = max(F - K, 0) if call else max(K - F, 0)
    if T <= 0 or price <= intrinsic:
        return np.nan
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if black76(F, K, T, mid, call) < price else (lo, mid)
    return (lo + hi) / 2


def strike_for_delta(F, T, vol, delta, call):
    """Strike whose Black-76 delta equals `delta` (absolute value for puts)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(delta if call else 1 - delta)
    return F * exp(-z * vol * sqrt(T) + 0.5 * vol * vol * T)


@lru_cache(None)
def table():
    frames = []
    for p in sorted((data.DATA / "options" / "RTS").glob("*.csv")):
        df = pd.read_csv(p, usecols=["TRADEDATE", "SECID", "SETTLEPRICE", "OPENPOSITION"])
        frames.append(df)
    t = pd.concat(frames)
    parsed = t["SECID"].str.extract(CODE)
    t = t[parsed[0].notna()].copy()
    parsed = parsed.dropna()
    t["K"] = parsed[0].astype(float)
    letter = parsed[1]
    t["call"] = letter <= "L"
    t["TRADEDATE"] = pd.to_datetime(t["TRADEDATE"])
    expiry = t.groupby("SECID")["TRADEDATE"].max()
    t["expiry"] = t["SECID"].map(expiry)
    return t


@lru_cache(None)
def settle():
    return table().pivot_table(index="TRADEDATE", columns="SECID", values="SETTLEPRICE").reindex(data.calendar()).ffill()


@lru_cache(None)
def futures_price():
    """Raw (not adjusted) close of the most liquid RTS futures: the options' underlying."""
    contracts, front, _ = chain("RTS")
    cal = data.calendar()
    px = pd.Series(np.nan, index=cal)
    for i, sid in enumerate(front):
        if isinstance(sid, str):
            px.iloc[i] = contracts[sid][1]["C"].iloc[i]
    return px.ffill()


def series_on(date):
    """Monthly option series alive on `date` with 20–50 calendar days left: (expiry, frame of its options)."""
    t = table()
    day = t[t["TRADEDATE"] == date]
    dte = (day["expiry"] - date).dt.days
    day = day[(dte >= MIN_DTE) & (dte <= MAX_DTE) & (day["expiry"] < data.calendar()[-1])]
    if day.empty:
        return None, None
    exp_ = day.groupby("expiry")["OPENPOSITION"].sum().idxmax()
    return exp_, day[day["expiry"] == exp_]


def atm_vol(date, opts, F):
    T = (opts["expiry"].iloc[0] - date).days / 365
    calls = opts[opts["call"]]
    k = calls.iloc[(calls["K"] - F).abs().argsort()[:1]]
    if k.empty:
        return np.nan
    return implied_vol(k["SETTLEPRICE"].iloc[0], F, k["K"].iloc[0], T, True)


def pick(opts, K, call):
    side = opts[opts["call"] == call]
    if side.empty:
        return None
    return side.iloc[(side["K"] - K).abs().argsort()[:1]]["SECID"].iloc[0]


def leg_cost(price):
    return max(COST_PTS, COST_PCT * price)


def simulate_cycles(build, entry_days=None, exit_rule=None):
    """Generic engine: on each entry day `build(date, F, opts, vol)` returns {secid: qty, 'FUT': qty} or None.

    Positions run to expiry unless `exit_rule(pnl_pts, premium_pts, date)` says to close."""
    cal = data.calendar()
    S, F = settle(), futures_price()
    fut_ret = continuous("RTS")[0]["C"].reindex(cal).pct_change().fillna(0)
    ret = pd.Series(0.0, index=cal)
    exposure = pd.Series(0.0, index=cal)
    costs = pd.Series(0.0, index=cal)
    pos, f0, premium, expiry, entry_value = None, None, 0.0, None, 0.0
    candidates = set(entry_days) if entry_days is not None else None
    for i in range(1, len(cal)):
        d = cal[i]
        if pos is not None:
            pnl = sum(q * (S.at[d, s] - S.at[cal[i - 1], s]) for s, q in pos.items() if s != "FUT")
            day_ret = pnl / f0 + pos.get("FUT", 0) * fut_ret.iloc[i] * F.iloc[i - 1] / f0
            ret.iloc[i] += day_ret
            exposure.iloc[i] = 1.0
            value = sum(q * S.at[d, s] for s, q in pos.items() if s != "FUT")
            done = d >= expiry or (exit_rule is not None and exit_rule(value - entry_value, premium, d))
            if done:
                if d < expiry:
                    c = sum(abs(q) * leg_cost(S.at[d, s]) for s, q in pos.items() if s != "FUT") / f0
                    costs.iloc[i] += c
                c_fut = abs(pos.get("FUT", 0)) * FUT_COST
                costs.iloc[i] += c_fut
                pos = None
            continue
        if candidates is not None and d not in candidates:
            continue
        exp_, opts = series_on(d)
        if opts is None or not F.iloc[i] > 0:
            continue
        vol = atm_vol(d, opts, F.iloc[i])
        if not vol > 0:
            continue
        legs = build(d, F.iloc[i], opts, vol)
        if not legs:
            continue
        pos, f0, expiry = legs, F.iloc[i], exp_
        entry_value = sum(q * S.at[d, s] for s, q in pos.items() if s != "FUT")
        premium = -entry_value
        costs.iloc[i] += sum(abs(q) * leg_cost(S.at[d, s]) for s, q in pos.items() if s != "FUT") / f0
        costs.iloc[i] += abs(pos.get("FUT", 0)) * FUT_COST
    total = ret - costs + money_returns()
    return Result(total, exposure, exposure.diff().abs().fillna(0), costs)


def strikes(F, T, vol, opts, spec):
    """spec: list of (delta, call, qty) → {secid: qty}."""
    out = {}
    for delta, call, qty in spec:
        sid = pick(opts, strike_for_delta(F, T, vol, delta, call), call)
        if sid is None:
            return None
        out[sid] = out.get(sid, 0) + qty
    return out


def builder(spec, futures=0):
    def build(d, F, opts, vol):
        T = (opts["expiry"].iloc[0] - d).days / 365
        legs = strikes(F, T, vol, opts, spec)
        if legs is not None and futures:
            legs["FUT"] = futures
        return legs
    return build


def stop_loss(mult):
    return lambda pnl, premium, d: premium > 0 and pnl < -mult * premium


def take_profit(frac):
    return lambda pnl, premium, d: premium > 0 and pnl > frac * premium


def realized_vol():
    c = continuous("RTS")[0]["C"].reindex(data.calendar()).ffill()
    return c.pct_change().rolling(20).std() * np.sqrt(252)


def s95():
    rv = realized_vol()
    base = builder([(0.15, True, -1), (0.15, False, -1), (0.05, True, 1), (0.05, False, 1)])

    def build(d, F, opts, vol):
        return base(d, F, opts, vol) if vol - rv.get(d, np.nan) > 0.03 else None
    return simulate_cycles(build, exit_rule=take_profit(0.5))


def sma_up():
    F = futures_price()
    return F > F.rolling(200).mean()


def s96():
    up = sma_up()
    with_put = builder([(0.25, False, 1)], futures=1)
    no_put = lambda d, F, opts, vol: {"FUT": 1}

    def build(d, F, opts, vol):
        return no_put(d, F, opts, vol) if up.get(d, True) else with_put(d, F, opts, vol)
    return simulate_cycles(build)


def s98():
    cal = data.calendar()
    days = []
    for m in data.cbr_meetings():
        i = cal.searchsorted(m)
        if i >= 3:
            days.append(cal[i - 3])
    meeting_after = {cal[cal.searchsorted(m) - 3]: m for m in data.cbr_meetings() if cal.searchsorted(m) >= 3}
    straddle = builder([(0.5, True, 1), (0.5, False, 1)])

    def exit_rule(pnl, premium, d):
        return any(m < d for e, m in meeting_after.items() if e <= d and (d - m).days <= 5)
    return simulate_cycles(straddle, entry_days=days, exit_rule=exit_rule)


def rsi2_days():
    F = futures_price()
    d = F.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    r = 100 - 100 / (1 + up / dn)
    return list(F.index[(r < 10) & sma_up()])


def breakout_days():
    F = futures_price()
    return list(F.index[F > F.rolling(20).max().shift()])


STRATEGIES = {
    "91": lambda: simulate_cycles(builder([(0.25, True, -1)], futures=1)),
    "92": lambda: simulate_cycles(builder([(0.25, False, -1)])),
    "93": lambda: simulate_cycles(builder([(0.15, True, -1), (0.15, False, -1)]), exit_rule=stop_loss(2)),
    "94": lambda: simulate_cycles(builder([(0.15, True, -1), (0.15, False, -1), (0.05, True, 1), (0.05, False, 1)]),
                                  exit_rule=take_profit(0.5)),
    "95": s95,
    "96": s96,
    "97": lambda: simulate_cycles(builder([(0.2, False, 1), (0.2, True, -1)], futures=1)),
    "98": s98,
    "99": lambda: simulate_cycles(builder([(0.30, False, -1), (0.15, False, 1)]), entry_days=rsi2_days(),
                                  exit_rule=take_profit(0.5)),
    "100": lambda: simulate_cycles(builder([(0.6, True, 1)]), entry_days=breakout_days(),
                                   exit_rule=lambda pnl, premium, d: pnl < 0.5 * premium),
    "base_fut": lambda: simulate_cycles(lambda d, F, opts, vol: {"FUT": 1}),
}


def main(ids):
    catalog = {**names(), "base_fut": "Контроль: фьючерс RTS 1x"}
    rows, results = [], {}
    for sid in ids:
        res = STRATEGIES[sid]()
        results[sid] = res
        row = {"id": sid, "name": catalog.get(sid, sid)}
        for p in ("train", "valid"):
            for k, v in metrics(res, *PERIODS[p]).items():
                row[f"{p}_{k}"] = v
        rows.append(row)
        print(f"{sid:>8} {row['name'][:42]:42} train SR {row.get('train_sharpe', np.nan):5.2f} "
              f"cagr {row.get('train_cagr', np.nan):6.1%} dd {row.get('train_max_dd', np.nan):6.1%} | "
              f"valid SR {row.get('valid_sharpe', np.nan):5.2f} cagr {row.get('valid_cagr', np.nan):6.1%} "
              f"dd {row.get('valid_max_dd', np.nan):6.1%}", flush=True)
    table_ = pd.DataFrame(rows).set_index("id")
    ok = (table_["train_sharpe"].fillna(-9) > FINALIST_SHARPE) & (table_["valid_sharpe"].fillna(-9) > FINALIST_SHARPE)
    for sid in table_.index[ok]:
        for k, v in metrics(results[sid], *PERIODS["test"]).items():
            table_.loc[sid, f"test_{k}"] = v
    table_.to_csv(ROOT / "results" / "stage4.csv", float_format="%.4f")
    print(f"\nфиналисты: {list(table_.index[ok])}")
    if ok.any():
        print(table_.loc[ok].filter(like="test_").round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or list(STRATEGIES))
