"""Stage 2: futures strategies 76–90 on back-adjusted continuous contracts.

python -m backtest.futures [id ...]
"""
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from . import data
from .engine import Asset, Result, metrics, money_returns, simulate
from .run import PERIODS, FINALIST_SHARPE, ROOT, names
from .strategies import month_ends, on_dates, sma, atr, positions

ROLL_DAYS = 5  # switch to the next contract this many trading days before expiry
SLIPPAGE = 0.0005


def _contract(asset, secid):
    df = pd.read_csv(data.DATA / "futures" / asset / f"{secid}.csv", parse_dates=["TRADEDATE"]).set_index("TRADEDATE")
    c = df["CLOSE"].where(df["CLOSE"] > 0).fillna(df["SETTLEPRICE"].where(df["SETTLEPRICE"] > 0))
    out = pd.DataFrame({"C": c})
    for k, col in (("O", "OPEN"), ("H", "HIGH"), ("L", "LOW")):
        out[k] = df[col].where(df[col] > 0).fillna(c)
    out["OI"] = df["OPENPOSITION"]
    return out.dropna(subset=["C"])


@lru_cache(None)
def chain(asset):
    """(front, next, roll) by trading day: front/next = dict of OHLC frames per contract role, plus expiries."""
    cal = data.calendar()
    series = pd.read_csv(data.DATA / "futures" / asset / "_series.csv", parse_dates=["expiration_date"]).dropna(subset=["expiration_date"])
    contracts = {}
    for _, s in series.iterrows():
        try:
            contracts[s["secid"]] = (s["expiration_date"], _contract(asset, s["secid"]).reindex(cal))
        except FileNotFoundError:
            continue
    contracts = {sid: v for sid, v in contracts.items() if v[1]["C"].notna().sum() > 5}
    oi = pd.DataFrame({sid: df["OI"] for sid, (e, df) in contracts.items()}).fillna(0)
    expiry = pd.Series({sid: e for sid, (e, df) in contracts.items()})
    front, nxt = pd.Series(index=cal, dtype=object), pd.Series(index=cal, dtype=object)
    for i in range(1, len(cal)):
        horizon = cal[min(i + ROLL_DAYS, len(cal) - 1)]
        live = expiry[expiry > horizon].index
        day_oi = oi.iloc[i - 1][live]
        day_oi = day_oi[day_oi > 0]
        if day_oi.empty:
            continue
        f = day_oi.idxmax()
        front.iloc[i] = f
        later = day_oi[expiry[day_oi.index] > expiry[f]]
        nxt.iloc[i] = later.idxmax() if not later.empty else None
    return contracts, front, nxt


@lru_cache(None)
def continuous(asset):
    """Back-adjusted OHLC of the contract held each day; returns never jump at rolls."""
    contracts, front, _ = chain(asset)
    cal = data.calendar()
    held = front.shift(1)  # position chosen yesterday earns today's move
    ret = pd.Series(np.nan, index=cal)
    rows = {}
    for i, d in enumerate(cal):
        sid = held.iloc[i]
        if not isinstance(sid, str):
            continue
        df = contracts[sid][1]
        c_prev, c_now = df["C"].iloc[i - 1], df["C"].iloc[i]
        ret.iloc[i] = c_now / c_prev - 1 if c_prev > 0 and c_now > 0 else 0.0
        rows[d] = df[["O", "H", "L", "C"]].iloc[i] / c_now if c_now > 0 else None
    level = (1 + ret.fillna(0)).cumprod().where(held.notna())
    rel = pd.DataFrame({d: r for d, r in rows.items() if r is not None}).T.reindex(cal)
    out = rel.mul(level, axis=0).dropna()
    roll = (front != front.shift(1)) & front.notna() & front.shift(1).notna()
    return out, roll


@lru_cache(None)
def carry(asset):
    """Annualised roll yield ln(front/next) per year to the next expiry: >0 is backwardation."""
    contracts, front, nxt = chain(asset)
    cal = data.calendar()
    vals = pd.Series(np.nan, index=cal)
    for i in range(len(cal)):
        f, n = front.iloc[i], nxt.iloc[i]
        if isinstance(f, str) and isinstance(n, str):
            (ef, df), (en, dn) = contracts[f], contracts[n]
            pf, pn = df["C"].iloc[i], dn["C"].iloc[i]
            years = (en - ef).days / 365
            if pf > 0 and pn > 0 and years > 0:
                vals.iloc[i] = np.log(pf / pn) / years
    return vals.ffill()


def fut(asset):
    ohlc, roll = continuous(asset)
    return Asset(ohlc, slippage=SLIPPAGE, futures=True, roll=roll)


def close(asset):
    return continuous(asset)[0]["C"].reindex(data.calendar())


def long_short_positions(entry_long, exit_long, entry_short, exit_short):
    return positions(entry_long, exit_long) - positions(entry_short, exit_short)


def donchian_ls(asset, n_in=55, n_out=20):
    df = continuous(asset)[0].reindex(data.calendar()).ffill()
    hi, lo = df["H"].rolling(n_in).max().shift(), df["L"].rolling(n_in).min().shift()
    ex_lo, ex_hi = df["L"].rolling(n_out).min().shift(), df["H"].rolling(n_out).max().shift()
    w = long_short_positions(df["C"] > hi, df["C"] < ex_lo, df["C"] < lo, df["C"] > ex_hi)
    return pd.DataFrame({asset: w}), {asset: fut(asset)}


def tsmom(assets, lookback=252, target_vol=None):
    cal = data.calendar()
    w = {}
    for a in assets:
        c = close(a).ffill()
        sig = np.sign(c / c.shift(lookback) - 1)
        if target_vol:
            vol = c.pct_change().rolling(60).std() * np.sqrt(252)
            sig = sig * (target_vol / vol).clip(upper=2)
        w[a] = on_dates(sig, month_ends(cal)).fillna(0)
    w = pd.DataFrame(w)
    if len(assets) > 1:
        w = w / len(assets) * 2  # cap gross exposure around 2x capital across the book
    return w, {a: fut(a) for a in assets}


def seasonal(asset, months_days, sign):
    """Hold `sign` (+1 long / −1 short) between (month, day) start and end, every year."""
    cal = data.calendar()
    (m0, d0), (m1, d1) = months_days
    md = cal.month * 100 + cal.day
    flag = (md >= m0 * 100 + d0) & (md <= m1 * 100 + d1)
    w = pd.Series(np.where(flag, sign, 0.0), index=cal).shift(-1).fillna(0)
    return pd.DataFrame({asset: w}), {asset: fut(asset)}


def s81():
    df = continuous("MIX")[0].reindex(data.calendar()).ffill()
    vol = df["C"].pct_change().rolling(20).std() * np.sqrt(252)
    w = (0.15 / vol).clip(upper=1).where(df["C"] > sma(df["C"], 100), 0.0)
    w = (w * 4).round() / 4
    return pd.DataFrame({"MIX": w.fillna(0)}), {"MIX": fut("MIX")}


def s86():
    cal = data.calendar()
    book = ["BR", "NG", "GOLD", "SILV"]
    cy = pd.DataFrame({a: carry(a) for a in book})
    rank = cy.rank(axis=1)
    n = cy.notna().sum(axis=1)
    w = pd.DataFrame(0.0, index=cal, columns=book)
    w[rank.eq(n, axis=0)] = 0.5
    w[(rank == 1) & (n >= 2)] = -0.5
    w = on_dates(w, month_ends(cal)).fillna(0)
    return w, {a: fut(a) for a in book}


def s87():
    """Collect the funding of the perpetual USDRUBF: be on the side that receives it (5-day mean)."""
    df = pd.read_csv(data.DATA / "futures" / "perpetual" / "USDRUBF.csv", parse_dates=["TRADEDATE"]).set_index("TRADEDATE")
    cal = data.calendar()
    c = df["CLOSE"].fillna(df["SETTLEPRICE"]).reindex(cal).ffill()
    swap = df["SWAPRATE"].fillna(0).reindex(cal).fillna(0)
    side = -np.sign(swap.rolling(5).mean()).fillna(0)  # positive swap: longs pay, so stay short
    pos = side.shift(1).fillna(0)
    ret = pos * (c.pct_change().fillna(0)) - pos * swap / c.shift(1)
    turnover = pos.diff().abs().fillna(0)
    costs = turnover * (0.0005 + SLIPPAGE)
    ret = ret.fillna(0) - costs + money_returns()
    listed = c.notna()
    ret = ret.where(listed, money_returns())
    return Result(ret, pos.abs(), turnover, costs)


def s90():
    cal = data.calendar()
    g, s = close("GOLD").ffill(), close("SILV").ffill()
    ratio = np.log(g / s)
    z = (ratio - ratio.rolling(250).mean()) / ratio.rolling(250).std()
    long_silver = positions(z > 2, z < 0)
    short_silver = positions(z < -2, z > 0)
    side = long_silver - short_silver
    return pd.DataFrame({"SILV": 0.5 * side, "GOLD": -0.5 * side}), {"SILV": fut("SILV"), "GOLD": fut("GOLD")}


STRATEGIES = {
    "76": lambda: donchian_ls("Si"), "77": lambda: donchian_ls("CNY"),
    "78": lambda: tsmom(["BR"]), "79": lambda: tsmom(["GOLD"]),
    "80": lambda: tsmom(["Si", "CNY", "BR", "GOLD", "MIX", "NG"], target_vol=0.15),
    "81": s81, "82": lambda: seasonal("NG", ((8, 25), (11, 15)), 1), "83": lambda: seasonal("NG", ((3, 1), (4, 30)), -1),
    "84": lambda: seasonal("BR", ((2, 15), (4, 30)), 1), "86": s86, "87": s87, "90": s90,
}


def run(sid):
    out = STRATEGIES[sid]()
    return out if isinstance(out, Result) else simulate(*out)


def main(ids):
    catalog, rows, results = names(), [], {}
    for sid in ids:
        res = run(sid)
        results[sid] = res
        row = {"id": sid, "name": catalog.get(sid, sid)}
        for p in ("train", "valid"):
            for k, v in metrics(res, *PERIODS[p]).items():
                row[f"{p}_{k}"] = v
        rows.append(row)
        print(f"{sid:>4} {row['name'][:45]:45} train SR {row.get('train_sharpe', np.nan):5.2f}  valid SR {row.get('valid_sharpe', np.nan):5.2f}", flush=True)
    table = pd.DataFrame(rows).set_index("id")
    ok = table["train_sharpe"].fillna(-9) > FINALIST_SHARPE
    ok &= table["valid_sharpe"].fillna(-9) > FINALIST_SHARPE
    for sid in table.index[ok]:
        for k, v in metrics(results[sid], *PERIODS["test"]).items():
            table.loc[sid, f"test_{k}"] = v
    table.to_csv(ROOT / "results" / "stage2.csv", float_format="%.4f")
    print(f"\nфиналисты: {list(table.index[ok])}")
    return table


if __name__ == "__main__":
    main(sys.argv[1:] or list(STRATEGIES))
