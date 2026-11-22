"""Load the downloaded MOEX history into aligned, adjusted OHLC frames."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

# old ticker -> ticker after redomiciliation; the gap between them is treated as flat
RENAMES = {"YNDX": "YDEX", "TCSG": "T", "FIVE": "X5", "MAIL": "VKCO", "HHRU": "HEAD", "AGRO": "RAGR", "FIXP": "FIXR"}
CLOSED_2022 = ("2022-02-25", "2022-03-24")  # exchange closed between these dates
SPLIT_RATIO = 5  # no share moves 5x in a day on its own, so a jump that large is a split or consolidation


def _read(path, close_cols=("CLOSE",)):
    df = pd.read_csv(path, parse_dates=["TRADEDATE"]).drop_duplicates("TRADEDATE").set_index("TRADEDATE").sort_index()
    close = df[close_cols[0]]
    for col in close_cols[1:]:
        close = close.fillna(df[col])
    out = pd.DataFrame({"C": close.astype(float)})
    for k, col in (("O", "OPEN"), ("H", "HIGH"), ("L", "LOW")):
        out[k] = df[col].astype(float) if col in df else np.nan
    out["V"] = df["VALUE"].astype(float) if "VALUE" in df else np.nan
    return out


def _fill(df):
    df["C"] = df["C"].where(df["C"] > 0)
    df = df[df["C"].notna() | (df["O"] > 0)].copy()
    df["C"] = df["C"].ffill()
    for k in "OHL":
        df[k] = df[k].where(df[k] > 0).fillna(df["C"])
    df["V"] = df["V"].fillna(0)
    return df.dropna(subset=["C"])


@lru_cache(None)
def calendar():
    return _read(DATA / "indices" / "IMOEX.csv").index


@lru_cache(None)
def index(secid):
    return _fill(_read(DATA / "indices" / f"{secid}.csv"))


@lru_cache(None)
def fx(secid):
    return _fill(_read(DATA / "fx" / f"{secid}.csv"))


@lru_cache(None)
def money_rate():
    """Annual money-market rate in %, by trading day: RUSFAR, before 2018 the CBR key rate."""
    key = pd.read_csv(DATA / "calendar" / "cbr_key_rate.csv", parse_dates=["date"]).set_index("date")["key_rate"]
    rusfar = index("RUSFAR")["C"]
    rate = rusfar.reindex(calendar()).fillna(key.reindex(calendar(), method="ffill"))
    return rate.ffill().fillna(5.5)


@lru_cache(None)
def index_fund():
    """IMOEX tracker: IMOEX intraday moves scaled to the total-return index MCFTR."""
    imoex, tr = index("IMOEX"), index("MCFTR")["C"].reindex(index("IMOEX").index).ffill()
    scale = tr / imoex["C"]
    out = imoex.copy()
    for k in "OHLC":
        out[k] = imoex[k] * scale
    return out


def _dividends():
    d = pd.read_csv(DATA / "dividends" / "dividends.csv", parse_dates=["last_day_with_dividend", "record_date"])
    d["dividend_rub"] = pd.to_numeric(d["dividend_rub"], errors="coerce")
    cancelled = pd.read_csv(DATA / "dividends_cancelled.csv", parse_dates=["last_day_with_dividend"])
    key = d.set_index(["secid", "last_day_with_dividend"]).index
    d = d[~key.isin(cancelled.set_index(["secid", "last_day_with_dividend"]).index)]
    return d.dropna(subset=["dividend_rub"])


@lru_cache(None)
def dividends():
    return _dividends()


def _adjust_splits(df):
    ratio = df["C"] / df["C"].shift()
    for date, r in ratio[(ratio >= SPLIT_RATIO) | (ratio <= 1 / SPLIT_RATIO)].items():
        before = df.index < date
        df.loc[before, list("OHLC")] *= r
    return df


def _adjust_dividends(df, secid):
    divs = dividends()
    divs = divs[divs["secid"] == secid]
    for _, d in divs.iterrows():
        last = d["last_day_with_dividend"]
        if pd.isna(last):
            continue
        after = df.index[df.index > last]
        if after.empty or last < df.index[0]:
            continue
        ex = after[0]
        prev_close = df.loc[df.index < ex, "C"].iloc[-1]
        factor = (prev_close - d["dividend_rub"]) / prev_close
        if 0.3 < factor < 1:
            df.loc[df.index < ex, list("OHLC")] *= factor
    return df


@lru_cache(None)
def share(secid, total_return=True):
    """Daily OHLCV of a share on TQBR, renamed tickers joined, split- and (optionally) dividend-adjusted."""
    old = [o for o, n in RENAMES.items() if n == secid]
    df = _fill(_read(DATA / "shares" / f"{secid}.csv", ("CLOSE", "LEGALCLOSEPRICE")))
    df = _adjust_splits(df)
    if total_return:
        df = _adjust_dividends(df, secid)
    for o in old:
        path = DATA / "shares" / f"{o}.csv"
        if not path.exists():
            continue
        prev = _adjust_splits(_fill(_read(path, ("CLOSE", "LEGALCLOSEPRICE"))))
        if total_return:
            prev = _adjust_dividends(prev, o)
        prev = prev[prev.index < df.index[0]]
        prev[list("OHLC")] *= df["O"].iloc[0] / prev["C"].iloc[-1]
        df = pd.concat([prev, df])
    return df


def all_share_ids():
    ids = {p.stem for p in (DATA / "shares").glob("*.csv")}
    return sorted(ids - set(RENAMES))


@lru_cache(None)
def composition():
    """IMOEX members by quarter: {quarter_date: [secid,...]} with renamed tickers mapped to the new ones."""
    c = pd.read_csv(DATA / "index_composition" / "IMOEX.csv", parse_dates=["tradedate"])
    c["secid"] = c["secids"].map(lambda s: RENAMES.get(s, s))
    return {d: sorted(set(g["secid"])) for d, g in c.groupby("tradedate")}


def members_on(date):
    """Index members known on `date` (the latest quarter composition not after it)."""
    comp = composition()
    known = [d for d in comp if d <= date]
    return comp[max(known)] if known else comp[min(comp)]


@lru_cache(None)
def panel(field, total_return=True):
    """Wide frame (calendar x share) of one OHLCV field."""
    cols = {}
    for s in all_share_ids():
        try:
            cols[s] = share(s, total_return)[field]
        except Exception:
            continue
    return pd.DataFrame(cols).reindex(calendar())


@lru_cache(None)
def membership():
    """Boolean frame (calendar x share): share was in IMOEX per the latest known quarterly composition."""
    cal, comp = calendar(), composition()
    m = pd.DataFrame(False, index=cal, columns=panel("C").columns)
    dates = sorted(comp)
    for i, d in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else cal[-1] + pd.Timedelta(days=1)
        rows = (cal > d) & (cal <= end)
        cols = [s for s in comp[d] if s in m.columns]
        m.loc[rows, cols] = True
    first = dates[0]
    m.loc[cal <= first, [s for s in comp[first] if s in m.columns]] = True
    return m


@lru_cache(None)
def futures_expirations(asset="RTS"):
    s = pd.read_csv(DATA / "futures" / asset / "_series.csv", parse_dates=["expiration_date"])
    return sorted(s["expiration_date"].dropna().unique())


@lru_cache(None)
def cbr_meetings():
    m = pd.read_csv(DATA / "calendar" / "cbr_meetings.csv", parse_dates=["meeting_date"])["meeting_date"]
    return sorted(m[m <= calendar()[-1]])


@lru_cache(None)
def key_rate_changes():
    return pd.read_csv(DATA / "calendar" / "cbr_key_rate_changes.csv", parse_dates=["effective_date"])
