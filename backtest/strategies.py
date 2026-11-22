"""Stage-1 strategies from strategies.yaml (daily index and share data). Each returns target weights or a Result."""
import numpy as np
import pandas as pd

from . import data
from .engine import Asset, Result, simulate

TRAIN_END = "2019-12-31"
MAX_POSITIONS = 5
LIQUID_N = 10

# ---------- assets ----------

def fund():
    return Asset(data.index_fund(), slippage=0.0005, annual_fee=0.008)


def bonds():
    return Asset(data.index("RGBITR"), slippage=0.0005, annual_fee=0.005)


def gold():
    return Asset(data.fx("GLDRUB_TOM"), slippage=0.001, annual_fee=0.008)


def small_caps():
    return Asset(data.index("MCXSM"), slippage=0.002, annual_fee=0.01)


def share_assets(cols):
    return {s: Asset(data.share(s), slippage=0.001) for s in cols}


# ---------- indicators ----------

def sma(s, n):
    return s.rolling(n).mean()


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n):
    tr = pd.concat([df["H"] - df["L"], (df["H"] - df["C"].shift()).abs(), (df["L"] - df["C"].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def month_ends(idx):
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max())


def on_dates(frame_or_series, dates):
    """Keep values only on decision dates and hold them until the next one."""
    out = frame_or_series.astype(float).copy()
    out.loc[~out.index.isin(dates)] = np.nan
    return out.ffill()


# ---------- position state machine for entry/exit rules ----------

def positions(entry, exit_, time_stop=None, stop_pct=None, close=None):
    """1 while in a position: enter on `entry`, leave on `exit_`, after `time_stop` days, or below `stop_pct`."""
    e, x = entry.fillna(False).values, exit_.fillna(False).values
    c = close.values if close is not None else None
    out = np.zeros(len(e))
    inside, held, entry_px = False, 0, 0.0
    for i in range(len(e)):
        if inside:
            held += 1
            stopped = stop_pct is not None and c is not None and c[i] < entry_px * (1 - stop_pct)
            if x[i] or (time_stop and held >= time_stop) or stopped:
                inside = False
        elif e[i]:
            inside, held = True, 0
            entry_px = c[i] if c is not None else 0.0
        out[i] = 1.0 if inside else 0.0
    return pd.Series(out, index=entry.index)


def on_fund(rule, **kw):
    df = data.index_fund()
    entry, exit_ = rule(df)
    pos = positions(entry, exit_, close=df["C"], **kw)
    return pd.DataFrame({"FUND": pos}), {"FUND": fund()}


def liquid_universe(n=LIQUID_N):
    """Top-n IMOEX members by 60-day traded value, refreshed monthly."""
    value = data.panel("V").rolling(60).mean().where(data.membership())
    rank = value.rank(axis=1, ascending=False)
    top = (rank <= n).where(value.notna(), False)
    return on_dates(top.astype(float), month_ends(top.index)).fillna(0).astype(bool)


def on_shares(rule, universe=None, max_pos=MAX_POSITIONS, **kw):
    universe = liquid_universe() if universe is None else universe
    pos = {}
    for s in universe.columns[universe.any()]:
        df = data.share(s).reindex(data.calendar())
        if df["C"].notna().sum() < 250:
            continue
        df = df.ffill()
        entry, exit_ = rule(df)
        entry = entry & universe[s]
        pos[s] = positions(entry, exit_, close=df["C"], **kw)
    pos = pd.DataFrame(pos).fillna(0)
    weights = pos.div(np.maximum(pos.sum(axis=1), max_pos), axis=0)
    return weights, share_assets(weights.columns)


def top_n(score, dates, n=MAX_POSITIONS):
    """Equal weight in the n best scores on each decision date, held until the next one."""
    score = score.where(data.membership())
    rank = score.rank(axis=1, ascending=False)
    pick = ((rank <= n) & score.notna()).astype(float)
    w = pick.div(pick.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    w = on_dates(w, dates).fillna(0)
    w = w.loc[:, w.any()]
    return w, share_assets(w.columns)


def returns_result(ret, exposure, turnover, costs):
    return Result(ret.fillna(0), exposure, turnover, costs)


# ---------- rules (df has O,H,L,C) ----------

def r_golden(df):
    up = sma(df["C"], 50) > sma(df["C"], 200)
    return up, ~up


def r_donchian(n_in, n_out):
    def rule(df):
        return df["C"] > df["H"].rolling(n_in).max().shift(), df["C"] < df["L"].rolling(n_out).min().shift()
    return rule


def r_macd(df):
    macd = df["C"].ewm(span=12, adjust=False).mean() - df["C"].ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    return (macd > sig) & (df["C"] > sma(df["C"], 200)), macd < sig


def r_adx(df, n=14):
    up, dn = df["H"].diff(), -df["L"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    adx = (100 * (pdi - mdi).abs() / (pdi + mdi)).ewm(alpha=1 / n, adjust=False).mean()
    return (adx > 25) & (pdi > mdi), (mdi > pdi) | (adx < 20)


def r_supertrend(df, n=10, mult=3):
    a, hl2 = atr(df, n), (df["H"] + df["L"]) / 2
    upper, lower = (hl2 + mult * a).to_numpy(copy=True), (hl2 - mult * a).to_numpy(copy=True)
    c = df["C"].to_numpy()
    trend = np.ones(len(c))
    for i in range(1, len(c)):
        if c[i - 1] <= upper[i - 1]:
            upper[i] = min(upper[i], upper[i - 1]) if not np.isnan(upper[i - 1]) else upper[i]
        if c[i - 1] >= lower[i - 1]:
            lower[i] = max(lower[i], lower[i - 1]) if not np.isnan(lower[i - 1]) else lower[i]
        trend[i] = 1 if c[i] > upper[i - 1] else -1 if c[i] < lower[i - 1] else trend[i - 1]
    t = pd.Series(trend, index=df.index)
    return t > 0, t < 0


def r_keltner(df):
    mid = df["C"].ewm(span=20, adjust=False).mean()
    return df["C"] > mid + 2 * atr(df, 10), df["C"] < mid


def r_boll_break(df):
    m, sd = sma(df["C"], 20), df["C"].rolling(20).std()
    return df["C"] > m + 2 * sd, df["C"] < m


def r_nr7(df):
    rng = df["H"] - df["L"]
    nr7 = rng == rng.rolling(7).min()
    return nr7.shift(1, fill_value=False) & (df["C"] > df["H"].shift()), df["C"] < df["L"].shift(1)


def r_inside(df):
    inside = (df["H"] < df["H"].shift()) & (df["L"] > df["L"].shift())
    entry = inside.shift(1, fill_value=False) & (df["C"] > df["H"].shift()) & (df["C"] > sma(df["C"], 50))
    return entry, df["C"] < df["L"].shift(2)


def uptrend(df):
    return df["C"] > sma(df["C"], 200)


def r_rsi2(df):
    r = rsi(df["C"], 2)
    return uptrend(df) & (r < 10), (r > 70) | (df["C"] > sma(df["C"], 5))


def r_double7(df):
    return uptrend(df) & (df["C"] <= df["C"].rolling(7).min()), df["C"] >= df["C"].rolling(7).max()


def r_ibs(df):
    ibs = (df["C"] - df["L"]) / (df["H"] - df["L"]).replace(0, np.nan)
    return uptrend(df) & (ibs < 0.2), (ibs > 0.8) | (df["C"] > df["H"].shift())


def r_three_down(df):
    down = df["C"] < df["C"].shift()
    return uptrend(df) & down & down.shift(1, fill_value=False) & down.shift(2, fill_value=False), df["C"] > sma(df["C"], 5)


def r_cum_rsi2(df):
    r = rsi(df["C"], 2)
    return uptrend(df) & (r + r.shift() < 35), r > 65


def r_boll_low(df):
    m, sd = sma(df["C"], 20), df["C"].rolling(20).std()
    return uptrend(df) & (df["C"] < m - 2 * sd), df["C"] > m


def r_zscore(df):
    z = (df["C"] - sma(df["C"], 20)) / df["C"].rolling(20).std()
    return uptrend(df) & (z < -2), z > 0


def r_williams(df):
    wr = -100 * (df["H"].rolling(2).max() - df["C"]) / (df["H"].rolling(2).max() - df["L"].rolling(2).min()).replace(0, np.nan)
    return uptrend(df) & (wr < -90), wr > -30


def r_crash(df):
    return df["C"].pct_change() < -0.03, pd.Series(False, index=df.index)


# ---------- strategies ----------

def s1():
    return pd.DataFrame({"FUND": 1.0}, index=data.calendar()), {"FUND": fund()}


def s2():
    return pd.DataFrame({"FUND": 0.0}, index=data.calendar()), {"FUND": fund()}


def s3():
    return pd.DataFrame({"FUND": 0.6, "BONDS": 0.4}, index=data.calendar()), {"FUND": fund(), "BONDS": bonds()}


def monthly_fund_close():
    c = data.index_fund()["C"]
    return c, month_ends(c.index)


def s4():
    c, me = monthly_fund_close()
    m = c.loc[me]
    sig = (m > m.rolling(10).mean()).astype(float).reindex(c.index).ffill().fillna(0)
    return pd.DataFrame({"FUND": sig}), {"FUND": fund()}


def money_index():
    from .engine import money_returns
    return (1 + money_returns()).cumprod()


def s5():
    c, me = monthly_fund_close()
    mi = money_index().reindex(c.index)
    sig = ((c / c.shift(252)) > (mi / mi.shift(252))).astype(float)
    return pd.DataFrame({"FUND": on_dates(sig, me).fillna(0)}), {"FUND": fund()}


def s6():
    cal = data.calendar()
    closes = pd.DataFrame({"FUND": data.index_fund()["C"], "GOLD": data.fx("GLDRUB_TOM")["C"],
                           "BONDS": data.index("RGBITR")["C"]}).reindex(cal).ffill()
    mi = money_index().reindex(cal)
    mom = closes / closes.shift(252) - 1
    best = mom.fillna(-np.inf).idxmax(axis=1)
    beats = mom.max(axis=1) > (mi / mi.shift(252) - 1)
    w = pd.DataFrame(0.0, index=cal, columns=closes.columns)
    for col in closes.columns:
        w[col] = ((best == col) & beats).astype(float)
    w = on_dates(w, month_ends(cal)).fillna(0)
    return w, {"FUND": fund(), "GOLD": gold(), "BONDS": bonds()}


def s10():
    df = data.index_fund()
    vol = df["C"].pct_change().rolling(20).std() * np.sqrt(252)
    w = (0.15 / vol).clip(upper=1).where(uptrend(df), 0.0)
    w = (w * 4).round() / 4  # 25% steps so daily vol noise does not churn the position
    return pd.DataFrame({"FUND": w.fillna(0)}), {"FUND": fund()}


def s11():
    c = data.panel("C")
    return top_n(c.shift(21) / c.shift(252) - 1, month_ends(c.index))


def s12():
    c = data.panel("C")
    me = [d for d in month_ends(c.index) if d.month in (3, 6, 9, 12)]
    return top_n(c.shift(21) / c.shift(84) - 1, me)


def s13():
    c, h = data.panel("C"), data.panel("H")
    return top_n(c / h.rolling(252, min_periods=200).max(), month_ends(c.index))


SECTORS = ["MOEXOG", "MOEXFN", "MOEXMM", "MOEXEU", "MOEXTL", "MOEXCN", "MOEXCH", "MOEXTN"]


def s14():
    cal = data.calendar()
    c = pd.DataFrame({s: data.index(s)["C"] for s in SECTORS}).reindex(cal).ffill()
    score = c / c.shift(126) - 1
    pick = (score.rank(axis=1, ascending=False) <= 2).astype(float) / 2
    w = on_dates(pick.where(score.notna().all(axis=1), 0), month_ends(cal)).fillna(0)
    return w, {s: Asset(data.index(s), slippage=0.001, annual_fee=0.008) for s in SECTORS}


def s15():
    imoex = data.index("IMOEX")["C"].reindex(data.calendar()).ffill()

    def rule(df):
        ratio = df["C"] / imoex
        return (ratio >= ratio.rolling(50).max()) & (df["C"] > sma(df["C"], 50)), ratio < sma(ratio, 20)
    return on_shares(rule)


def s26():
    c = data.panel("C")
    vol = c.pct_change().rolling(252, min_periods=200).std()
    me = [d for d in month_ends(c.index) if d.month in (3, 6, 9, 12)]
    return top_n(-vol, me)


def s28():
    c = data.panel("C", total_return=False)
    divs = data.dividends()
    paid = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for _, d in divs.iterrows():
        if d["secid"] in paid.columns and pd.notna(d["last_day_with_dividend"]):
            after = paid.index[paid.index > d["last_day_with_dividend"]]
            if len(after):
                paid.loc[after[0], d["secid"]] += d["dividend_rub"]
    trailing = paid.rolling(252, min_periods=1).sum()
    ye = [d for d in month_ends(c.index) if d.month == 12]
    return top_n(trailing / c, ye)


def s29():
    cal = data.calendar()
    sc, ix = data.index("MCXSM")["C"].reindex(cal).ffill(), data.index("IMOEX")["C"].reindex(cal).ffill()
    small = ((sc / sc.shift(126)) > (ix / ix.shift(126))).astype(float)
    small = on_dates(small, month_ends(cal)).fillna(0)
    return pd.DataFrame({"SMALL": small, "FUND": 1 - small}), {"SMALL": small_caps(), "FUND": fund()}


def s41():
    c = data.panel("C")
    fridays = c.index[c.index.weekday == 4]
    universe = liquid_universe(15)
    score = -(c / c.shift(5) - 1)
    score = score.where(universe)
    rank = score.rank(axis=1, ascending=False)
    pick = ((rank <= 3) & score.notna()).astype(float) / 3
    w = on_dates(pick, fridays).fillna(0)
    w = w.loc[:, w.any()]
    return w, share_assets(w.columns)


def s42():
    c = data.panel("C")
    mom = (c / c.shift(126) - 1).where(data.membership())
    leaders = on_dates((mom.rank(axis=1, ascending=False) <= 10).astype(float), month_ends(c.index)).fillna(0).astype(bool)
    imoex = data.index("IMOEX")["C"].reindex(c.index).ffill()
    market_up = imoex > sma(imoex, 200)

    def rule(df):
        down = df["C"] < df["C"].shift()
        three = down & down.shift(1, fill_value=False) & down.shift(2, fill_value=False)
        return market_up & ((rsi(df["C"], 2) < 10) | three), df["C"] > sma(df["C"], 5)
    return on_shares(rule, universe=leaders, time_stop=7, stop_pct=0.08)


def intraday_fund(mask_fn, overnight=False):
    """Return series of a one-day trade on the index fund when mask_fn(df) is true (decided with today's open)."""
    df = data.index_fund().reindex(data.calendar()).ffill()
    trade = mask_fn(df).fillna(False)
    leg = df["O"] / df["C"].shift() - 1 if overnight else df["C"] / df["O"] - 1
    from .engine import COST_MULT
    cost = 2 * (0.003 + 0.0005) * COST_MULT
    ret = (leg - cost).where(trade, 0.0)
    from .engine import money_returns
    ret = ret + money_returns().where(~trade, 0.0)
    return returns_result(ret, trade.astype(float), trade.astype(float) * 2, trade.astype(float) * cost)


def s40():
    return intraday_fund(lambda df: df["O"] < df["L"].shift() * 0.99)


def s44():
    return intraday_fund(lambda df: pd.Series(True, index=df.index), overnight=True)


def s45():
    return intraday_fund(lambda df: pd.Series(True, index=df.index))


def calendar_fund(flag):
    """Hold the index fund on days where `flag` is true (decided the evening before)."""
    return pd.DataFrame({"FUND": flag.astype(float).shift(-1).fillna(0)}), {"FUND": fund()}


def trading_days():
    return pd.Series(data.calendar(), index=data.calendar())


def s49():
    cal = trading_days()
    g = cal.groupby([cal.index.year, cal.index.month])
    pos_from_end = g.cumcount(ascending=False)
    pos_from_start = g.cumcount()
    return calendar_fund((pos_from_end == 0) | (pos_from_start <= 2))


def holidays():
    cal = data.calendar()
    weekdays = pd.bdate_range(cal[0], cal[-1])
    closed = (weekdays > data.CLOSED_2022[0]) & (weekdays < data.CLOSED_2022[1])
    return weekdays[~weekdays.isin(cal) & ~closed]


def s50():
    cal = data.calendar()
    before = set()
    for h in holidays():
        prev = cal[cal < h]
        if len(prev):
            before.add(prev[-1])
    return calendar_fund(pd.Series(cal.isin(list(before)), index=cal))


def s51():
    cal = trading_days()
    jan_n = cal.groupby([cal.index.year, cal.index.month]).cumcount()
    return calendar_fund(pd.Series((cal.index.month == 12) | ((cal.index.month == 1) & (jan_n < 10)), index=cal.index))


def s52():
    cal = data.calendar()
    return calendar_fund(pd.Series(~cal.month.isin([5, 6, 7, 8, 9, 10]), index=cal))


def s53():
    c = data.index_fund()["C"]
    monthly = c.resample("ME").last().pct_change()
    train = monthly[:TRAIN_END]
    best = train.groupby(train.index.month).mean().nlargest(4).index
    cal = data.calendar()
    return calendar_fund(pd.Series(cal.month.isin(best), index=cal))


def s54():
    c = data.index_fund()["C"]
    r = c.pct_change()[:TRAIN_END]
    best = r.groupby(r.index.weekday).mean().idxmax()
    cal = data.calendar()
    return calendar_fund(pd.Series(cal.weekday == best, index=cal))


def s55():
    cal = data.calendar()
    flag = pd.Series(((cal.month == 12) & (cal.day >= 15)) | (cal.month == 1), index=cal)
    return pd.DataFrame({"SMALL": flag.astype(float).shift(-1).fillna(0)}), {"SMALL": small_caps()}


def s56():
    cal = data.calendar()
    exp = [pd.Timestamp(e) for e in data.futures_expirations("RTS") if pd.Timestamp(e).month in (3, 6, 9, 12)]
    flag = pd.Series(False, index=cal)
    for e in exp:
        monday = e - pd.Timedelta(days=e.weekday())
        flag[(cal >= monday) & (cal <= e)] = True
    return calendar_fund(flag)


def s57():
    cal = trading_days()
    g = cal.groupby([cal.index.year, cal.index.month])
    last_dec = (cal.index.month == 12) & (g.cumcount(ascending=False) == 0)
    first_jan = (cal.index.month == 1) & (g.cumcount() <= 2)
    return calendar_fund(pd.Series(last_dec | first_jan, index=cal.index))


def s58():
    cal = data.calendar()
    flag = pd.Series(False, index=cal)
    for m in data.cbr_meetings():
        prev = cal[cal < m]
        if len(prev):
            flag[prev[-1]] = True
    return calendar_fund(flag)


def s59():
    cal = data.calendar()
    ch = data.key_rate_changes()
    cuts = ch[ch["key_rate"].diff() < 0]["effective_date"]
    flag = pd.Series(0.0, index=cal)
    for e in cuts:
        idx = cal.searchsorted(e)
        flag.iloc[idx:idx + 20] = 1.0
    return pd.DataFrame({"BONDS": flag.shift(-1).fillna(0)}), {"BONDS": bonds()}


def dividend_window(days_before, days_after, start_offset=0):
    cal = data.calendar()
    divs = data.dividends()
    w = {}
    for _, d in divs.iterrows():
        s, last = d["secid"], d["last_day_with_dividend"]
        if pd.isna(last) or s not in data.panel("C").columns:
            continue
        i = cal.searchsorted(last)
        if i >= len(cal) or cal[i] != last:
            continue
        lo, hi = max(i - days_before + start_offset, 0), min(i + days_after, len(cal))
        member = data.membership()[s] if s in data.membership() else None
        if member is None or not member.iloc[max(lo, 0)]:
            continue
        col = w.setdefault(s, pd.Series(0.0, index=cal))
        col.iloc[lo:hi] = 1.0
    pos = pd.DataFrame(w).fillna(0)
    weights = pos.div(np.maximum(pos.sum(axis=1), MAX_POSITIONS), axis=0)
    return weights, share_assets(weights.columns)


def s60():
    # hold from 20 days before the last day with dividend; selling at the next open forgoes the dividend
    return dividend_window(20, 0)


def s61():
    return dividend_window(0, 20, start_offset=1)


def s65():
    comp = data.composition()
    dates = sorted(comp)
    cal = data.calendar()
    w = {}
    for prev, cur in zip(dates, dates[1:]):
        for s in set(comp[cur]) - set(comp[prev]):
            if s not in data.panel("C").columns:
                continue
            i = cal.searchsorted(cur)
            col = w.setdefault(s, pd.Series(0.0, index=cal))
            col.iloc[max(i - 10, 0):i] = 1.0
    pos = pd.DataFrame(w).fillna(0)
    weights = pos.div(np.maximum(pos.sum(axis=1), MAX_POSITIONS), axis=0)
    return weights, share_assets(weights.columns)


def pair_switch(common, pref):
    cal = data.calendar()
    a, b = data.share(common)["C"].reindex(cal).ffill(), data.share(pref)["C"].reindex(cal).ffill()
    spread = np.log(b / a)
    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    hold_pref = pd.Series(np.nan, index=cal)
    hold_pref[z < -2] = 1.0
    hold_pref[z > 2] = 0.0
    hold_pref = hold_pref.ffill().fillna(0)
    listed = a.notna() & b.notna()
    return (pd.DataFrame({pref: hold_pref.where(listed, 0), common: (1 - hold_pref).where(listed, 0)}),
            share_assets([common, pref]))


def s73():
    cal = data.calendar()
    usd = data.fx("USD000UTSTOM")["C"].reindex(cal).ffill()
    ytd = usd / usd.groupby(cal.year).transform("first") - 1
    divs = data.dividends()
    lasts = sorted(divs[divs["secid"] == "SNGSP"]["last_day_with_dividend"].dropna())
    flag = pd.Series(0.0, index=cal)
    for year in sorted(set(cal.year)):
        q4 = (cal.year == year) & (cal.month >= 10) & (ytd > 0.10).values
        if not q4.any():
            continue
        start = cal[q4][0]
        exit_ = next((l for l in lasts if l.year == year + 1), None)
        if exit_ is None:
            continue
        i = cal.searchsorted(exit_) + 1  # hold through the ex-dividend day to receive the dividend
        flag[(cal >= start) & (cal <= cal[min(i, len(cal) - 1)])] = 1.0
    return pd.DataFrame({"SNGSP": flag}), share_assets(["SNGSP"])


def both(rule, **kw):
    return {"fund": lambda: on_fund(rule, **kw), "shares": lambda: on_shares(rule, **kw)}


STRATEGIES = {
    "1": s1, "2": s2, "3": s3, "4": s4, "5": s5, "6": s6,
    "7": lambda: on_fund(r_golden), "8": lambda: on_fund(r_donchian(20, 10)), "9": lambda: on_fund(r_donchian(55, 20)),
    "10": s10, "11": s11, "12": s12, "13": s13, "14": s14, "15": s15,
    "26": s26, "28": s28, "29": s29, "41": s41, "42": s42,
    "40": s40, "44": s44, "45": s45,
    "49": s49, "50": s50, "51": s51, "52": s52, "53": s53, "54": s54, "55": s55, "56": s56, "57": s57,
    "58": s58, "59": s59, "60": s60, "61": s61, "65": s65,
    "70": lambda: pair_switch("SBER", "SBERP"), "71": lambda: pair_switch("TATN", "TATNP"),
    "72": lambda: pair_switch("RTKM", "RTKMP"), "73": s73,
    "39": lambda: on_fund(r_crash, time_stop=3),
}
for sid, rule, kw in [
    ("16", r_macd, {}), ("17", r_adx, {}), ("18", r_supertrend, {}), ("19", r_keltner, {}), ("20", r_boll_break, {}),
    ("21", r_nr7, {"time_stop": 5}), ("22", r_inside, {"time_stop": 5}),
    ("31", r_rsi2, {"time_stop": 7, "stop_pct": 0.08}), ("32", r_double7, {}), ("33", r_ibs, {"time_stop": 5}),
    ("34", r_three_down, {"time_stop": 5}), ("35", r_cum_rsi2, {}), ("36", r_boll_low, {"time_stop": 10}),
    ("37", r_zscore, {}), ("38", r_williams, {}),
]:
    variants = both(rule, **kw)
    STRATEGIES[f"{sid}i"] = variants["fund"]
    STRATEGIES[f"{sid}s"] = variants["shares"]


def run(sid):
    out = STRATEGIES[sid]()
    if isinstance(out, Result):
        return out
    weights, assets = out
    return simulate(weights, assets)
