"""Strict check of strategy #60 (dividend run-up): trade-level simulation with lots on a small account.

python -m backtest.div_runup
"""
import numpy as np
import pandas as pd

from . import data
from .engine import COMMISSION, money_returns

CAPITAL = 10_000
SLIPPAGE = 0.001
PERIODS = {"train": ("2013-03-25", "2019-12-31"), "valid": ("2020-01-01", "2022-12-31"), "test": ("2023-01-01", None)}


def events(include_cancelled=True):
    """Dividend events with the board recommendation date (`announced`, NaT if unknown) and the planned amount."""
    divs = data.dividends()[["secid", "last_day_with_dividend", "record_date", "dividend_rub"]].copy()
    divs["planned"] = divs["dividend_rub"]
    ann = pd.read_csv(data.DATA / "dividends" / "announcements.csv", parse_dates=["announced", "record_date"])
    ann = ann.sort_values("record_date")
    divs = divs.dropna(subset=["record_date"]).sort_values("record_date")
    divs = pd.merge_asof(divs, ann[["secid", "record_date", "announced"]], on="record_date", by="secid",
                         direction="nearest", tolerance=pd.Timedelta(days=5))
    divs.loc[divs["announced"] >= divs["last_day_with_dividend"], "announced"] = pd.NaT  # old regime: AGM after record date
    if include_cancelled:
        canc = pd.read_csv(data.DATA / "dividends_cancelled.csv", parse_dates=["announced", "last_day_with_dividend", "record_date"])
        canc["planned"] = canc["planned_dividend_rub"]
        divs = pd.concat([divs, canc[["secid", "last_day_with_dividend", "record_date", "dividend_rub", "planned", "announced"]]])
    divs = divs.dropna(subset=["last_day_with_dividend"]).drop_duplicates(["secid", "last_day_with_dividend"])
    return divs.sort_values("last_day_with_dividend")


def lot_sizes():
    lots = pd.read_csv(data.DATA / "lot_sizes.csv").set_index("secid")["lot"]
    return {**{o: lots.get(n, 1) for o, n in data.RENAMES.items()}, **lots.to_dict()}


def trades(window, shift=0, include_cancelled=True, after_announcement=False, min_yield=0.0, min_days=3):
    """Entry at the open `window-1` days before the last day with dividend, exit at that day's open.

    `shift` moves every window by that many trading days (placebo: windows unrelated to dividends).
    `after_announcement`: never decide before the board recommendation; events without a known date are dropped.
    `min_yield`: planned dividend / price on the decision day."""
    cal, member = data.calendar(), data.membership()
    fund = data.index_fund()["O"].reindex(cal).ffill()
    rows = []
    for _, e in events(include_cancelled).iterrows():
        s, last = e["secid"], e["last_day_with_dividend"]
        if s not in member.columns:
            continue
        i = cal.searchsorted(last) + shift
        decide = i - window
        if after_announcement:
            if pd.isna(e["announced"]):
                continue
            decide = max(decide, cal.searchsorted(e["announced"]))
        entry, exit_ = decide + 1, i
        if decide < 0 or exit_ >= len(cal) or exit_ - entry < min_days or not member[s].iloc[decide]:
            continue
        raw = data.share(s, total_return=False)
        px = raw["O"].reindex(cal)
        p_in, p_out = px.iloc[entry], px.iloc[exit_]
        p_decide = raw["C"].reindex(cal).ffill().iloc[decide]
        if not (p_in > 0 and p_out > 0):
            continue
        dy = e["planned"] / p_decide
        if dy < min_yield:
            continue
        rows.append({"secid": s, "entry": cal[entry], "exit": cal[exit_], "p_in": p_in, "p_out": p_out, "dy": dy,
                     "ret": p_out / p_in - 1, "fund_ret": fund.iloc[exit_] / fund.iloc[entry] - 1,
                     "cancelled": e["dividend_rub"] == 0})
    t = pd.DataFrame(rows)
    t["excess"] = t["ret"] - t["fund_ret"]
    return t


def portfolio(t, slots=5, capital=CAPITAL, lots=True):
    """Daily equity of a cash account taking the trades in order, `slots` equal positions, whole lots."""
    cal = data.calendar()
    lot = lot_sizes()
    close = {s: data.share(s, total_return=False)["C"].reindex(cal).ffill() for s in t["secid"].unique()}
    mr = money_returns()
    by_entry = {d: g for d, g in t.groupby("entry")}
    by_exit = {d: g for d, g in t.groupby("exit")}
    cash, held, equity, taken, skipped = float(capital), {}, [], [], 0
    for d in cal:
        cash *= 1 + mr[d]
        for _, tr in by_exit.get(d, pd.DataFrame()).iterrows():
            key = (tr["secid"], tr["entry"])
            if key in held:
                qty = held.pop(key)
                cash += qty * tr["p_out"] * (1 - COMMISSION - SLIPPAGE)
        value = cash + sum(q * close[s][d] for (s, _), q in held.items())
        for _, tr in by_entry.get(d, pd.DataFrame()).iterrows():
            if len(held) >= slots:
                skipped += 1
                continue
            budget = min(cash, value / slots)
            unit = tr["p_in"] * (1 + COMMISSION + SLIPPAGE)
            qty = budget / unit
            if lots:
                step = lot.get(tr["secid"], 1)
                qty = np.floor(qty / step) * step
            if qty <= 0:
                skipped += 1
                continue
            cash -= qty * unit
            held[(tr["secid"], tr["entry"])] = qty
            taken.append(tr)
        equity.append(cash + sum(q * close[s][d] for (s, _), q in held.items()))
    eq = pd.Series(equity, index=cal)
    return eq, pd.DataFrame(taken), skipped


def stats(eq, start, end):
    e = eq[start:end]
    r = e.pct_change().dropna()
    m = money_returns()[r.index]
    years = (e.index[-1] - e.index[0]).days / 365.25
    ex = r - m
    return {"cagr": (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1, "money": (1 + m).prod() ** (1 / years) - 1,
            "sharpe": ex.mean() / ex.std() * np.sqrt(252), "max_dd": (e / e.cummax() - 1).min()}


def trade_stats(t):
    n = len(t)
    return {"trades": n, "win": (t["ret"] > 0).mean(), "mean_ret": t["ret"].mean(), "mean_excess": t["excess"].mean(),
            "t_excess": t["excess"].mean() / t["excess"].std() * np.sqrt(n) if n > 2 else np.nan}


def main():
    pd.set_option("display.width", 220)
    rows = []
    for window in (5, 10, 15, 20, 30, 40):
        t = trades(window)
        for slots in (3, 5):
            eq, taken, skipped = portfolio(t, slots=slots)
            row = {"window": window, "slots": slots, "skipped": skipped, **trade_stats(taken)}
            for p, (a, b) in PERIODS.items():
                st = stats(eq, a, b)
                row[f"{p}_cagr"], row[f"{p}_sharpe"], row[f"{p}_dd"] = st["cagr"], st["sharpe"], st["max_dd"]
            rows.append(row)
            print(f"window {window:2} slots {slots}: test SR {row['test_sharpe']:.2f}", flush=True)
    grid = pd.DataFrame(rows)

    base = trades(20)
    placebo = pd.DataFrame([{"shift": sh, **trade_stats(trades(20, shift=sh))} for sh in (0, -40, -60, -120, 40)])
    base["year"] = base["entry"].dt.year
    by_year = base.groupby("year").agg(trades=("ret", "size"), mean_ret=("ret", "mean"), mean_excess=("excess", "mean"),
                                       win=("ret", lambda r: (r > 0).mean()))
    by_stock = base.groupby("secid")["excess"].agg(["size", "mean", "sum"]).sort_values("sum", ascending=False)
    no_canc = trade_stats(trades(20, include_cancelled=False))
    canc = base[base["cancelled"]]

    eq, taken, skipped = portfolio(base, slots=5)
    eq_frac, _, _ = portfolio(base, slots=5, lots=False)
    out = data.DATA.parent / "results"
    grid.to_csv(out / "div_runup_grid.csv", float_format="%.4f")
    base.to_csv(out / "div_runup_trades_w20.csv", float_format="%.4f", index=False)
    pd.DataFrame({"lots_10k": eq, "fractional": eq_frac}).to_csv(out / "div_runup_equity_w20.csv", float_format="%.2f")

    print("\n== сетка окно × слоты (с лотами, 10 тыс. ₽)")
    print(grid.round(3).to_string(index=False))
    print("\n== плацебо: те же бумаги, окно сдвинуто от дивиденда")
    print(placebo.round(4).to_string(index=False))
    print("\n== по годам (окно 20)")
    print(by_year.round(4).to_string())
    print("\n== вклад бумаг (топ-8 и низ-5 по сумме избыточной доходности)")
    print(pd.concat([by_stock.head(8), by_stock.tail(5)]).round(4).to_string())
    print("\n== отменённые дивиденды в выборке:")
    print(canc[["secid", "entry", "exit", "ret", "excess"]].round(4).to_string(index=False))
    print("без отменённых:", {k: round(v, 4) for k, v in no_canc.items()})
    print(f"\nс лотами: пропущено {skipped}; итог 10 тыс. ₽ → {eq.iloc[-1]:,.0f} ₽ (дробные доли: {eq_frac.iloc[-1]:,.0f} ₽)")


def period_trade_stats(t):
    out = {}
    for p, (a, b) in PERIODS.items():
        s = t[(t["entry"] >= a) & ((t["entry"] <= b) if b else True)]
        st = trade_stats(s)
        out[f"{p}_n"], out[f"{p}_excess"], out[f"{p}_t"] = st["trades"], st["mean_excess"], st["t_excess"]
    return out


def strict():
    """Entry only after the board recommendation; yield threshold picked on train, then judged on valid and test."""
    pd.set_option("display.width", 220)
    rows = []
    for window in (15, 20, 30):
        for min_yield in (0.0, 0.03, 0.05, 0.07):
            t = trades(window, after_announcement=True, min_yield=min_yield)
            row = {"window": window, "min_yield": min_yield, **period_trade_stats(t)}
            for slots in (2, 3):
                eq, _, skipped = portfolio(t, slots=slots)
                for p, (a, b) in PERIODS.items():
                    row[f"{p}_sr_{slots}slots"] = stats(eq, a, b)["sharpe"]
                row[f"skipped_{slots}"] = skipped
                if slots == 3:
                    for p, (a, b) in PERIODS.items():
                        st = stats(eq, a, b)
                        row[f"{p}_cagr_3"], row[f"{p}_money"], row[f"{p}_dd_3"] = st["cagr"], st["money"], st["max_dd"]
            rows.append(row)
            print(f"window {window} yield≥{min_yield:.0%}: train t {row['train_t']:.2f}", flush=True)
    grid = pd.DataFrame(rows)
    out = data.DATA.parent / "results"
    grid.to_csv(out / "div_runup_strict.csv", float_format="%.4f", index=False)
    pick = grid.loc[grid["train_t"].idxmax()]
    print("\n== строгий тест (вход после рекомендации)")
    print(grid.round(3).to_string(index=False))
    print(f"\nвыбор по обучению (макс. t на train): окно {pick['window']:.0f}, доходность ≥ {pick['min_yield']:.0%}")
    print(pick.round(4).to_string())
    t = trades(int(pick["window"]), after_announcement=True, min_yield=pick["min_yield"])
    eq, _, _ = portfolio(t, slots=3)
    eq.to_csv(out / "div_runup_strict_equity.csv", float_format="%.2f")
    t.to_csv(out / "div_runup_strict_trades.csv", float_format="%.4f", index=False)
    print(f"10 тыс. ₽ → {eq.iloc[-1]:,.0f} ₽; отменённые в сделках: {t[t['cancelled']][['secid', 'entry', 'ret']].round(3).values.tolist()}")


if __name__ == "__main__":
    import sys
    strict() if sys.argv[1:] == ["strict"] else main()
