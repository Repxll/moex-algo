"""Portfolio simulation: target weights decided on the close of day t are traded at the open of day t+1."""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data

COST_MULT = float(os.environ.get("BT_COST_MULT", 1))  # 0 = gross run, to see the edge before costs
COMMISSION = 0.003 * COST_MULT  # T-Invest «Инвестор» tariff, per side
FUT_COMMISSION = 0.0005 * COST_MULT  # assumed broker + exchange fee on futures notional, per side


@dataclass
class Asset:
    ohlc: pd.DataFrame
    slippage: float = 0.001
    annual_fee: float = 0.0  # fund management fee, charged on held weight
    futures: bool = False  # margin only: the capital keeps earning the money rate; weight may be negative
    roll: pd.Series = None  # True on days the position is rolled into the next contract (pays a round trip)


@dataclass
class Result:
    returns: pd.Series
    exposure: pd.Series
    turnover: pd.Series
    costs: pd.Series
    notes: list = field(default_factory=list)


def simulate(weights: pd.DataFrame, assets: dict) -> Result:
    cal = data.calendar()
    w_target = weights.reindex(cal).ffill().fillna(0.0)
    cols = list(w_target.columns)
    O = pd.DataFrame({c: assets[c].ohlc["O"] for c in cols}).reindex(cal)
    C = pd.DataFrame({c: assets[c].ohlc["C"] for c in cols}).reindex(cal)
    listed = C.notna()
    C, O = C.ffill(), O.fillna(C.ffill())
    w_target = w_target.where(listed, 0.0)

    w_new = w_target.shift(1).fillna(0.0)  # executed at today's open
    w_new = w_new.where(listed, 0.0)
    w_old = w_new.shift(1).fillna(0.0)     # held overnight from yesterday's close
    r_gap = (O / C.shift(1) - 1).fillna(0.0)
    r_day = (C / O - 1).fillna(0.0)

    days = pd.Series(cal, index=cal).diff().dt.days.fillna(1).clip(upper=10)
    cash_r = data.money_rate() / 100 * days / 365
    fee = pd.Series({c: assets[c].annual_fee for c in cols})
    slip = pd.Series({c: assets[c].slippage for c in cols}) * COST_MULT

    turnover = (w_new - w_old).abs()
    for c in cols:
        if assets[c].roll is not None:
            turnover[c] += 2 * w_new[c].abs() * assets[c].roll.reindex(cal).fillna(False).astype(float)
    is_fut = pd.Series({c: assets[c].futures for c in cols})
    commission = pd.Series({c: FUT_COMMISSION if assets[c].futures else COMMISSION for c in cols})
    costs = (turnover * (commission + slip)).sum(axis=1)
    fees = (w_new * fee).sum(axis=1) * days / 365
    gross = (1 + (w_old * r_gap).sum(axis=1)) * (1 + (w_new * r_day).sum(axis=1)) - 1
    cash = (1 - w_new.loc[:, ~is_fut.values].sum(axis=1)).clip(lower=0) * cash_r
    ret = gross + cash - costs - fees
    return Result(ret, w_new.abs().sum(axis=1), turnover.sum(axis=1), costs)


def money_returns():
    cal = data.calendar()
    days = pd.Series(cal, index=cal).diff().dt.days.fillna(1).clip(upper=10)
    return data.money_rate() / 100 * days / 365


def metrics(res: Result, start, end):
    sl = slice(start, end)
    r, m = res.returns[sl], money_returns()[sl]
    if len(r) < 20:
        return {}
    years = (r.index[-1] - r.index[0]).days / 365.25
    eq = (1 + r).cumprod()
    excess = r - m
    sd = excess.std()
    return {
        "cagr": eq.iloc[-1] ** (1 / years) - 1,
        "money_cagr": (1 + m).prod() ** (1 / years) - 1,
        "sharpe": excess.mean() / sd * np.sqrt(252) if sd > 0 else 0.0,
        "max_dd": (eq / eq.cummax() - 1).min(),
        "exposure": res.exposure[sl].mean(),
        "trades_per_year": res.turnover[sl].sum() / 2 / years,
        "cost_per_year": res.costs[sl].sum() / years,
    }
