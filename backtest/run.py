"""Run stage-1 strategies on train and validation; open the locked test period only for finalists.

python -m backtest.run            # train + validation table, then finalists on test
python -m backtest.run 31i 4      # selected strategies
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

from .engine import COST_MULT, metrics
from .strategies import STRATEGIES, run

PERIODS = {"train": ("2013-03-25", "2019-12-31"), "valid": ("2020-01-01", "2022-12-31"), "test": ("2023-01-01", None)}
FINALIST_SHARPE = 0.3
ROOT = Path(__file__).resolve().parent.parent


def names():
    cat = yaml.safe_load((ROOT / "strategies.yaml").read_text(encoding="utf-8"))["strategies"]
    return {str(s["id"]): s["name"] for s in cat}


def label(sid, catalog):
    base = sid.rstrip("is")
    suffix = {"i": " [фонд IMOEX]", "s": " [топ-10 акций]"}.get(sid[-1], "") if sid != base else ""
    return catalog.get(base, sid) + suffix


def main(ids):
    catalog = names()
    rows, results = [], {}
    for sid in ids:
        try:
            res = run(sid)
        except Exception as exc:
            print(f"{sid}: ошибка {exc!r}", flush=True)
            continue
        results[sid] = res
        row = {"id": sid, "name": label(sid, catalog)}
        for p in ("train", "valid"):
            for k, v in metrics(res, *PERIODS[p]).items():
                row[f"{p}_{k}"] = v
        rows.append(row)
        print(f"{sid:>4} {row['name'][:45]:45} train SR {row.get('train_sharpe', 0):5.2f}  valid SR {row.get('valid_sharpe', 0):5.2f}", flush=True)
    table = pd.DataFrame(rows).set_index("id")
    finalists = table[(table["train_sharpe"] > FINALIST_SHARPE) & (table["valid_sharpe"] > FINALIST_SHARPE)].index
    for sid in finalists:
        for k, v in metrics(results[sid], *PERIODS["test"]).items():
            table.loc[sid, f"test_{k}"] = v
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tag = "" if COST_MULT == 1 else f"_costx{COST_MULT:g}"
    table.to_csv(out / f"stage1{tag}.csv", float_format="%.4f")
    pd.DataFrame({sid: r.returns for sid, r in results.items()}).to_csv(out / f"stage1{tag}_daily_returns.csv", float_format="%.6f")
    print(f"\nфиналисты ({len(finalists)}): {list(finalists)}")
    return table


if __name__ == "__main__":
    main(sys.argv[1:] or list(STRATEGIES))
