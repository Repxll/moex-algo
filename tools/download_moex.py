"""Download free MOEX ISS history for the strategy backtests into ./data.

Usage: python3 tools/download_moex.py [indices fx shares dividends composition futures hourly keyrate]
With no arguments every dataset is downloaded. Re-running skips files that already exist.
"""
import csv
import datetime as dt
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ISS = "https://iss.moex.com/iss"
DATA = Path(__file__).resolve().parent.parent / "data"
START = "2013-01-01"
TODAY = dt.date.today().isoformat()

INDICES = [
    "IMOEX", "MCFTR", "RTSI", "MCXSM", "MOEXBC", "RVI", "RGBI", "RGBITR", "RUCBITR", "RUSFAR",
    "MOEXOG", "MOEXFN", "MOEXMM", "MOEXEU", "MOEXTL", "MOEXCN", "MOEXCH", "MOEXTN", "MOEXRE", "MOEXIT",
]
FX = ["USD000UTSTOM", "CNYRUB_TOM", "EUR_RUB__TOM", "GLDRUB_TOM", "SLVRUB_TOM"]
FUTURES_ASSETS = ["Si", "CNY", "Eu", "RTS", "MIX", "MXI", "BR", "GOLD", "SILV", "NG", "SBRF", "GAZR"]
PERPETUALS = ["USDRUBF", "CNYRUBF", "EURRUBF", "IMOEXF", "GLDRUBF", "SBERF", "GAZPF"]
HOURLY = [
    "SBER", "SBERP", "GAZP", "LKOH", "ROSN", "NVTK", "GMKN", "TATN", "TATNP", "SNGS", "SNGSP", "PLZL",
    "MGNT", "VTBR", "CHMF", "NLMK", "ALRS", "MOEX", "AFLT", "MTSS", "YDEX", "T",
]


def get_json(path, params=None, retries=4):
    query = urllib.parse.urlencode({"iss.meta": "off", **(params or {})})
    url = f"{ISS}/{path}?{query}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_all(path, block, params=None):
    """Page through an ISS table by `start` until it runs out; returns (columns, rows)."""
    params = dict(params or {})
    start, columns, rows, previous = 0, [], [], None
    while True:
        resp = get_json(path, {**params, "start": start})
        page = resp.get(block, {})
        data = page.get("data", [])
        columns = page.get("columns", columns)
        if not data or data == previous:  # some tables ignore `start` and return everything every time
            return columns, rows
        rows.extend(data)
        start += len(data)
        previous = data
        cursor = resp.get(f"{block}.cursor", {})
        if cursor.get("data"):
            total = dict(zip(cursor["columns"], cursor["data"][0])).get("TOTAL")
            if total is not None and start >= total:
                return columns, rows


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)


def save_history(path_out, iss_path, params=None, block="history"):
    if path_out.exists():
        return "skip"
    columns, rows = fetch_all(iss_path, block, {"from": START, "till": TODAY, **(params or {})})
    write_csv(path_out, columns, rows)
    return len(rows)


def run_parallel(fn, items, workers=6):
    with ThreadPoolExecutor(workers) as pool:
        for item, result in zip(items, pool.map(fn, items)):
            print(f"  {item}: {result}", flush=True)


def indices():
    run_parallel(lambda s: save_history(DATA / "indices" / f"{s}.csv",
                                        f"history/engines/stock/markets/index/securities/{s}.json"), INDICES)


def fx():
    run_parallel(lambda s: save_history(DATA / "fx" / f"{s}.csv",
                                        f"history/engines/currency/markets/selt/boards/CETS/securities/{s}.json"), FX)


def composition():
    """IMOEX constituents and weights on the last trading day of each quarter."""
    out = DATA / "index_composition" / "IMOEX.csv"
    if out.exists():
        return
    rows, columns = [], []
    for year in range(2013, dt.date.today().year + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            date = dt.date(year, month, day)
            if date > dt.date.today():
                break
            part = []
            for back in range(10):  # quarter end may fall on a weekend or holiday
                columns, part = fetch_all("statistics/engines/stock/markets/index/analytics/IMOEX.json",
                                          "analytics", {"date": (date - dt.timedelta(days=back)).isoformat()})
                if part:
                    break
            rows.extend(part)
            print(f"  {date}: {len(part)}", flush=True)
    write_csv(out, columns, rows)


def universe():
    path = DATA / "index_composition" / "IMOEX.csv"
    with path.open(encoding="utf-8") as f:
        tickers = {row["secids"] for row in csv.DictReader(f)}
    return sorted(tickers | set(HOURLY))


def shares():
    run_parallel(lambda s: save_history(DATA / "shares" / f"{s}.csv",
                                        f"history/engines/stock/markets/shares/boards/TQBR/securities/{s}.json"),
                 universe())


def dividends():
    """Paid dividends from smart-lab.ru (ISS no longer serves them); one page lists all share classes."""
    out = DATA / "dividends" / "dividends.csv"
    if out.exists():
        return
    rows, tickers = [], set(universe())
    for secid in sorted(tickers):
        req = urllib.request.Request(f"https://smart-lab.ru/q/{secid}/dividend/", headers={"User-Agent": "Mozilla/5.0"})
        try:
            page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
        except Exception as exc:
            print(f"  {secid}: {exc}")
            continue
        table = page[page.find("<table"):page.find("</table>")]
        found = 0
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
            cells = [re.sub(r"<[^>]+>|&nbsp;|₽|%", "", c).strip()
                     for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            if len(cells) >= 5 and cells[0] in tickers and re.match(r"\d{2}\.\d{2}\.\d{4}$", cells[2]):
                to_iso = lambda d: dt.datetime.strptime(d, "%d.%m.%Y").date().isoformat() if d else ""
                rows.append([cells[0], to_iso(cells[1]), to_iso(cells[2]), cells[3], cells[4].replace(",", ".")])
                found += 1
        print(f"  {secid}: {found}", flush=True)
        time.sleep(1)
    rows = sorted({tuple(r) for r in rows})  # the common and the preferred page list the same payouts
    write_csv(out, ["secid", "last_day_with_dividend", "record_date", "period", "dividend_rub"], rows)


def futures():
    for asset in FUTURES_ASSETS:
        series_file = DATA / "futures" / asset / "_series.csv"
        if series_file.exists():
            with series_file.open(encoding="utf-8") as f:
                series = list(csv.DictReader(f))
        else:
            columns, rows = fetch_all("statistics/engines/futures/markets/forts/series.json", "series",
                                      {"asset_code": asset, "show_expired": 1})
            write_csv(series_file, columns, rows)
            series = [dict(zip(columns, r)) for r in rows]
        series = [s for s in series if (s["expiration_date"] or "9999") >= START]
        print(f"{asset}: {len(series)} contracts", flush=True)
        run_parallel(lambda s: save_history(
            DATA / "futures" / asset / f"{s['secid']}.csv",
            f"history/engines/futures/markets/forts/securities/{s['secid']}.json",
            {"from": max(START, s["start_date"] or START)}), series)
    run_parallel(lambda s: save_history(DATA / "futures" / "perpetual" / f"{s}.csv",
                                        f"history/engines/futures/markets/forts/securities/{s}.json"), PERPETUALS)


def hourly():
    """Hourly candles for the most liquid shares (the candles endpoint has no cursor block)."""
    def one(secid):
        out = DATA / "hourly" / f"{secid}.csv"
        if out.exists():
            return "skip"
        columns, rows = fetch_all(f"engines/stock/markets/shares/boards/TQBR/securities/{secid}/candles.json",
                                  "candles", {"interval": 60, "from": START, "till": TODAY})
        write_csv(out, columns, rows)
        return len(rows)
    run_parallel(one, HOURLY, workers=4)


def keyrate():
    """CBR key rate by day, parsed from the cbr.ru table page."""
    out = DATA / "calendar" / "cbr_key_rate.csv"
    if out.exists():
        return
    url = ("https://www.cbr.ru/hd_base/KeyRate/?UniDbQuery.Posted=True&UniDbQuery.From=13.09.2013"
           f"&UniDbQuery.To={dt.date.today():%d.%m.%Y}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    pairs = re.findall(r"<td>(\d{2}\.\d{2}\.\d{4})</td>\s*<td>([\d,]+)</td>", html)
    rows = sorted((dt.datetime.strptime(d, "%d.%m.%Y").date().isoformat(), float(r.replace(",", ".")))
                  for d, r in pairs)
    write_csv(out, ["date", "key_rate"], rows)
    changes = [(d, r) for i, (d, r) in enumerate(rows) if i == 0 or r != rows[i - 1][1]]
    write_csv(DATA / "calendar" / "cbr_key_rate_changes.csv", ["effective_date", "key_rate"], changes)
    print(f"  key rate: {len(rows)} days, {len(changes)} changes")


def dividend_announcements():
    """Board recommendation dates from dohod.ru (date the dividend became known), per share class."""
    out = DATA / "dividends" / "announcements.csv"
    if out.exists():
        return
    to_iso = lambda d: dt.datetime.strptime(d, "%d.%m.%Y").date().isoformat()
    rows = []
    for secid in universe():
        req = urllib.request.Request(f"https://www.dohod.ru/ik/analytics/dividend/{secid.lower()}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        try:
            page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
        except Exception as exc:
            print(f"  {secid}: {exc}")
            continue
        found = 0
        for table in re.findall(r"<table.*?</table>", page, re.S):
            if "Дата объявления" not in table:
                continue
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
                cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
                if len(cells) >= 4 and re.match(r"\d{2}\.\d{2}\.\d{4}$", cells[0]) and re.match(r"\d{2}\.\d{2}\.\d{4}$", cells[1]):
                    rows.append([secid, to_iso(cells[0]), to_iso(cells[1]), cells[2], cells[3]])
                    found += 1
        print(f"  {secid}: {found}", flush=True)
        time.sleep(1)
    write_csv(out, ["secid", "announced", "record_date", "year", "dividend_rub"], rows)


OPTION_ASSETS = ["RTS", "Si"]
OPTIONS_START = "2015-01-01"


def options():
    """Daily settle prices of every option on each asset, one file per trading day (resumable)."""
    with (DATA / "indices" / "IMOEX.csv").open(encoding="utf-8") as f:
        days = [r["TRADEDATE"] for r in csv.DictReader(f) if r["TRADEDATE"] >= OPTIONS_START]
    for asset in OPTION_ASSETS:
        def one(day):
            out = DATA / "options" / asset / f"{day}.csv"
            if out.exists():
                return None
            try:
                columns, rows = fetch_all("history/engines/futures/markets/options/securities.json", "history",
                                          {"date": day, "assetcode": asset})
            except Exception:
                return -1
            write_csv(out, columns, rows)
            return len(rows)
        with ThreadPoolExecutor(6) as pool:
            done = [n for n in pool.map(one, days) if n is not None]
        failed = done.count(-1)
        print(f"  {asset}: {len(done) - failed} new days, {failed} failed (re-run to retry)", flush=True)


MONTHS_RU = {m: i for i, m in enumerate(
    ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября",
     "декабря"], 1)}


def cbr_meetings():
    """Dates of CBR key-rate board meetings (incl. holds) from the cbr.ru meeting calendar."""
    out = DATA / "calendar" / "cbr_meetings.csv"
    if out.exists():
        return
    req = urllib.request.Request("https://www.cbr.ru/dkp/cal_mp/", headers={"User-Agent": "Mozilla/5.0"})
    text = re.sub(r"<[^>]+>|&nbsp;", " ", urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))
    found = re.findall(r"(\d{1,2}) (\w+) (\d{4}) года\s+Заседание Совета директоров Банка России\s+по\s+ключевой ставке",
                       re.sub(r"\s+", " ", text))
    dates = sorted({dt.date(int(y), MONTHS_RU[m], int(d)).isoformat() for d, m, y in found if m in MONTHS_RU})
    write_csv(out, ["meeting_date"], [[d] for d in dates])
    print(f"  CBR meetings: {len(dates)} ({dates[0]} → {dates[-1]})")


STEPS = {"indices": indices, "fx": fx, "composition": composition, "shares": shares, "dividends": dividends,
         "futures": futures, "hourly": hourly, "keyrate": keyrate, "cbr_meetings": cbr_meetings,
         "options": options, "dividend_announcements": dividend_announcements}

if __name__ == "__main__":
    for name in sys.argv[1:] or STEPS:
        print(f"== {name}", flush=True)
        STEPS[name]()
