"""Strategy #60 (dividend run-up, strict version) evaluated on today's data.

Rules (validated in results/DIV_RUNUP_CHECK.md):
- share is in IMOEX today, the board has already recommended the dividend, dividend / price >= 7%;
- buy no earlier than the recommendation and no earlier than 20 trading days before the last day with dividend;
- sell at the open of the last day with dividend (the dividend itself is not received); at most 3 positions.
"""
import datetime as dt
import json
import re
import urllib.request

MIN_YIELD = 0.07
WINDOW = 20
MIN_DAYS = 3
SLOTS = 3
ISS = "https://iss.moex.com/iss"


def _get(url, as_json=True):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    body = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
    return json.loads(body) if as_json else body


def index_members():
    d = _get(f"{ISS}/statistics/engines/stock/markets/index/analytics/IMOEX.json?iss.meta=off&limit=100")
    a = d["analytics"]
    return sorted({dict(zip(a["columns"], r))["secids"] for r in a["data"]})


def prices(secids):
    d = _get(f"{ISS}/engines/stock/markets/shares/boards/TQBR/securities.json?iss.meta=off"
             f"&iss.only=securities,marketdata&securities={','.join(secids)}")
    sec = {r[0]: dict(zip(d["securities"]["columns"], r)) for r in d["securities"]["data"]}
    md = {r[d["marketdata"]["columns"].index("SECID")]: dict(zip(d["marketdata"]["columns"], r)) for r in d["marketdata"]["data"]}
    out = {}
    for s in secids:
        last = (md.get(s) or {}).get("LAST") or (sec.get(s) or {}).get("PREVPRICE")
        if last:
            out[s] = {"price": float(last), "lot": int(sec[s]["LOTSIZE"])}
    return out


def upcoming_dividends(secid, today):
    """Recommended (not forecast) dividends with a record date ahead, from dohod.ru."""
    page = _get(f"https://www.dohod.ru/ik/analytics/dividend/{secid.lower()}", as_json=False)
    out = []
    for table in re.findall(r"<table.*?</table>", page, re.S):
        if "Дата объявления" not in table:
            continue
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            if len(cells) < 4 or "прогноз" in cells[1] or not re.match(r"\d{2}\.\d{2}\.\d{4}", cells[1]):
                continue
            # a recommended-but-unpaid dividend is listed with announcement "n/a" until it is paid
            announced = dt.datetime.strptime(cells[0], "%d.%m.%Y").date() if re.match(r"\d{2}\.\d{2}\.\d{4}$", cells[0]) else today
            record = dt.datetime.strptime(cells[1][:10], "%d.%m.%Y").date()
            if record > today and announced <= today:
                out.append({"announced": announced, "record": record, "dividend": float(cells[3].replace(",", "."))})
    return out


def prev_weekday(d, n=1):
    while n:
        d -= dt.timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def weekdays_between(a, b):
    """Trading days from a (exclusive) to b (inclusive), holidays ignored."""
    n, d = 0, a
    while d < b:
        d += dt.timedelta(days=1)
        n += d.weekday() < 5
    return n


def scan(today=None, held=()):
    """Return (buy_candidates, sell_signals). `held` = tickers currently held with their last_day."""
    today = today or dt.date.today()
    members = index_members()
    px = prices(members)
    buys = []
    for s in members:
        try:
            divs = upcoming_dividends(s, today)
        except Exception:
            continue
        for d in divs:
            last_day = prev_weekday(d["record"])  # T+1 settlement: last day with dividend is one trading day before the record date
            left = weekdays_between(today, last_day)
            if not (MIN_DAYS <= left <= WINDOW) or s not in px:
                continue
            dy = d["dividend"] / px[s]["price"]
            if dy >= MIN_YIELD:
                buys.append({"secid": s, "price": px[s]["price"], "lot": px[s]["lot"], "dividend": d["dividend"],
                             "yield": dy, "announced": d["announced"].isoformat(), "last_day": last_day.isoformat(),
                             "days_left": left})
    buys.sort(key=lambda b: -b["yield"])
    sells = [h for h in held if dt.date.fromisoformat(h["last_day"]) <= today]
    return buys, sells
