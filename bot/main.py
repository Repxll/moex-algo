"""Telegram bot for strategy #60: every weekday it scans, proposes trades with ✅/❌, executes approved ones in the
T-Invest sandbox and logs everything. Run: python3 -m bot.main   (settings in bot/.env, see bot/README.md)
"""
import csv
import datetime as dt
import json
import time
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

from . import signals
from .tinvest import Sandbox

HERE = Path(__file__).resolve().parent
STATE, JOURNAL = HERE / "state.json", HERE / "journal.csv"
MSK = ZoneInfo("Europe/Moscow")
SCAN_AT = dt.time(10, 5)


OLD_ENV = Path.home() / "moex_signals" / ".env"  # the existing signals bot: same Telegram bot and sandbox account
CAPITAL = 20_000  # the sandbox account may hold more; #60 sizes positions from this budget only


def load_env():
    """Environment variables (GitHub Actions secrets), else bot/.env, else the old project's .env."""
    import os
    if os.environ.get("TELEGRAM_TOKEN"):
        return dict(os.environ)
    path = HERE / ".env" if (HERE / ".env").exists() else OLD_ENV
    env = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    env.setdefault("TINVEST_TOKEN", env.get("T_INVEST_TOKEN", ""))
    return env


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"positions": [], "pending": {}, "last_scan": None, "offset": 0}


def save_state(st):
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))


def journal(event, **kw):
    new = not JOURNAL.exists()
    with JOURNAL.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "event", "details"])
        w.writerow([dt.datetime.now(MSK).isoformat(timespec="seconds"), event, json.dumps(kw, ensure_ascii=False)])


class Telegram:
    def __init__(self, token, chat_id):
        self.base, self.chat_id = f"https://api.telegram.org/bot{token}/", str(chat_id)

    def call(self, method, **params):
        req = urllib.request.Request(self.base + method, data=json.dumps(params).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=70) as resp:
            return json.load(resp)["result"]

    def send(self, text, buttons=None):
        kw = {"chat_id": self.chat_id, "text": text}
        if buttons:
            kw["reply_markup"] = {"inline_keyboard": [buttons]}
        return self.call("sendMessage", **kw)


def describe_buy(b, lots, cost):
    return (f"🟢 Сигнал на ПОКУПКУ {b['secid']}\n"
            f"Дивиденд {b['dividend']:g} ₽ = {b['yield']:.1%} от цены {b['price']:g} ₽\n"
            f"Последний день с дивидендом {b['last_day']} (через {b['days_left']} торг. дн.) — продажа в этот день на открытии\n"
            f"Объём: {lots} лот(ов) ≈ {cost:,.0f} ₽")


def scan(tg, sb, account, st):
    st["last_scan"] = dt.date.today().isoformat()
    buys, sells = signals.scan(held=st["positions"])
    for p in sells:
        sid = f"sell-{p['secid']}-{p['last_day']}"
        if sid not in st["pending"]:
            st["pending"][sid] = {"side": "sell", **p}
            tg.send(f"🔴 Пора ПРОДАТЬ {p['secid']} ({p['lots']} лот.): сегодня последний день с дивидендом.",
                    [{"text": "✅ Продать", "callback_data": f"ok:{sid}"}, {"text": "❌ Держать", "callback_data": f"no:{sid}"}])
    held = {p["secid"] for p in st["positions"]}
    free = signals.SLOTS - len(st["positions"])
    equity = min(sb.portfolio(account)[0], CAPITAL)
    for b in buys:
        if free <= 0 or b["secid"] in held:
            continue
        sid = f"buy-{b['secid']}-{b['last_day']}"
        if sid in st["pending"] or any(p.get("id") == sid for p in st["positions"]):
            continue
        lots = int(equity / signals.SLOTS // (b["price"] * b["lot"]))
        if lots < 1:
            tg.send(f"ℹ️ {b['secid']}: сигнал есть (дивиденд {b['yield']:.1%}), но на 1 лот ({b['price'] * b['lot']:,.0f} ₽) не хватает трети счёта.")
            journal("skip_lot", **b)
            continue
        st["pending"][sid] = {"side": "buy", "lots": lots, **b}
        tg.send(describe_buy(b, lots, lots * b["lot"] * b["price"]),
                [{"text": "✅ Купить", "callback_data": f"ok:{sid}"}, {"text": "❌ Пропустить", "callback_data": f"no:{sid}"}])
        journal("signal", id=sid, **b)
        free -= 1
    if not buys and not sells:
        tg.send("Сегодня сигналов по №60 нет.")
    save_state(st)


def on_callback(tg, sb, account, st, cq):
    if str(cq["message"]["chat"]["id"]) != tg.chat_id:
        return
    action, sid = cq["data"].split(":", 1)
    sig = st["pending"].pop(sid, None)
    try:
        tg.call("answerCallbackQuery", callback_query_id=cq["id"])
    except Exception:
        pass  # Telegram rejects answers to presses older than a few minutes; the scheduled run is usually later
    if sig is None:
        tg.send("Сигнал уже обработан.")
        return
    if action == "no":
        journal("rejected", id=sid)
        tg.send(f"Пропущено: {sid}")
    else:
        ins = sb.share(sig["secid"])
        res = sb.market_order(account, ins["uid"], sig["lots"], buy=sig["side"] == "buy")
        journal("executed", id=sid, side=sig["side"], lots=sig["lots"], status=res.get("executionReportStatus"),
                price=res.get("executedOrderPrice"))
        if sig["side"] == "buy":
            st["positions"].append({"id": sid, "secid": sig["secid"], "lots": sig["lots"], "last_day": sig["last_day"]})
        else:
            st["positions"] = [p for p in st["positions"] if p["secid"] != sig["secid"]]
        tg.send(f"✅ Песочница: {sig['side']} {sig['secid']} {sig['lots']} лот. — {res.get('executionReportStatus')}")
    save_state(st)


def step(tg, sb, account, st, poll_timeout):
    """Daily scan when due, then handle button presses and commands received since the last step."""
    now = dt.datetime.now(MSK)
    if now.weekday() < 5 and now.time() >= SCAN_AT and st["last_scan"] != now.date().isoformat():
        scan(tg, sb, account, st)
    for u in tg.call("getUpdates", offset=st["offset"], timeout=poll_timeout):
        st["offset"] = u["update_id"] + 1
        if "callback_query" in u:
            on_callback(tg, sb, account, st, u["callback_query"])
        msg = u.get("message", {})
        if str(msg.get("chat", {}).get("id")) != tg.chat_id:
            continue
        if msg.get("text") == "/scan":
            scan(tg, sb, account, st)
        elif msg.get("text") == "/status":
            eq, pos = sb.portfolio(account)
            tg.send(f"Счёт (песочница): {eq:,.0f} ₽\nПозиции: {st['positions'] or 'нет'}")
    save_state(st)


def main(once=False, loop_minutes=None, persist=None):
    env = load_env()
    tg, sb = Telegram(env["TELEGRAM_TOKEN"], env["TELEGRAM_CHAT_ID"]), Sandbox(env["TINVEST_TOKEN"])
    st = load_state()
    fresh = "account" not in st
    account = st["account"] = sb.ensure_account(preferred=st.get("account"))
    save_state(st)
    if fresh:
        tg.send("Бот №60 запущен (песочница). Команды: /scan — проверить сейчас, /status — счёт.")
    if once:
        step(tg, sb, account, st, poll_timeout=0)
        return
    deadline = time.time() + loop_minutes * 60 if loop_minutes else None
    while deadline is None or time.time() < deadline:
        before = STATE.read_text() if STATE.exists() else ""
        step(tg, sb, account, st, poll_timeout=25)
        if persist and STATE.read_text() != before:
            persist()
        time.sleep(1)


def git_persist():
    """Commit state and journal right away, so a cancelled run cannot replay an executed order."""
    import subprocess
    run = lambda *a: subprocess.run(["git", *a], cwd=HERE.parent, capture_output=True)
    run("add", "bot/state.json", "bot/journal.csv")
    if run("diff", "--cached", "--quiet").returncode:
        run("commit", "-m", "bot state")
        run("pull", "--rebase", "-q")
        run("push", "-q")


if __name__ == "__main__":
    import os
    import sys
    loop = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--loop-minutes=")), None)
    main(once="--once" in sys.argv, loop_minutes=loop, persist=git_persist if os.environ.get("GITHUB_ACTIONS") else None)
