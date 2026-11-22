"""Minimal T-Invest API client over the REST gateway. Sandbox only: real orders are deliberately not wired."""
import json
import ssl
import urllib.request
import uuid
from pathlib import Path

SANDBOX = "https://sandbox-invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1."
# the sandbox certificate chains to the Russian Trusted Root CA, absent from macOS and certifi
CA_BUNDLE = Path(__file__).resolve().parent / "ca_bundle.pem"
SSL_CTX = ssl.create_default_context(cafile=str(CA_BUNDLE)) if CA_BUNDLE.exists() else None


def money(units, currency="rub"):
    return {"currency": currency, "units": str(int(units)), "nano": 0}


def to_float(q):
    return int(q.get("units", 0)) + q.get("nano", 0) / 1e9 if q else 0.0


class Sandbox:
    def __init__(self, token):
        self.token = token

    def call(self, method, body=None):
        req = urllib.request.Request(SANDBOX + method, data=json.dumps(body or {}).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as resp:
            return json.load(resp)

    def ensure_account(self, deposit=10_000, preferred=None):
        """Reuse `preferred` if it is a sandbox account, otherwise open a new one funded with `deposit`."""
        accounts = [a["id"] for a in self.call("SandboxService/GetSandboxAccounts").get("accounts", [])]
        if preferred in accounts:
            return preferred
        acc = self.call("SandboxService/OpenSandboxAccount", {"name": "strategy-60"})["accountId"]
        self.call("SandboxService/SandboxPayIn", {"accountId": acc, "amount": money(deposit)})
        return acc

    def share(self, ticker):
        ins = self.call("InstrumentsService/ShareBy", {"idType": "INSTRUMENT_ID_TYPE_TICKER", "classCode": "TQBR", "id": ticker})["instrument"]
        return {"uid": ins["uid"], "lot": int(ins["lot"])}

    def portfolio(self, account):
        p = self.call("SandboxService/GetSandboxPortfolio", {"accountId": account})
        return to_float(p.get("totalAmountPortfolio")), p.get("positions", [])

    def market_order(self, account, uid, lots, buy):
        return self.call("SandboxService/PostSandboxOrder", {
            "accountId": account, "instrumentId": uid, "quantity": str(lots), "orderId": str(uuid.uuid4()),
            "direction": "ORDER_DIRECTION_BUY" if buy else "ORDER_DIRECTION_SELL", "orderType": "ORDER_TYPE_MARKET"})
