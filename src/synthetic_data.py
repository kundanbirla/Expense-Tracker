"""Synthetic (fake) transaction generator for demos, tests and anomaly evaluation.

No real personal financial data is ever used or needed.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

# category: [(description, low, high, need_want, weight)]
SPEC: dict[str, list[tuple[str, int, int, str, int]]] = {
    "Food": [
        ("Lunch with friends", 180, 450, "Want", 4), ("Weekly groceries", 400, 1200, "Need", 3),
        ("Late night food delivery", 200, 600, "Want", 4), ("Chai and samosa", 20, 80, "Both", 5),
        ("Canteen lunch", 60, 140, "Need", 6), ("Swiggy dinner order", 220, 520, "Want", 3),
        ("Coffee with classmate", 120, 300, "Want", 2), ("Milk and vegetables", 80, 250, "Need", 4),
    ],
    "Travel": [
        ("Bus fare", 20, 90, "Need", 6), ("Metro recharge", 200, 500, "Need", 2),
        ("Auto to college", 40, 150, "Need", 4), ("Uber ride", 150, 450, "Both", 2),
        ("Weekend trip transport", 600, 1800, "Want", 1),
    ],
    "Shopping": [
        ("Amazon order", 300, 2500, "Both", 2), ("Clothes from Myntra", 700, 2500, "Want", 1),
        ("Phone cover", 150, 500, "Want", 1), ("Birthday gift for friend", 400, 1500, "Both", 1),
    ],
    "Education": [
        ("Stationery for class", 60, 300, "Need", 2), ("Photocopy of notes", 20, 120, "Need", 2),
        ("Online course", 400, 1500, "Both", 1), ("Reference book", 300, 900, "Need", 1),
    ],
    "Entertainment": [
        ("Movie tickets", 250, 700, "Want", 2), ("Netflix subscription", 199, 649, "Want", 1),
        ("Gaming top up", 80, 800, "Want", 2), ("Bowling with friends", 300, 700, "Want", 1),
    ],
    "Health": [
        ("Medicines from pharmacy", 80, 600, "Need", 2), ("Doctor consultation", 300, 800, "Need", 1),
        ("Protein powder", 900, 2200, "Both", 1),
    ],
    "Household": [
        ("Cleaning supplies", 100, 500, "Need", 2), ("Kitchen utensils", 150, 700, "Need", 1),
        ("Room decor items", 200, 900, "Want", 1),
    ],
    "Other": [("Haircut", 100, 400, "Both", 2), ("Laundry", 80, 250, "Need", 2), ("Donation", 50, 500, "Both", 1)],
}
_CATEGORY_WEIGHT = {"Food": 10, "Travel": 6, "Shopping": 2, "Education": 2, "Entertainment": 2,
                    "Health": 1, "Household": 1.5, "Other": 1.5}
_PAYMENTS = ["UPI"] * 12 + ["Cash"] * 4 + ["Card"] * 3 + ["Other"]


def _round(x: float) -> int:
    return max(5, int(round(x / 5)) * 5)


def generate_transactions(end: date | None = None, days: int = 75, seed: int = 42,
                          anomaly_rate: float = 0.02, avg_per_day: float = 1.6) -> list[dict]:
    """Return a list of transaction dicts ending on ``end`` (default: 2026-10-11).

    Rows carry an extra ``is_injected_anomaly`` flag (ignored by the DB layer) so detectors can be
    scored. Injected anomalies are 4-8x the normal amount for that item.
    """
    rng = random.Random(seed)
    end = end or date(2026, 10, 11)
    start = end - timedelta(days=days - 1)
    cats = list(_CATEGORY_WEIGHT)
    weights = [_CATEGORY_WEIGHT[c] for c in cats]
    rows: list[dict] = []

    def add(d: date, ttype: str, category: str, amount: float, desc: str, nw=None, anomaly=False):
        waste = context = None
        if ttype == "expense":
            if nw == "Want":
                waste = rng.choices(["Yes", "Not sure", "No"], [4, 3, 3])[0]
                context = rng.choice(["Impulse", "Planned", "Other"])
            elif nw == "Both":
                waste = rng.choices(["Yes", "Not sure", "No"], [1, 2, 7])[0]
                context = rng.choice(["Planned", "Other"])
            else:
                waste = rng.choices(["No", "Not sure"], [19, 1])[0]
                context = rng.choice(["Necessary/urgent", "Planned"])
        rows.append({
            "transaction_date": d.isoformat(), "amount": amount, "transaction_type": ttype,
            "category": category, "description": desc, "wasteful_label": waste, "need_want": nw if ttype == "expense" else None,
            "payment_method": rng.choice(_PAYMENTS), "context": context,
            "is_injected_anomaly": anomaly,
        })

    d = start
    while d <= end:
        # fixed monthly items
        if d.day == 1:
            add(d, "income", "Salary", 45000, "Monthly salary")
        if d.day == 5:
            add(d, "expense", "Bills", 8000, "Room rent", "Need")
        if d.day == 12:
            add(d, "expense", "Bills", _round(rng.uniform(600, 1100)), "Electricity bill", "Need")
        if d.day == 15:
            add(d, "expense", "Bills", 599, "WiFi bill", "Need")
        if d.day == 20:
            add(d, "expense", "Bills", 299, "Mobile recharge", "Need")
        if d.day == 18 and rng.random() < 0.7:
            add(d, "income", "Freelance", _round(rng.uniform(4000, 12000)), "Freelance project payment")
        # daily spending
        n = max(0, round(rng.gauss(avg_per_day, 1.0)))
        for _ in range(n):
            cat = rng.choices(cats, weights)[0]
            desc, lo, hi, nw, _w = rng.choices(SPEC[cat], [x[4] for x in SPEC[cat]])[0]
            amt = _round(rng.uniform(lo, hi))
            anomaly = rng.random() < anomaly_rate
            if anomaly:
                amt = _round(hi * rng.uniform(4, 8))
            add(d, "expense", cat, amt, desc, nw, anomaly)
        d += timedelta(days=1)

    # a couple of refunds
    shopping = [r for r in rows if r["category"] == "Shopping" and r["transaction_type"] == "expense"]
    for r in rng.sample(shopping, min(2, len(shopping))):
        rd = min(end, date.fromisoformat(r["transaction_date"]) + timedelta(days=rng.randint(2, 6)))
        rows.append({"transaction_date": rd.isoformat(), "amount": round(r["amount"] * 0.5),
                     "transaction_type": "refund", "category": "Shopping",
                     "description": "Partial refund: " + r["description"], "wasteful_label": None,
                     "need_want": None, "payment_method": "UPI", "context": None, "is_injected_anomaly": False})
    rows.sort(key=lambda r: r["transaction_date"])
    return rows
