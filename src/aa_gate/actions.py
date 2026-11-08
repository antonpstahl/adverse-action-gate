"""Customer actions as consistent changes of every feature they touch (plan §3.6). Each rule is an assumption, named
here and in the thesis."""

import numpy as np
import pandas as pd

from aa_gate.settings import BILLS, PAYMENTS, STATUS

ON_TIME = 0  # repayment status "paid on time": the data's most common value without a delay


def on_time(x: pd.Series, months) -> pd.Series | None:
    """Pay on time for the next `months` months (1-6). Every monthly series moves back by that many months; the new
    months carry status ON_TIME, the newest balance and the six-month average payment. Paying on time means paying:
    the newest payment, often 0 NT$ in a late month, would make a customer on time who pays nothing (decision 43)."""
    if months != int(months) or not 1 <= months <= 6:
        return None
    k, y = int(months), x.copy()
    usual = round(float(x[PAYMENTS].mean()))
    for cols, fill in ((STATUS, ON_TIME), (BILLS, x[BILLS[0]]), (PAYMENTS, usual)):
        y[cols] = np.concatenate([[fill] * k, x[cols].to_numpy()[: len(cols) - k]])
    return y


def pay_more(x: pd.Series, amount) -> pd.Series | None:
    """Pay `amount` more this month: the newest payment rises by it and the newest balance falls by it. A balance
    below 0 is a credit balance, which the data contain (decision 23)."""
    if amount <= 0:
        return None
    y = x.copy()
    y[PAYMENTS[0]] = x[PAYMENTS[0]] + amount
    y[BILLS[0]] = x[BILLS[0]] - amount
    return y


def both(x: pd.Series, amount, months) -> pd.Series | None:
    """Pay on time for the next `months` months and `amount` more in the last of them; 0 more is the months on time
    alone, which least_action reports as 0 (decision 43)."""
    y = on_time(x, months)
    return y if y is None or amount == 0 else pay_more(y, amount)


ACTIONS = {"on_time": on_time, "pay_more": pay_more, "both": both}


def apply(x: pd.Series, action: str, amount, lo: pd.Series, hi: pd.Series, months=None) -> pd.Series | None:
    """The changed applicant, or None if the action is unknown or invalid, or moves a feature outside the range
    observed in the training data. Only both uses `months`."""
    if action not in ACTIONS or amount is None or (action == "both" and months is None):
        return None
    y = both(x, amount, months) if action == "both" else ACTIONS[action](x, amount)
    if y is None:
        return None
    changed = y != x
    return None if ((y < lo) | (y > hi))[changed].any() else y
