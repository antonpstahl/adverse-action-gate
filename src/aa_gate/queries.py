"""The query catalogue (plan §3.4): what a statement claims, run on the model. A verdict is True, False, DEPENDS (Reg B's
methods 1 and 2 disagree) or None (not checkable). A query is a dict with the fields of QUERY below; fields a query
does not use are None or missing."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aa_gate import actions
from aa_gate.data import KINDS, kind, split
from aa_gate.model import cutoff, cuts, fit, risk
from aa_gate.reference import groups, reasons, shapley
from aa_gate.settings import BILLS, BLOCKS, FEATURES, PAYMENTS, STATUS

DEPENDS = "reference-dependent"
TYPES = ["outcome", "action", "least", "reason", "compare", "measure", "used"]
# What a statement may say the model uses or does not use: its blocks, and attributes it leaves out
ATTRIBUTES = list(BLOCKS) + ["age", "sex", "marital status", "income", "employment", "education"]


def _null(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


# A query as JSON schema: the reader's output format (gate.READING) and the reading page's labels use these fields
_BLOCK = {"type": "string", "enum": list(BLOCKS)}
QUERY_FIELDS = {
    "type": {"type": "string", "enum": TYPES},
    "quantity": _null({"type": "string", "enum": ["decision", "risk", "cutoff", "gap"]}),
    "decision": _null({"type": "string", "enum": ["approved", "declined"]}),
    "direction": _null({"type": "string", "enum": ["lower", "higher"]}),
    "value": _null({"type": "number"}),
    "reference_value": _null({"type": "number"}),
    "decimals": _null({"type": "integer"}),
    "action": _null({"type": "string", "enum": list(actions.ACTIONS)}),
    "amount": _null({"type": "number"}),
    "months": _null({"type": "integer"}),  # the months on time of the action both
    "block": _null(_BLOCK),
    "other_block": _null(_BLOCK),
    "rank": _null({"type": "integer"}),
    "feature": _null({"type": "string", "enum": FEATURES + ATTRIBUTES}),  # ATTRIBUTES holds the blocks too
    "status": _null({"type": "string", "enum": list(KINDS.values()) + ["late"]}),
    "is_used": _null({"type": "boolean"}),
}
QUERY = {"type": "object", "properties": QUERY_FIELDS, "required": list(QUERY_FIELDS), "additionalProperties": False}


@dataclass
class Bank:
    """One model with its cutoff, Reg B reference groups, the observed feature ranges and the declined test
    applicants."""

    kind: str
    model: object
    cutoff: float
    refs: dict
    lo: pd.Series
    hi: pd.Series
    declined: pd.DataFrame
    memo: dict = field(default_factory=dict)


def build(kind: str) -> Bank:
    X_train, X_test, y_train, _ = split()
    model = fit(kind, X_train, y_train)
    cut = cutoff(model, X_train)
    declined = X_test[risk(model, X_test) > cut]
    return Bank(kind, model, cut, groups(model, X_train, cut), X_train.min(), X_train.max(), declined)


def applicant_risk(bank: Bank, x: pd.Series) -> float:
    return float(risk(bank.model, x.to_frame().T.astype(float))[0])


def shortfalls(bank: Bank, x: pd.Series, method: int) -> pd.Series:
    key = ("phi", method, tuple(x.to_numpy()))
    if key not in bank.memo:
        bank.memo[key] = shapley(bank.model, x, bank.refs[method])
    return bank.memo[key]


def _cuts(bank: Bank, feature: str) -> np.ndarray:
    """The model's cut points on one feature, computed once per bank."""
    if ("cuts", feature) not in bank.memo:
        bank.memo[("cuts", feature)] = cuts(bank.model, feature)
    return bank.memo[("cuts", feature)]


def least(bank: Bank, x: pd.Series, action: str, months: int | None = None) -> dict:
    """The least the applicant would have to do, all else equal, to be approved: the fewest months on time, the
    smallest extra payment in whole NT$, or for both the smallest extra payment after `months` months on time (0 if
    those months are enough alone; decision 43). Exact: the risk can change only where a feature the action moves
    crosses one of the model's cut points, so the model is evaluated at the whole amounts next to every crossing and
    is constant in between. `least` is None if nothing within the range of the data is approved."""
    key = ("least", action, months, tuple(x.to_numpy()))
    if key in bank.memo:
        return bank.memo[key]
    if action == "both":
        y = actions.apply(x, "on_time", months, bank.lo, bank.hi)
        out = ({"least": None, "larger_also_approved": None} if y is None else
               {"least": 0, "larger_also_approved": True} if applicant_risk(bank, y) <= bank.cutoff else
               least(bank, y, "pay_more"))
        bank.memo[key] = {**out, "action": "both", "months": months}
        return bank.memo[key]
    if action == "on_time":
        candidates = np.arange(1, 7, dtype=float)
    else:  # pay_more
        pay, bill = x[PAYMENTS[0]], x[BILLS[0]]
        top = np.floor(min(bank.hi[PAYMENTS[0]] - pay, bill - bank.lo[BILLS[0]]))  # beyond it the data end
        crossing = np.floor(np.concatenate([_cuts(bank, PAYMENTS[0]) - pay, bill - _cuts(bank, BILLS[0])]))
        candidates = np.unique(np.concatenate([[1.0], crossing, crossing + 1]))
        candidates = candidates[(candidates >= 1) & (candidates <= top)]
    rows = [(c, actions.apply(x, action, c, bank.lo, bank.hi)) for c in candidates]
    rows = [(c, y) for c, y in rows if y is not None]
    approved = (risk(bank.model, pd.DataFrame([y for _, y in rows]).astype(float)) <= bank.cutoff if rows
                else np.array([], dtype=bool))  # no amount within the data: nothing is approved
    first = int(np.argmax(approved)) if approved.any() else None
    bank.memo[key] = {"action": action, "least": None if first is None else int(rows[first][0]),
                      "larger_also_approved": None if first is None else bool(approved[first:].all())}
    return bank.memo[key]


def describe(bank: Bank, x: pd.Series, cols: list, method: int = 1) -> dict:
    """The applicant's and the reference group's values as statements use them, for the tools and the checks alike.
    Money: the mean over the columns. One month's payment status: its kind and months late (0 if not late); the
    group as the share late that month. The six months of payment history: the number of late months; the group as
    its mean number (decision 33)."""
    group = bank.refs[method][cols]
    if cols == STATUS:
        return {"you": float((x[cols] >= 1).sum()), "group": float((group >= 1).sum(axis=1).mean())}
    if cols[0] in STATUS:
        code = float(x[cols[0]])
        return {"you": max(code, 0.0), "kind": kind(code), "group": float((group[cols[0]] >= 1).mean()),
                "group_months_late": float(group[cols[0]].clip(lower=0).mean())}
    return {"you": float(x[cols].mean()), "group": float(group.to_numpy().mean())}


def number(stated, decimals, actual: float) -> bool | None:
    """True if `actual` rounds to `stated` at the precision written: 0.31 (decimals 2) covers [0.305, 0.315]. A value
    exactly halfway rounds either way: the tools round half to even, so NT$26,542.5 is shown as 26,542 (decision 56)."""
    if stated is None or decimals is None:
        return None
    half = 0.5 * 10.0**-decimals
    return bool(stated - half <= actual <= stated + half)


def _all(checks: list) -> bool | None:
    return None if not checks or any(c is None for c in checks) else all(checks)


def _outcome(bank, x, q):
    r = applicant_risk(bank, x)
    if q.get("quantity") == "decision" and q.get("decision"):
        return (r <= bank.cutoff) == (q["decision"] == "approved")
    actual = {"risk": r, "cutoff": bank.cutoff, "gap": r - bank.cutoff}.get(q.get("quantity"))
    return None if actual is None else number(q.get("value"), q.get("decimals"), actual)


def _action(bank, x, q):
    y = actions.apply(x, q.get("action"), q.get("amount"), bank.lo, bank.hi, q.get("months"))
    if y is None:
        return None
    before, after = applicant_risk(bank, x), applicant_risk(bank, y)
    checks = []
    if q.get("decision"):
        checks.append((after <= bank.cutoff) == (q["decision"] == "approved"))
    if q.get("direction"):
        checks.append(after < before if q["direction"] == "lower" else after > before)
    if q.get("quantity") == "risk":
        checks.append(number(q.get("value"), q.get("decimals"), after))
    return _all(checks)


def _least(bank, x, q):
    a = q.get("action")
    if a not in actions.ACTIONS:
        return None
    if a == "both" and q.get("months") is None:  # "no way to approval": not even both together (decision 43)
        if q.get("decision") != "declined":
            return None
        return least(bank, x, "pay_more")["least"] is None and all(
            least(bank, x, "both", k)["least"] is None for k in range(1, 7))
    if a == "both" and actions.apply(x, "on_time", q["months"], bank.lo, bank.hi) is None:
        return None  # those months on time are not possible within the data: not checkable, not "nothing helps"
    best = least(bank, x, a, q.get("months"))["least"]
    if q.get("value") is not None:  # "you would need at least ...": the true least rounded up at the precision
        if best is None or q.get("decimals") is None:  # written, never down (decision 29): with a least of 20,564,
            return False if best is None else None  # "20,570" and "21,000" are true, "20,560" is not
        return bool(0 <= q["value"] - best < 10.0 ** -q["decimals"])
    if q.get("decision") == "declined":  # "nothing of this kind would get you approved"
        return best is None
    return None


def _reason(bank, x, q, m):
    b = q.get("block")
    if b not in BLOCKS:
        return None
    phi = shortfalls(bank, x, m)
    checks = []
    if q.get("rank") is not None:  # position among all blocks, ordered by how much they raised the risk
        checks.append(list(phi.sort_values(ascending=False).index).index(b) + 1 == q["rank"])
    if q.get("direction"):  # higher: the block raised the risk; lower: it lowered it
        checks.append(bool(phi[b] > 0) if q["direction"] == "higher" else bool(phi[b] < 0))
    return _all(checks)


def _compare(bank, x, q, m):
    a, b = q.get("block"), q.get("other_block")
    if a not in BLOCKS or b not in BLOCKS or a == b:
        return None
    phi = shortfalls(bank, x, m)
    return bool(phi[a] > phi[b])  # a counted more against the applicant than b


def _measure(bank, x, q, m):
    f = q.get("feature")
    cols = BLOCKS.get(f, [f] if f in FEATURES else None)
    if cols is None:
        return None
    d, checks = describe(bank, x, cols, m), []
    if q.get("status"):  # the kind of one month
        checks.append(d["kind"] == q["status"] if "kind" in d else None)
    if q.get("value") is not None:
        checks.append(number(q["value"], q.get("decimals"), d["you"]))
    if q.get("reference_value") is not None:
        checks.append(number(q["reference_value"], q.get("decimals"), d["group"]))
    if q.get("direction"):  # one month: months late against the group's mean months late
        ref = d.get("group_months_late", d["group"])
        checks.append(d["you"] > ref if q["direction"] == "higher" else d["you"] < ref)
    return _all(checks)


def _used(bank, x, q):
    a = q.get("feature")
    if a not in ATTRIBUTES or q.get("is_used") is None:
        return None
    return (a in BLOCKS) == q["is_used"]  # the model uses its blocks and nothing else


def combine(a, b):
    """One verdict from methods 1 and 2: the shared verdict, DEPENDS if they differ, None if either is None."""
    if a is None or b is None:
        return None
    return a if a == b else DEPENDS


def run(bank: Bank, x: pd.Series, q: dict, methods: tuple = (1, 2)):
    """The verdict of one query. Reasons and comparisons depend on the reference group, so the evaluation checks them
    under both methods (plan §3.5) and the gate under method 1 alone, the bank's own method, which also orders the
    reasons in the notice. A measurement names its group, the customers approved just inside the approval line, so
    it is checked against that group, method 1 (decision 50)."""
    t = q.get("type")
    if t == "outcome":
        v = _outcome(bank, x, q)
    elif t == "action":
        v = _action(bank, x, q)
    elif t == "least":
        v = _least(bank, x, q)
    elif t in ("reason", "compare"):
        f = _reason if t == "reason" else _compare
        v = combine(*(f(bank, x, q, m) for m in methods)) if len(methods) == 2 else f(bank, x, q, methods[0])
    elif t == "measure":
        v = _measure(bank, x, q, 1)
    elif t == "used":
        v = _used(bank, x, q)
    else:
        v = None
    return v if v is None or isinstance(v, str) else bool(v)


def complete(bank: Bank, x: pd.Series, stated: set, methods: tuple = (1, 2)):
    """Whether a notice states every principal reason (Reg B comment 9(b)(2)-4): `stated` holds the blocks its
    sentences give as reasons."""
    v = [set(reasons(shortfalls(bank, x, m))) <= stated for m in methods]
    return combine(*v) if len(v) == 2 else v[0]
