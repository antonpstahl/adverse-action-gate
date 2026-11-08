"""The metrics (plan §4) per model and condition, and pooled over the models, with 95% intervals bootstrapped over
applicants, and the difference of the error rates with its interval; as in He & Martens 2026, no test (decision 44).

    uv run python -m aa_gate.evaluate --run confirm --kind ebm xgb

The answer rate needs only the run. The error rate needs the scorer's reading (reading.py; decision 45): the gate's
own reader never scores anything. Anton's blind check of a sample gives the scorer's agreement with a person. Writes
results/<run>/metrics_<kinds>.csv, difference_<kinds>.csv, agreement_<kinds>.csv and answered_<kinds>.csv (the
answer rate by question)."""

import argparse
import functools

import numpy as np
import pandas as pd

from aa_gate.gate import ANSWERED, check, stated_reasons
from aa_gate.queries import DEPENDS, build, complete
from aa_gate.reading import KEY, labels
from aa_gate.settings import RESULTS, SEED

_bank = functools.cache(build)  # one bank per model, so its Shapley values and least actions are computed once


def _sums(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per applicant, the count in `col` and the number of rows."""
    return pd.DataFrame({"kind": df["kind"].to_numpy(), "applicant": df["applicant"].to_numpy(),
                         "x": df[col].to_numpy(float), "w": 1.0}).groupby(["kind", "applicant"])[["x", "w"]].sum()


def bootstrap(df: pd.DataFrame, col: str, B: int = 2000) -> dict:
    """The share in `col` and its 95% percentile interval, resampling applicants."""
    s = _sums(df, col)
    x, n = s.x.to_numpy(), s.w.to_numpy()
    pick = np.random.default_rng(SEED).integers(len(x), size=(B, len(x)))
    lo, hi = np.percentile(x[pick].sum(axis=1) / n[pick].sum(axis=1), [2.5, 97.5])
    return {"share": float(x.sum() / n.sum()), "lo": float(lo), "hi": float(hi), "n": len(df)}


def difference(err: pd.DataFrame, B: int = 2000) -> dict:
    """The error rate without the gate minus with it, and its 95% percentile interval, resampling applicants with both
    conditions together, so an error that runs through a conversation counts with its applicant, not as many
    independent pairs. Also the questions both conditions answered where the gate kept an answer from being wrong
    (prevented) or made one wrong (caused). There is no test (decision 44)."""
    u, g = (_sums(err[err.condition == c], "wrong") for c in ("unchecked", "gated"))
    u, g = u.align(g, join="outer", fill_value=0.0)
    pick = np.random.default_rng(SEED).integers(len(u), size=(B, len(u)))
    rate = lambda s: s.x.to_numpy()[pick].sum(axis=1) / s.w.to_numpy()[pick].sum(axis=1)  # noqa: E731
    lo, hi = np.percentile(rate(u) - rate(g), [2.5, 97.5])
    w = err.pivot_table(index=["kind", *KEY], columns="condition", values="wrong").dropna()
    return {"difference": float(u.x.sum() / u.w.sum() - g.x.sum() / g.w.sum()), "lo": float(lo), "hi": float(hi),
            "prevented": int(((w.unchecked == 1) & (w.gated == 0)).sum()),
            "caused": int(((w.unchecked == 0) & (w.gated == 1)).sum())}


def errors(run: str, kind: str, source: str = "scores") -> pd.DataFrame:
    """One row per read answer and condition. Wrong: a query of the reading is False on the model, or a notice leaves
    out a principal reason (Reg B comment 9(b)(2)-4). `source` is the scorer's reading or Anton's labels."""
    bank, lab = _bank(kind), labels(run, kind, source)
    folder = RESULTS / run / "reading"
    answers = pd.read_csv(folder / f"sample_{kind}.csv").set_index("item")["answer"]
    rows = []
    for r in pd.read_csv(folder / f"key_{kind}.csv").to_dict("records"):
        if r["item"] not in lab:
            continue
        x = bank.declined.loc[r["applicant"]]
        sentences, _ = check(bank, x, answers[r["item"]], lab[r["item"]])
        verdicts = [v for s in sentences for v in s["verdicts"]]
        if r["script"] == -1:
            verdicts.append(complete(bank, x, stated_reasons(sentences)))
        rows.append({**r, "kind": kind, "wrong": False in verdicts, "depends": DEPENDS in verdicts,
                     "unmappable": any(s["kind"] == "unmappable" for s in sentences)})
    return pd.DataFrame(rows)


def agreement(run: str, kind: str) -> dict:
    """Anton's blind check against the scorer (decision 45): per answer, who finds it wrong; per sentence that both
    quote alike, whether they chose the same kind (no claim, claims, not expressible)."""
    human = errors(run, kind, "labels").drop_duplicates("item").set_index("item").wrong.astype(bool)
    scorer = errors(run, kind).drop_duplicates("item").set_index("item").wrong.reindex(human.index).astype(bool)
    mine, theirs = labels(run, kind, "labels"), labels(run, kind)
    pairs = [(s["kind"], {t["sentence"].strip(): t["kind"] for t in theirs[i]["sentences"]}.get(s["sentence"].strip()))
             for i in mine for s in mine[i]["sentences"]]
    pairs = [(a, b) for a, b in pairs if b is not None]
    return {"answers": len(human), "both wrong": int((human & scorer).sum()), "only Anton": int((human & ~scorer).sum()),
            "only scorer": int((~human & scorer).sum()), "sentences": len(pairs),
            "same kind": sum(a == b for a, b in pairs)}


def by_kind(df: pd.DataFrame):
    """Each model's rows, then both models' rows together."""
    yield from df.groupby("kind")
    if df["kind"].nunique() > 1:
        yield "both", df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--kind", choices=["ebm", "xgb"], nargs="+", required=True)
    a = p.parse_args()
    folder, name = RESULTS / a.run, "_".join(a.kind)
    turns = pd.concat([pd.read_json(folder / f"turns_{k}.jsonl", lines=True) for k in a.kind], ignore_index=True)
    turns["answered"] = turns.exit.isin(ANSWERED)
    rows = [{"kind": k, "condition": c, "metric": "answer rate", **bootstrap(g, "answered")}
            for k, d in by_kind(turns) for c, g in d.groupby("condition")]
    print(pd.crosstab([turns.kind, turns.condition], turns.exit), "\n")
    print("cost per answer (USD, list price):", turns.groupby(["kind", "condition"]).cost.mean().round(3).to_dict(), "\n")
    answered = turns.pivot_table(index=["script", "turn"], columns=["kind", "condition"], values="answered")
    answered.to_csv(folder / f"answered_{name}.csv")
    if scored := [k for k in a.kind if (folder / "reading" / f"scores_{k}.jsonl").exists()]:
        err = pd.concat([errors(a.run, k) for k in scored], ignore_index=True)
        rows += [{"kind": k, "condition": c, "metric": metric, **bootstrap(g, col)}  # the last two: diagnostics
                 for col, metric in (("wrong", "error rate"), ("depends", "reference-dependent"),
                                     ("unmappable", "not checkable"))
                 for k, d in by_kind(err) for c, g in d.groupby("condition")]
        diff = pd.DataFrame([{"kind": k, **difference(d)} for k, d in by_kind(err)])
        diff.to_csv(folder / f"difference_{name}.csv", index=False)
        print(diff.round(4).to_string(index=False), "\n")
        if checked := [k for k in scored if (folder / "reading" / f"labels_{k}.jsonl").exists()]:
            agree = pd.DataFrame([{"kind": k, **agreement(a.run, k)} for k in checked])
            agree.to_csv(folder / f"agreement_{name}.csv", index=False)
            print(agree.to_string(index=False), "\n")
    table = pd.DataFrame(rows)
    table.to_csv(folder / f"metrics_{name}.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
