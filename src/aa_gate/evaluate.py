"""The metrics (plan §4) per model and condition, and pooled over the models, with 95% intervals bootstrapped over
applicants, and the difference of the error rates with its interval; as in He & Martens 2026, no test (decision 44).

    uv run python -m aa_gate.evaluate --run pilot6 --kind ebm xgb

The answer rate needs only the run. The error rate needs the scorer's reading (reading.py; decision 45): the gate's
own reader never scores anything. Anton's blind check of a sample gives the scorer's agreement with a person. Writes
results/<run>/metrics_<kinds>.csv, difference_<kinds>.csv, agreement_<kinds>.csv and answered_<kinds>.csv (the
answer rate by question).

The counterfactual test (decision 59), a secondary analysis of RQ1, keeps each released answer and its readings and
changes the facts: does the gate's verdict follow the model? counterfactual_<kinds>.csv: error rate and answer rate on
the changed applicants; counterfactual_cases_<kinds>.csv: one row per case, every disagreement with both readings."""

import argparse
import functools
import json
import random

import numpy as np
import pandas as pd

from aa_gate.gate import ANSWERED, check, stated_reasons
from aa_gate.queries import DEPENDS, TYPES, applicant_risk, build, complete, shortfalls
from aa_gate.reading import KEY, labels
from aa_gate.reference import reasons
from aa_gate.settings import BLOCKS, RESULTS, SEED

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


def verdicts(bank, x, answer: str, reading: dict, notice: bool, methods: tuple = (1, 2)) -> tuple[list, bool, list]:
    """The checked sentences, whether they cover the answer, and every verdict, a notice's completeness last."""
    sentences, covered = check(bank, x, answer, reading, methods)
    found = [v for s in sentences for v in s["verdicts"]]
    return sentences, covered, found + ([complete(bank, x, stated_reasons(sentences), methods)] if notice else [])


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
        sentences, _, found = verdicts(bank, x, answers[r["item"]], lab[r["item"]], r["script"] == -1)
        rows.append({**r, "kind": kind, "wrong": False in found, "depends": DEPENDS in found,
                     "unmappable": any(s["kind"] == "unmappable" for s in sentences)})
    return pd.DataFrame(rows)


def counterfactual(run: str, kind: str, turns: pd.DataFrame) -> pd.DataFrame:
    """One row per answer the gate released and changed applicant: x with one block set to the values of a row of
    method 1's group, drawn once per applicant (a block whose values stay is skipped). Neither reader sees the
    applicant's data (gate.request_text), so both stored readings hold: the gate releases if its reading passes, and
    the truth is the rule of errors() under method 1, the gate's standard. Answers false on x stay out of the rates."""
    bank, ref, scores, rows = _bank(kind), _bank(kind).refs[1], labels(run, kind), []
    key = pd.read_csv(RESULTS / run / "reading" / f"key_{kind}.csv").query("condition == 'gated'").set_index(KEY).item
    for t in turns[(turns.kind == kind) & (turns.exit == "released")].itertuples():
        x, notice, item = bank.declined.loc[t.applicant], t.script == -1, key[t.applicant, t.script, t.turn]
        r = ref.iloc[random.Random(f"{SEED}-{t.applicant}").randrange(len(ref))]
        assert t.rounds[-1]["draft"] == t.text
        false_on_x = False in verdicts(bank, x, t.text, scores[item], notice, (1,))[2]
        for b, cols in [(b, cols) for b, cols in BLOCKS.items() if (x[cols] != r[cols]).any()]:
            xb = x.where(~x.index.isin(cols), r)  # block b from r, the others from x
            mine, _, truth = verdicts(bank, xb, t.text, scores[item], notice, (1,))
            gate, covered, found = verdicts(bank, xb, t.text, {"sentences": t.rounds[-1]["sentences"]}, notice, (1,))
            releases = covered and all(s["ok"] for s in gate) and False not in found
            types = sorted({q["type"] for s in mine for q, v in zip(s["queries"], s["verdicts"]) if v is False})
            types += ["missing reason"] if notice and truth[-1] is False else []
            disagree = false_on_x or (wrong := False in truth) == releases  # for Anton: both readings by sentence
            by = [{s["sentence"].strip(): {"kind": s["kind"], "verdicts": s["verdicts"], "queries": [
                {f: v for f, v in q.items() if v is not None} for q in s["queries"]]} for s in rd} for rd in (mine, gate)]
            out = [{"sentence": q, "scorer": by[0].get(q), "gate": by[1].get(q)} for q in {**by[0], **by[1]}]
            out += [{"principal reasons": reasons(shortfalls(bank, xb, 1)), "scorer states": sorted(stated_reasons(mine)),
                     "gate states": sorted(stated_reasons(gate))}] if notice else []
            rows.append({"kind": kind, "applicant": t.applicant, "block": b,
                         "approved": applicant_risk(bank, xb) <= bank.cutoff, "script": t.script, "turn": t.turn,
                         "item": item, "false_on_x": false_on_x, "scorer_false": wrong, "gate_releases": releases,
                         "false_types": ";".join(types), "readings": json.dumps(out) if disagree else ""})
    return pd.DataFrame(rows)


def counterfactual_table(cf: pd.DataFrame) -> pd.DataFrame:
    """Of the cases the change made false, the share the gate still releases (error rate); of those that stay true, the
    share it blocks (cost in answer rate); by the scorer's false query type and by approval; the cases false on x."""
    rel, blk, parts = ("released though false", "gate_releases"), ("blocked though true", "blocked"), []
    for k, d in by_kind(cf.assign(blocked=~cf.gate_releases)):
        false, true = d[~d.false_on_x & d.scorer_false], d[~d.false_on_x & ~d.scorer_false]
        p = [("false before the change", "false_on_x", "all", d), (*rel, "all", false), (*blk, "all", true)]
        p += [(*rel, t, false[false.false_types.str.split(";").map(lambda ts: t in ts)])
                for t in TYPES + ["missing reason"]]
        p += [(m, c, g, s[s.approved == (g == "approved")]) for g in ("approved", "declined") for m, c, _, s in p[1:3]]
        parts += [(k, *q) for q in p]
    return pd.DataFrame([{"kind": k, "measure": m, "group": g, "count": int(s[c].sum()), **bootstrap(s, c)}
                         for k, m, c, g, s in parts if len(s)])


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
    if scored:  # the counterfactual test (decision 59), after the tables above, which it leaves as they are
        cf = pd.concat([counterfactual(a.run, k, turns) for k in scored], ignore_index=True)
        cf.to_csv(folder / f"counterfactual_cases_{name}.csv", index=False)
        (summary := counterfactual_table(cf)).to_csv(folder / f"counterfactual_{name}.csv", index=False)
        print("\n", summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
