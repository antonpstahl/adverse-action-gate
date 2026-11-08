"""The reading (plan §3.3, decision 45). Every shown answer is read by the scorer, a model of another vendor
(settings.SCORER) that follows gate.READER_SYSTEM and writes the format of gate.READING; identical answers are one
item. Anton reads N_CHECK of them per model blind, half where the conditions differ, to check the scorer. Neither
sees the condition or the model's facts.

    uv run python -m aa_gate.reading --run confirm --kind xgb

writes results/<run>/reading/sample_<kind>.csv (every item), key_<kind>.csv (for the evaluation), scores_<kind>.jsonl
(the scorer's reading), check_<kind>.csv and page_<kind>.html, the reading page for the check, with the answers split
into sentences and no trace of the condition. Anton's labels go into results/<run>/reading/labels_<kind>.jsonl, one
{"item": ..., "sentences": [...]} per line."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from aa_gate.gate import ANSWERED, READER_SYSTEM, READING, request_text, transcript
from aa_gate.llm import SPENT, batch_openai
from aa_gate.settings import MAX_TOKENS, N_CHECK, RESULTS, SCORER, SCORER_EFFORT, SEED
from aa_gate.writer import NOTICE_REQUEST

KEY = ["applicant", "script", "turn"]
SENTENCE_END = re.compile(r'(?:(?<=[.!?])|(?<=[.!?]["”]))\s+(?=[A-Z“"(0-9])')


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]


def page(run: str, kind: str) -> Path:
    """The reading page: the check's answers as sentences, built from reading_page.html."""
    folder = RESULTS / run / "reading"
    items = pd.read_csv(folder / f"sample_{kind}.csv")
    items = items[items.item.isin(pd.read_csv(folder / f"check_{kind}.csv").item)]
    data = [{"item": r.item, "conversation": r.conversation, "sentences": sentences(r.answer)}
            for r in items.itertuples()]
    html = Path(__file__).with_name("reading_page.html").read_text()
    out = folder / f"page_{kind}.html"
    out.write_text(html.replace("__ITEMS__", json.dumps(data).replace("<", "\\u003c")))
    return out


def export(run: str, kind: str) -> pd.DataFrame:
    turns = pd.read_json(RESULTS / run / f"turns_{kind}.jsonl", lines=True)
    conv = turns.groupby(KEY[:2] + ["condition"])
    notices = turns[turns.script == -1].set_index(["applicant", "condition"]).text

    def context(row) -> str:
        earlier = conv.get_group((row.applicant, row.script, row.condition))
        earlier = earlier[earlier.turn < row.turn]
        history = list(zip(earlier.question, earlier.text))
        if row.script != -1:  # every conversation starts from the notice the applicant saw
            history = [(NOTICE_REQUEST, notices[(row.applicant, row.condition)])] + history
        return transcript(history, row.question)

    rows = turns[turns.exit.isin(ANSWERED)].copy()
    rows["conversation"] = rows.apply(context, axis=1)
    items = rows[["conversation", "text"]].drop_duplicates().sample(frac=1, random_state=SEED).reset_index(drop=True)
    items["item"] = [f"R{i + 1:03d}" for i in range(len(items))]
    rows = rows.merge(items, on=["conversation", "text"])
    pair = rows.groupby(KEY).item  # an item is the same in both conditions if its question has one item, read twice
    same = ((pair.transform("size") == 2) & (pair.transform("nunique") == 1)).groupby(rows.item).all()
    differ, alike = same.index[~same].to_series(), same.index[same].to_series()
    half = min(N_CHECK // 2, len(differ))
    check = pd.concat([differ.sample(n=half, random_state=SEED),
                       alike.sample(n=min(N_CHECK - half, len(alike)), random_state=SEED)])
    out = RESULTS / run / "reading"
    out.mkdir(parents=True, exist_ok=True)
    items.rename(columns={"text": "answer"})[["item", "conversation", "answer"]].to_csv(
        out / f"sample_{kind}.csv", index=False)
    rows[["item", *KEY, "condition"]].to_csv(out / f"key_{kind}.csv", index=False)
    pd.DataFrame({"item": sorted(check)}).to_csv(out / f"check_{kind}.csv", index=False)
    return items


def score(run: str, kind: str) -> Path:
    """The scorer's reading of every item, in one batch: the guide and format of the gate's reader, another vendor's
    model. A refusal raises, so no answer counts as right because the scorer did not read it."""
    folder = RESULTS / run / "reading"
    items = pd.read_csv(folder / f"sample_{kind}.csv")
    bodies = [{"model": SCORER, "reasoning_effort": SCORER_EFFORT, "max_completion_tokens": MAX_TOKENS,
               "messages": [{"role": "system", "content": READER_SYSTEM},
                            {"role": "user", "content": request_text(r.conversation, r.answer)}],
               "response_format": {"type": "json_schema",
                                   "json_schema": {"name": "reading", "strict": True, "schema": READING}}}
              for r in items.itertuples()]
    responses = batch_openai(bodies)  # before the file is opened, so a failed batch leaves no partial scores
    out = folder / f"scores_{kind}.jsonl"
    with out.open("w") as f:
        for item, response in zip(items.item, responses):
            content = response["choices"][0]["message"]["content"]
            if content is None:
                raise RuntimeError(f"the scorer did not read {item}: {response['choices'][0]['message'].get('refusal')}")
            f.write(json.dumps({"item": item, **json.loads(content)}) + "\n")
    return out


def labels(run: str, kind: str, source: str = "scores") -> dict[str, dict]:
    """The scorer's reading (source "scores") or Anton's labels (source "labels"), by item."""
    path = RESULTS / run / "reading" / f"{source}_{kind}.jsonl"
    return {r["item"]: r for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--kind", choices=["ebm", "xgb"], required=True)
    a = p.parse_args()
    n = len(export(a.run, a.kind))
    print(f"{n} items; scores: {score(a.run, a.kind)} ({SPENT[0]:.2f} USD); check page: {page(a.run, a.kind)}")
