"""The gate (plan §3.3). Its reader maps each sentence of the writer's draft to queries (queries.QUERY); the queries
run on the model; the gate releases the text, has the writer repair it once, or falls back to the template notice.
converse runs one conversation, unchecked or gated.

READER_SYSTEM is also the guide for the scorer and for the blind human check (reading.py), which use the same output
format (READING)."""

import json
import re
import time

import pandas as pd

from aa_gate.data import DESCRIPTIONS
from aa_gate.llm import call, price, text
from aa_gate.queries import ATTRIBUTES, QUERY, Bank, complete, run, shortfalls
from aa_gate.reference import reasons
from aa_gate.settings import BLOCKS, READER, READER_EFFORT
from aa_gate.writer import HANDOFF, NOTICE_REQUEST, write

ANSWERED = ("unchecked", "released")  # exits that count as an answer (plan §4)


def transcript(history: list[tuple[str, str]], question: str) -> str:
    """The conversation as the reader sees it; the notice is the opening turn."""
    def said(q: str) -> str:
        return "Bank: write the adverse-action notice." if q == NOTICE_REQUEST else f"Customer: {q}"
    return "\n\n".join([f"{said(q)}\nAssistant: {a}" for q, a in history] + [said(question)])


FEATURE_LINES = "\n".join(f"- {c}: {d}" for c, d in DESCRIPTIONS.items())

READER_SYSTEM = f"""You read a text that a bank's assistant wrote to a customer whose request for a higher credit \
limit was declined by the bank's credit model: the adverse-action notice or an answer to the customer's question. \
Split the text into its sentences, each quoted exactly as written, and decide for each sentence:
- no_claim: it states nothing about the customer's decision, the model or the data (a greeting, an offer of help, \
advice without a stated effect, a sentence that only names the comparison group of customers approved just inside \
the approval line, or a caveat: that figures are the model's estimates, how an estimate is made (each month paid at \
the six-month average, nothing else changing), that approval is not guaranteed, or what the assistant or its tools \
cannot tell).
- claims: everything it states about the decision, the model or the data can be written as the queries below. Write \
one query per claim.
- unmappable: it states something about the decision, the model or the data that no query can express (for example \
"generally", "slightly", how the model works in general, what is true of other customers, or a number no query \
covers).
Read what the sentence says to the customer, given the conversation. A sentence that refers to an earlier turn claims \
what it refers to. A sentence that names the comparison group and says what a factor did claims that effect. In a \
notice, each factor given as a reason is a reason query with direction higher, and with its rank if the reasons come \
in order. A bare yes or no claims the answer to the customer's question: to "Did my balance count more against me \
than my payment history?", "No" is compare with block payment history and other_block balance. Alternatives \
joined by or get one query each. Do not judge whether a claim is correct.

Factors: {", ".join(BLOCKS)}.
Features:
{FEATURE_LINES}

Queries. Fill the fields named and set every other field to null.
- outcome: quantity = decision with decision = approved or declined; or quantity = risk (the customer's estimated \
risk of default), cutoff (the highest risk the bank approves) or gap (risk minus cutoff), with value and decimals.
- action: action = on_time (pays on time for the next `amount` months), pay_more (pays `amount` NT dollars more \
this month) or both (pays on time for the next `months` months and `amount` NT dollars more), and the stated effect: \
decision (approved or declined), or direction (lower or higher risk), or quantity = risk with value and decimals (the \
new risk). A risk given for the action under discussion is that action's new risk, not the customer's current risk.
- least: action and either value with decimals (the fewest months, or the smallest extra payment in NT dollars, \
that the text says would be needed for approval with nothing else changing; for both, the smallest extra payment \
after `months` months on time) or decision = declined (the text says no number of months or amount of this kind \
would be enough). A text that says one action alone is not enough claims this for that action; a text that finds no \
way to approval at all claims it for both, with months null.
- reason: block and one or both of: rank (the factor's position when all factors are ordered by how much they \
raised the risk: 1 = the main reason, 2 = the next, and so on through the factors that lowered it), direction \
(higher = the factor raised the risk, counted against the customer or was a reason for the decline; lower = it \
lowered the risk, worked in their favour or was not a reason). A factor is a reason for the decline exactly when it \
raised the risk.
- compare: block counted more against the customer than other_block.
- measure: feature (a feature or a factor) and at least one of value and reference_value (each with decimals), \
direction and status. Money (balances, payments, credit limit): value is the customer's amount, for a factor the \
six-month average; reference_value is the average of customers approved just inside the approval line; direction \
says whether the customer's amount is higher or lower. One month's payment status: status is the kind of month \
(card not used, paid in full, paid on time, late); value is the months late, 0 if not late; reference_value is the \
share of those approved customers who were late that month; direction says whether the customer was later. The \
factor payment history: value is the number of late months out of six; reference_value is the approved customers' \
average number of late months; direction says whether the customer had more. Being on time or not late is 0 \
months late: one month with value 0, all six months as payment history with value 0. Use status for a sentence \
that names a kind: card not used, paid in full or late.
- used: feature (a factor, or one of {", ".join(a for a in ATTRIBUTES if a not in BLOCKS)}) and is_used (true if \
the sentence says the model uses it, false if it says the model does not; that such an attribute was not a reason or \
did not count against the customer says the model does not use it).

Numbers: one number per query, so a sentence with two numbers gets two queries. Write percentages as proportions \
(31.4% becomes 0.314; 9 percentage points becomes 0.09). decimals is the number of digits after the decimal point of \
that proportion as written (0.314: 3; 0.31: 2). For a whole number it is minus the number of its trailing zeros \
(150,000: -4; 20: -1; 3: 0). Every value and reference_value has its decimals. "About" or "almost" before a number does not change \
its precision."""

READING = {"type": "object", "additionalProperties": False, "required": ["sentences"], "properties": {"sentences": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["sentence", "kind", "queries"],
                               "properties": {"sentence": {"type": "string"},
                                              "kind": {"type": "string", "enum": ["no_claim", "claims", "unmappable"]},
                                              "queries": {"type": "array", "items": QUERY}}}}}}


def request_text(context: str, answer: str) -> str:
    """What a reader is given, the gate's reader and the scorer alike: the conversation and the answer to read."""
    return f"Conversation:\n{context}\n\nAnswer to read:\n{answer}"


def read(context: str, answer: str) -> tuple[dict, float]:
    r = call(model=READER, system=READER_SYSTEM, messages=[{"role": "user", "content": request_text(context, answer)}],
             output_config={"effort": READER_EFFORT, "format": {"type": "json_schema", "schema": READING}})
    if r["stop_reason"] == "refusal":  # nothing read is nothing released: the check finds the answer uncovered
        return {"sentences": []}, price(r)
    return json.loads(text(r)), price(r)


def check(bank: Bank, x: pd.Series, answer: str, reading: dict, methods: tuple = (1, 2)) -> tuple[list[dict], bool]:
    """Each sentence with its verdicts and whether it passes, and whether the sentences cover the whole answer.
    A sentence passes if it is quoted exactly and either makes no claim or has only true queries. The gate checks
    with method 1 alone (queries.run)."""
    out, rest = [], answer
    for s in reading["sentences"]:
        quote = s["sentence"].strip()
        found = bool(quote) and quote in rest
        rest = rest.replace(quote, "", 1)
        verdicts = [run(bank, x, q, methods) for q in s["queries"]] if s["kind"] == "claims" else []
        ok = found and (s["kind"] == "no_claim" or (s["kind"] == "claims" and bool(verdicts)
                                                     and all(v is True for v in verdicts)))
        out.append({**s, "found": found, "verdicts": verdicts, "ok": ok})
    return out, not re.search(r"[A-Za-z]{2,}", rest)


def _problems(sentences: list[dict], covered: bool) -> str:
    lines = []
    for s in sentences:
        if s["ok"]:
            continue
        if not s["found"]:
            why = "it could not be read reliably; state it more simply"
        elif s["kind"] == "unmappable":
            why = ("it cannot be checked against the model, as ranges and words of degree cannot; state it with "
                   "figures from the tools or leave it out")
        elif None in s["verdicts"] or not s["verdicts"]:
            why = "it cannot be checked, for example an action the bank's data do not cover; leave it out"
        else:
            why = "the model does not confirm it; correct it with the tools or leave it out"
        lines.append(f'- "{s["sentence"]}": {why}.')
    if not covered:
        lines.append("- Part of the answer is not in plain sentences; write plain sentences only.")
    return "\n".join(lines)


def template_notice(bank: Bank, x: pd.Series) -> str:
    """The fallback: the notice as banks write it today, the principal reasons by method 1."""
    top = reasons(shortfalls(bank, x, 1))
    return ("Your request for a higher credit limit was declined. The principal reasons were: "
            + "; ".join(top) + ".") if top else "Your request for a higher credit limit was declined."


def stated_reasons(sentences: list[dict]) -> set:
    """The blocks a text gives as reasons (reason queries with direction higher or a rank), true or not: whether a
    reason is stated correctly is the check's business, whether it is stated at all is this one's. With four blocks,
    a principal reason is exactly a block that raised the risk (decision 36)."""
    return {q["block"] for s in sentences for q in s["queries"]
            if q.get("type") == "reason" and (q.get("direction") == "higher" or q.get("rank") is not None)}


def gate(bank: Bank, x: pd.Series, history: list[tuple[str, str]], question: str, draft: str) -> dict:
    """Check the draft; if anything fails, have it repaired once and check again; then release it or fall back to the
    template notice. A notice must also state every principal reason. Texts are never cut down to the sentences that
    passed: in the smoke run that left "Those three lowered your risk." without the sentence that named the three."""
    context, cost, rounds = transcript(history, question), 0.0, []
    for attempt in (0, 1):
        reading, c = read(context, draft)
        sentences, covered = check(bank, x, draft, reading, methods=(1,))
        cost += c
        missing = question == NOTICE_REQUEST and not complete(bank, x, stated_reasons(sentences), methods=(1,))
        rounds.append({"draft": draft, "sentences": sentences, "covered": covered, "missing_reason": missing})
        if covered and all(s["ok"] for s in sentences) and not missing:
            return {"exit": "released", "text": draft, "rounds": rounds, "cost": cost}
        if attempt == 0:
            problems = _problems(sentences, covered)
            if missing:
                problems += "\n- The notice leaves out a principal reason; give every factor marked principal."
            draft, c = write(bank, x, history, question, feedback=(draft, problems))
            cost += c
            if draft is None or draft == HANDOFF:
                return {"exit": "handoff", "text": HANDOFF, "rounds": rounds, "cost": cost}
    return {"exit": "template", "text": template_notice(bank, x), "rounds": rounds, "cost": cost}


def converse(bank: Bank, x: pd.Series, script: list[str], gated: bool,
             history: list[tuple[str, str]] | None = None) -> list[dict]:
    """One conversation: each question is answered with what the customer saw so far as history, which starts with
    the notice. `seconds` is the wall time of a turn; it measures the wait only where the calls were not cached."""
    history, turns = list(history or []), []
    for question in script:
        start = time.perf_counter()
        draft, cost = write(bank, x, history, question)
        if draft is None:
            turn = {"exit": "refusal", "text": HANDOFF, "cost": cost}
        elif draft == HANDOFF:
            turn = {"exit": "handoff", "text": HANDOFF, "cost": cost}
        elif not gated:
            turn = {"exit": "unchecked", "text": draft, "cost": cost}
        else:
            turn = gate(bank, x, history, question, draft)
            turn["cost"] += cost
        turns.append({"question": question, "draft": draft, **turn, "seconds": round(time.perf_counter() - start, 1)})
        history.append((question, turn["text"]))
    return turns
