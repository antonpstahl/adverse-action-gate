"""The writer (plan §3.3). It first writes the adverse-action notice, then answers the declined applicant's questions,
with tools that read the model. What it writes is checked in gate.py."""

import json

import pandas as pd

from aa_gate import actions
from aa_gate.data import DESCRIPTIONS, label
from aa_gate.llm import call, price, text
from aa_gate.queries import ATTRIBUTES, Bank, applicant_risk, describe, least, shortfalls
from aa_gate.reference import reasons
from aa_gate.settings import BLOCKS, FEATURES, STATUS, WRITER, WRITER_EFFORT

HANDOFF = "I can't answer that here. I have passed your question to a colleague, who will get back to you."

NOTICE_REQUEST = "Write the adverse-action notice for this customer."

NOTICE_SYSTEM = """You write the adverse-action notice for a bank customer whose request for a higher credit limit \
was declined by the bank's credit model. Use the tools for every fact. Say that the request was declined and give \
every factor the tools mark as a principal reason, in their order, in plain words and in at most four short \
sentences. Give figures rather than words of degree such as "much", "slightly" or "generally". Promise nothing and \
add nothing else."""

WRITER_SYSTEM = """You answer questions from a bank customer whose request for a higher credit limit was declined by \
the bank's credit model; the conversation starts with the notice you wrote. Use the tools for every fact about the \
decision, and never state a number, ranking or comparison that a tool did not give you. If a question assumes \
something the tools do not support, say so. Answer in at most four short sentences, in plain words, without \
technical terms. Give figures rather than words of degree such as "much", "slightly" or "generally". Over the six \
months, give balances and payments as averages and payment history as the number of late months; give single months \
only when the customer asks, and never a range. When you say what an action would do, say that it is the model's \
estimate if nothing else changes, and do not promise approval. When you give the least a customer would have to \
pay, give it as the tool does or round it up, never down. Before you say that nothing would get the customer \
approved, check least_action with both. If the tools cannot answer the question, for example \
whether the decision is fair, call hand_off."""


def _tool(name: str, description: str, properties: dict | None = None) -> dict:
    return {"name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": properties or {}, "required": list(properties or {}),
                             "additionalProperties": False}}


TOOLS = [
    _tool("get_decision", "The customer's estimated risk of default, the highest risk the bank approves, and the "
          "decision. Risks are in percent."),
    _tool("get_reasons", "The factors ranked by how much each raised the customer's risk compared with customers "
          "approved just inside the approval line. 'principal' marks the principal reasons for the notice. The "
          "model uses these factors and nothing else."),
    _tool("get_profile", "The customer's own values next to those of customers approved just inside the approval "
          "line: amounts in whole NT dollars, each month's payment status with the share of those customers who were "
          "late that month, and six-month summaries."),
    _tool("simulate", "The model's estimate if the customer takes one action and nothing else changes. on_time: "
          "pays on time for the next `amount` months (1-6), each month their six-month average payment. pay_more: "
          "pays `amount` NT dollars more this month. both: pays on time for the next `months` months and `amount` NT "
          "dollars more in the last of them. `months` is null except for both. Returns the new risk and decision, or "
          "an error if the action is not possible.",
          {"action": {"type": "string", "enum": list(actions.ACTIONS)}, "amount": {"type": "number"},
           "months": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}),
    _tool("least_action", "The least the customer would have to do, with nothing else changing, to be approved: the "
          "fewest months of paying on time (on_time), the smallest extra payment this month in whole NT dollars "
          "(pay_more), or for each number of months on time the smallest extra payment in the last of them (both; 0 "
          "if the months alone are enough). Exact, from the model; least is null if nothing within the bank's data "
          "would be enough.",
          {"action": {"type": "string", "enum": list(actions.ACTIONS)}}),
    _tool("hand_off", "Pass the question to a human colleague when the tools cannot answer it.",
          {"reason": {"type": "string"}}),
]


def facts(bank: Bank, x: pd.Series, name: str, args: dict) -> dict:
    """What a tool returns. Everything comes from the model and Reg B's method 1, the bank's own method."""
    if name == "get_decision":
        r = applicant_risk(bank, x)  # the gap is rounded once here: in the pilot the writer subtracted two
        return {"risk_percent": round(100 * r, 1),  # rounded values and missed by 0.0005 (decision 16)
                "highest_approved_risk_percent": round(100 * bank.cutoff, 1),
                "gap_percentage_points": round(100 * (r - bank.cutoff), 1),
                "decision": "approved" if r <= bank.cutoff else "declined"}
    if name == "get_reasons":
        phi = shortfalls(bank, x, 1)
        top = reasons(phi)
        return {"factors": [{"factor": b, "rank": i + 1, "principal": b in top,
                             "effect": "raised your risk" if phi[b] > 0 else
                             "lowered your risk" if phi[b] < 0 else "made no difference"}
                            for i, b in enumerate(phi.sort_values(ascending=False).index)],
                "not_used_by_the_model": [a for a in ATTRIBUTES if a not in BLOCKS]}
    if name == "get_profile":  # kinds of months, shares late, whole NT$ (decisions 29, 32, 33)
        rows, months = [], {}
        for c in FEATURES:
            d = describe(bank, x, [c])
            rows.append({"feature": c, "meaning": DESCRIPTIONS[c], "you": label(x[c]),
                         "approved_near_line_share_late_percent": round(100 * d["group"], 1)} if c in STATUS else
                        {"feature": c, "meaning": DESCRIPTIONS[c], "you": round(d["you"]),
                         "approved_near_line_average": round(d["group"])})
        for b, cols in BLOCKS.items():
            if len(cols) > 1:
                d = describe(bank, x, cols)
                months[b] = ({"late_months_of_six": int(d["you"]),
                              "approved_near_line_average_late_months": round(d["group"], 1)} if cols == STATUS else
                             {"six_month_average": round(d["you"]), "approved_near_line_average": round(d["group"])})
        return {"features": rows, "six_months": months}
    if name == "simulate":
        y = actions.apply(x, args["action"], args["amount"], bank.lo, bank.hi, args.get("months"))
        if y is None:
            return {"error": "This action is not possible, or it goes beyond the range of the bank's data."}
        r = applicant_risk(bank, y)
        return {"new_risk_percent": round(100 * r, 1), "decision": "approved" if r <= bank.cutoff else "declined"}
    if name == "least_action":
        if args["action"] == "both":
            return {"action": "both", "least_extra_payment_by_months_on_time": [
                {"months": k, "least": least(bank, x, "both", k)["least"]} for k in range(1, 7)]}
        return least(bank, x, args["action"])


def write(bank: Bank, x: pd.Series, history: list[tuple[str, str]], question: str,
          feedback: tuple[str, str] | None = None) -> tuple[str | None, float]:
    """The writer's answer and its cost: text, HANDOFF, or None for a refusal. `history` holds the earlier questions
    and the answers the customer saw; `feedback` is (draft, problems) for the one repair. The feedback goes into the
    system prompt, not into the conversation, so the repaired answer does not speak to the customer about a draft
    they never saw."""
    messages = [m for q, a in history for m in ({"role": "user", "content": q}, {"role": "assistant", "content": a})]
    messages.append({"role": "user", "content": question})
    system = NOTICE_SYSTEM if question == NOTICE_REQUEST else WRITER_SYSTEM
    if feedback:
        system += (f"\n\nA draft of your answer to the customer's last question failed the bank's check before it "
                   f"was sent. The draft:\n{feedback[0]}\n\nWhat the check found:\n{feedback[1]}\n\nWrite the answer "
                   "again from the start. The customer never saw the draft, so do not mention it.")
    cost = 0.0
    for _ in range(12):
        r = call(model=WRITER, system=system, tools=TOOLS, messages=messages,
                 output_config={"effort": WRITER_EFFORT})
        cost += price(r)
        if r["stop_reason"] == "refusal":
            return None, cost
        uses = [b for b in r["content"] if b["type"] == "tool_use"]
        if not uses:
            return text(r), cost
        if any(u["name"] == "hand_off" for u in uses):
            return HANDOFF, cost
        messages += [{"role": "assistant", "content": r["content"]},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"],
                                                   "content": json.dumps(facts(bank, x, u["name"], u["input"]))}
                                                  for u in uses]}]
    raise RuntimeError("the writer did not answer within 12 rounds")
