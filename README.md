# adverse-action-gate

A checked assistant for adverse-action reasons. An LLM answers a declined applicant's questions about their credit decision; a gate maps each statement to a query, runs it on the credit model and releases only what the model confirms. Plan: `../../Preparation/Revision_Plan_Lean_Thesis.md`. Rules for working here: `CLAUDE.md`. Decisions: `docs/decisions.md`.

## Layout

| File | Does | Target lines |
|---|---|---|
| `src/aa_gate/settings.py` | Seeds, approval rate, blocks, monotone constraints, models, the scorer, prices, paths | ~40 |
| `src/aa_gate/data.py` | Load TaiwanCredit without SEX, MARRIAGE, AGE, EDUCATION; split; kinds of payment status | ~50 |
| `src/aa_gate/model.py` | Fit the constrained EBM and XGBoost; risk, log-odds, cutoff, cut points | ~60 |
| `src/aa_gate/reference.py` | Reg B reference groups (methods 1 and 2), exact block Shapley, principal reasons | ~60 |
| `src/aa_gate/actions.py` | Customer actions as consistent feature changes | ~55 |
| `src/aa_gate/queries.py` | The bank (model, cutoff, groups), the exact least action, the seven query types, their fields (QUERY) and their verdicts | ~270 |
| `src/aa_gate/writer.py` | The writer with tools: the notice, then the answers | ~145 |
| `src/aa_gate/gate.py` | The gate's reader, the check, the gate, conversations in both conditions | ~205 |
| `src/aa_gate/llm.py` | Cached Claude calls and OpenAI batches with truncation guard, list price | ~100 |
| `src/aa_gate/run.py` | Run both conditions on a slice of the declined applicants | ~70 |
| `src/aa_gate/reading.py` | The scorer's reading of every answer, the blind check sample and its reading page, labels in | ~110 |
| `src/aa_gate/reading_page.html` | The reading page: sentences to queries in the reader's format, reading time per answer, labels saved per rater | ~470 |
| `src/aa_gate/evaluate.py` | Answer rate, error rate, intervals, the difference, the scorer's agreement with the check | ~130 |
| `src/aa_gate/study.py` | Planned: the participant page for the user study, notice beside a chat whose answers pass the gate | ~150 |
| `tests/` | Reference, actions, queries and gate verdicts against hand-computed cases and brute force | ~350 |

## Use

```
uv sync
uv run pytest
uv run python -m aa_gate.run --kind xgb --start 0 --n 10 --out pilot      # the pilot: the first ten declined applicants
uv run python -m aa_gate.run --kind ebm --start 10 --n 40 --out confirm --batch  # the confirmation run (decisions 41, 53),
uv run python -m aa_gate.run --kind xgb --start 10 --n 40 --out confirm --batch  # one per model, batched at half the price
uv run python -m aa_gate.reading --run confirm --kind ebm                 # the scorer reads every answer; the check page,
uv run python -m aa_gate.reading --run confirm --kind xgb                 # one per model
uv run python -m aa_gate.evaluate --run confirm --kind ebm xgb            # every table: rates, intervals, the difference
```

The API key comes from `ANTHROPIC_API_KEY` or a git-ignored `.env`. Every LLM call is cached in `cache/llm/`; with `AA_GATE_OFFLINE=1` an uncached call raises instead of calling the API. Fitted models are cached in `cache/models/`; delete them after changing a setting.

## Figures

`docs/architecture.svg`: the architecture in three layers: the assistant and its gate, the deterministic core, and the evaluation. `docs/dialog_flow.svg`: one dialog turn as a UML sequence diagram, with a verified example from pilot 6 (applicant 191). Both are drawn in TikZ for the working paper of 8 November 2026.

## Data

`data/taiwan_credit.xls`: UCI dataset 350, CC BY 4.0, see `data/ATTRIBUTION.txt`.
