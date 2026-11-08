"""Claude calls, each cached on disk under a hash of the full request, so a rerun never pays twice. A truncated answer
(stop_reason max_tokens) raises and is not cached. A refusal is returned as is: there is no fallback model, so every
answer comes from the model named in settings (pre-study D39). The scorer's OpenAI calls (decision 45) go through
the Batch API and are cached and guarded the same way."""

import hashlib
import json
import os
import time

import anthropic
import openai

from aa_gate.settings import CACHE, MAX_TOKENS, PRICE, ROOT

_client = None
SPENT = [0.0]  # USD for calls that were not in the cache, in this process (list price, half for batches)
PENDING: dict | None = None  # set by a batched run (run.py --batch): the requests not in the cache are collected here


class Pending(Exception):
    """A request not in the cache while a batched run collects them: its conversation waits for the next batch."""


def _load_key(name: str = "ANTHROPIC_API_KEY") -> None:
    """The key `name` from the git-ignored .env (`KEY = value` lines), unless it is already set."""
    path = ROOT / ".env"
    if name in os.environ or not path.exists():
        return
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == name:
            os.environ[name] = value.strip()


def _claude() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _load_key()
        _client = anthropic.Anthropic(timeout=180, max_retries=3)  # a hung call fails after 3 minutes and retries
    return _client


def call(**request) -> dict:
    """One Messages API call, returned as a dict. Set AA_GATE_OFFLINE to raise instead of calling the API. In a
    batched run, a request not in the cache is collected in PENDING and raises Pending instead."""
    request = {"max_tokens": MAX_TOKENS, **request}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    path = CACHE / "llm" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    if os.environ.get("AA_GATE_OFFLINE"):
        raise RuntimeError(f"request {key} is not in the cache")
    if PENDING is not None:
        PENDING[key] = request
        raise Pending(key)
    response = _claude().messages.create(**request).to_dict()
    if response["stop_reason"] == "max_tokens":
        raise RuntimeError(f"truncated answer for request {key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(response))
    SPENT[0] += price(response)
    return response


def batch_claude(requests: dict[str, dict]) -> None:
    """Send requests (by cache key) through Anthropic's Message Batches API at half the price, wait for the batch and
    cache each answer as `call` would (decision 53). As with OpenAI, an interrupted call takes up the same batch; a
    truncated answer raises; requests that did not succeed are reported, and the next wave sends them anew."""
    client = _claude()
    sent = CACHE / "llm" / f"claude-batch-{hashlib.sha256(''.join(sorted(requests)).encode()).hexdigest()}.txt"
    if sent.exists():
        batch = client.messages.batches.retrieve(sent.read_text().strip())
    else:
        batch = client.messages.batches.create(requests=[{"custom_id": k, "params": r} for k, r in requests.items()])
        sent.parent.mkdir(parents=True, exist_ok=True)
        sent.write_text(batch.id)
    while batch.processing_status != "ended":
        time.sleep(30)
        batch = client.messages.batches.retrieve(batch.id)
    failed = []
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            failed.append(f"{result.custom_id}: {result.result.type}")
            continue
        response = result.result.message.to_dict()
        if response["stop_reason"] == "max_tokens":
            raise RuntimeError(f"truncated answer for request {result.custom_id}")
        (CACHE / "llm" / f"{result.custom_id}.json").write_text(json.dumps(response))
        SPENT[0] += 0.5 * price(response)
    sent.unlink()
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(requests)} requests in batch {batch.id} did not succeed, for "
                           f"example {failed[:3]}; run again to send them anew")


def batch_openai(bodies: list[dict]) -> list[dict]:
    """Chat Completions calls through OpenAI's Batch API, at half the price (decision 46): the requests not in the
    cache go out as one batch, which is waited for, checked for truncation and cached like a single call. The batch's
    id is kept until it ends, so a call that is interrupted takes up the same batch instead of paying twice. A batch
    that hangs can be cancelled at OpenAI: its finished answers are kept, and the next call sends only the rest.
    Returns the responses in the order of `bodies`."""
    keys = [hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest() for b in bodies]
    todo = {k: b for k, b in zip(keys, bodies) if not (CACHE / "llm" / f"openai-{k}.json").exists()}
    if todo:
        if os.environ.get("AA_GATE_OFFLINE"):
            raise RuntimeError(f"{len(todo)} OpenAI requests are not in the cache")
        _load_key("OPENAI_API_KEY")
        client = openai.OpenAI(timeout=600, max_retries=3)
        sent = CACHE / "llm" / f"openai-batch-{hashlib.sha256(''.join(sorted(todo)).encode()).hexdigest()}.txt"
        if sent.exists():
            batch = client.batches.retrieve(sent.read_text().strip())
        else:
            lines = "\n".join(json.dumps({"custom_id": k, "method": "POST", "url": "/v1/chat/completions", "body": b})
                              for k, b in todo.items())
            upload = client.files.create(file=("requests.jsonl", lines.encode()), purpose="batch")
            batch = client.batches.create(input_file_id=upload.id, endpoint="/v1/chat/completions",
                                          completion_window="24h")
            sent.parent.mkdir(parents=True, exist_ok=True)
            sent.write_text(batch.id)
        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            time.sleep(30)
            batch = client.batches.retrieve(batch.id)
        for line in client.files.content(batch.output_file_id).text.splitlines() if batch.output_file_id else []:
            r = json.loads(line)
            if (r.get("response") or {}).get("status_code") != 200:
                continue  # a failed request: reported below
            response = r["response"]["body"]
            if response["choices"][0]["finish_reason"] == "length":
                raise RuntimeError(f"truncated answer for OpenAI request {r['custom_id']}")
            (CACHE / "llm" / f"openai-{r['custom_id']}.json").write_text(json.dumps(response))
            inp, out = PRICE[todo[r["custom_id"]]["model"]]
            u = response["usage"]
            SPENT[0] += 0.5 * (u["prompt_tokens"] * inp + u["completion_tokens"] * out) / 1e6
        sent.unlink()
        failed = [k for k in todo if not (CACHE / "llm" / f"openai-{k}.json").exists()]
        if failed:  # failed requests are listed in the batch's error file, not in its output
            detail = client.files.content(batch.error_file_id).text[:500] if batch.error_file_id else batch.errors
            raise RuntimeError(f"{len(failed)} of {len(todo)} OpenAI requests not answered in batch {batch.id} "
                               f"({batch.status}): {detail}. Run again to send them anew.")
    return [json.loads((CACHE / "llm" / f"openai-{k}.json").read_text()) for k in keys]


def text(response: dict) -> str:
    return "".join(b["text"] for b in response["content"] if b["type"] == "text").strip()


def price(response: dict) -> float:
    """List price in USD of one response (input including cache writes and reads at the input rate)."""
    inp, out = PRICE[response["model"]]
    u = response["usage"]
    tokens_in = u["input_tokens"] + (u.get("cache_creation_input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
    return (tokens_in * inp + u["output_tokens"] * out) / 1e6
