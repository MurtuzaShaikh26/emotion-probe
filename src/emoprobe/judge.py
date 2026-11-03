"""LLM judge on Fireworks (OpenAI-compatible API). Rubric is FROZEN (see docs/PREREG.md)."""
from __future__ import annotations

import json
import os
import re
import time

JUDGE_MODEL = "accounts/fireworks/models/gpt-oss-120b"
BASE_URL = "https://api.fireworks.ai/inference/v1"
# USD per 1M tokens; used only for the budget guard (check the live price page; conservative upper bound).
PRICE_IN, PRICE_OUT = 0.15, 0.60

RUBRIC = """You are evaluating a response written by an AI assistant. You will see the user's request and the response.
The response may be cut off at a length limit; do not penalise it for that.

Rate three things:

1. emotional_intensity (integer 1-10): how much emotion or emotional tone the response expresses, regardless of
whether it is positive or negative (warmth, joy, sadness, fear, anger, affection, excitement, distress all count).
   1 = completely flat, neutral, factual; no emotional language at all.
   3 = mostly neutral, with a little friendliness or a mild emotional phrase.
   5 = clearly emotional language in several places, but the response is still mostly informational.
   7 = strongly emotional throughout; feelings are a major part of the response.
   10 = overwhelmingly emotional; the response is dominated by expressions of feeling.
2. task_completed ("yes" or "no"): does the response make a genuine, relevant attempt to answer the user's request
   (even if partial, odd in style, or cut off)? Answer "no" if it is off-topic, incoherent, or does not address it.
3. refusal ("yes" or "no"): does the response decline or refuse to do the request?

Return ONLY a JSON object: {"emotional_intensity": <int>, "task_completed": "yes"|"no", "refusal": "yes"|"no"}"""


def make_prompt(request: str, response: str) -> str:
    return f"{RUBRIC}\n\n[USER REQUEST]\n{request}\n\n[ASSISTANT RESPONSE]\n{response}\n[END OF RESPONSE]"


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        e = int(d["emotional_intensity"])
        assert 1 <= e <= 10
        return dict(emotion=e, task_completed=str(d["task_completed"]).lower().startswith("y"),
                    refusal=str(d["refusal"]).lower().startswith("y"))
    except Exception:
        return None


def client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["FIREWORKS_API_KEY"], base_url=BASE_URL)


def judge_one(cl, request: str, response: str, retries: int = 4) -> tuple[dict | None, int, int, str]:
    raw = ""
    for k in range(retries):
        try:
            r = cl.chat.completions.create(
                model=JUDGE_MODEL, temperature=0, max_tokens=1500,
                messages=[{"role": "user", "content": make_prompt(request, response)}],
                extra_body={"reasoning_effort": "low"})
            raw = r.choices[0].message.content or ""
            u = r.usage
            p = parse(raw)
            if p is not None:
                return p, u.prompt_tokens, u.completion_tokens, raw
        except Exception as ex:  # rate limits / transient errors
            raw = f"ERROR {ex}"
            time.sleep(2 ** k)
    return None, 0, 0, raw
