"""Build every dataset as a flat list of example dicts, split by topic group.

Example fields
--------------
id         unique string
dataset    d1 | d1neg | d1nosys | d2 | d3_<src>_<fmt> | steer | calib
split      train | test | eval
topic      topic group (split unit: no topic is in both train and test)
item       clustering unit for bootstrap (a task or a D2 pair); both labels of an item share it
label      1 = emotional/empathetic, 0 = neutral/cold, -1 = unlabelled
system     system prompt text or None (None = model's default chat template behaviour)
user       user message text
fmt        chat | plain   (plain = no chat template, "User:/Assistant:" completion style)
para       system-prompt paraphrase id 0-9 (0-4 empathetic, 5-9 cold) or -1
emotion    grief | joy | fear | anger | neutral | ""
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from . import texts

SEED = 0
D1_TEST_TOPICS_N = 5    # of 15 D1 topic groups
D2_TEST_TOPICS_N = 9    # of 30 D2 topic groups
FORMATS = ("mcq", "json", "plain")


def _split_topics(topics: list[str], n_test: int, seed: int) -> set[str]:
    rng = random.Random(seed)
    t = sorted(topics)
    rng.shuffle(t)
    return set(t[:n_test])


def _pretty(topic: str) -> str:
    return topic.replace("_", " ")


def wrap_mcq(content: str, topic: str, all_topics: list[str], rng: random.Random) -> str:
    distractors = rng.sample([t for t in all_topics if t != topic], 3)
    opts = distractors + [topic]
    rng.shuffle(opts)
    lines = "\n".join(f"{l}) {_pretty(o)}" for l, o in zip("ABCD", opts))
    return (
        "Read the message below and answer the multiple-choice question.\n\n"
        f"Message: {content}\n\n"
        f"Question: Which subject is this message mainly about?\n{lines}\n\n"
        "Answer with a single letter."
    )


def wrap_json(content: str) -> str:
    return content + '\n\nRespond only with a JSON object of the form {"answer": "..."}.'


def build_all(seed: int = SEED) -> list[dict]:
    ex: list[dict] = []
    rng = random.Random(seed)

    # ---------------- D1 ----------------
    d1_topics = list(texts.D1_TASKS)
    d1_test = _split_topics(d1_topics, D1_TEST_TOPICS_N, seed)
    k = 0
    d1_items = []  # (item, topic, task, split, para_idx)
    for topic in d1_topics:
        for j, task in enumerate(texts.D1_TASKS[topic]):
            split = "test" if topic in d1_test else "train"
            item = f"d1:{topic}:{j}"
            p = k % 5  # rotate paraphrases across tasks
            d1_items.append((item, topic, task, split, p))
            k += 1
    for item, topic, task, split, p in d1_items:
        for label, prompts, off in ((1, texts.EMPATHETIC_PROMPTS, 0), (0, texts.COLD_PROMPTS, 5)):
            ex.append(dict(id=f"{item}:{label}", dataset="d1", split=split, topic=topic, item=item,
                           label=label, system=prompts[p], user=task, fmt="chat", para=p + off, emotion=""))
        # no system prompt (for the instruction-presence direction)
        ex.append(dict(id=f"{item}:nosys", dataset="d1nosys", split=split, topic=topic, item=item,
                       label=-1, system=None, user=task, fmt="chat", para=-1, emotion=""))
        # negation control: test split only
        if split == "test":
            for n, (sysp, label) in enumerate(texts.NEGATED_PROMPTS):
                ex.append(dict(id=f"{item}:neg{n}", dataset="d1neg", split=split, topic=topic, item=item,
                               label=label, system=sysp, user=task, fmt="chat", para=-1, emotion=""))
        # D3 formats (train split kept for estimating the format subspace)
        for label, prompts, off in ((1, texts.EMPATHETIC_PROMPTS, 0), (0, texts.COLD_PROMPTS, 5)):
            for fmt in FORMATS:
                if fmt == "mcq":
                    user, f = wrap_mcq(task, topic, d1_topics, rng), "chat"
                elif fmt == "json":
                    user, f = wrap_json(task), "chat"
                else:
                    user, f = task, "plain"
                ex.append(dict(id=f"{item}:{label}:{fmt}", dataset=f"d3_d1_{fmt}", split=split, topic=topic,
                               item=item, label=label, system=prompts[p], user=user, fmt=f, para=p + off,
                               emotion=""))

    # ---------------- D2 ----------------
    d2_topics = list(texts.D2_PAIRS)
    d2_test = _split_topics(d2_topics, D2_TEST_TOPICS_N, seed + 1)
    for topic in d2_topics:
        split = "test" if topic in d2_test else "train"
        for j, (emo, emo_msg, neu_msg) in enumerate(texts.D2_PAIRS[topic]):
            item = f"d2:{topic}:{j}"
            for label, msg, e in ((1, emo_msg, emo), (0, neu_msg, "neutral")):
                ex.append(dict(id=f"{item}:{label}", dataset="d2", split=split, topic=topic, item=item,
                               label=label, system=None, user=msg, fmt="chat", para=-1, emotion=e))
                if split == "test":
                    for fmt in FORMATS:
                        if fmt == "mcq":
                            user, f = wrap_mcq(msg, topic, d2_topics, rng), "chat"
                        elif fmt == "json":
                            user, f = wrap_json(msg), "chat"
                        else:
                            user, f = msg, "plain"
                        ex.append(dict(id=f"{item}:{label}:{fmt}", dataset=f"d3_d2_{fmt}", split=split,
                                       topic=topic, item=item, label=label, system=None, user=user, fmt=f,
                                       para=-1, emotion=e))

    # ---------------- steering eval + calibration prompts ----------------
    for topic, prompts in texts.STEER_PROMPTS.items():
        for j, p in enumerate(prompts):
            ex.append(dict(id=f"steer:{topic}:{j}", dataset="steer", split="eval", topic=topic,
                           item=f"steer:{topic}:{j}", label=-1, system=None, user=p, fmt="chat", para=-1,
                           emotion=""))
    # calibration: first task of each D1 TRAIN topic (never used for steering evaluation)
    for topic in d1_topics:
        if topic not in d1_test:
            ex.append(dict(id=f"calib:{topic}", dataset="calib", split="calib", topic=topic,
                           item=f"calib:{topic}", label=-1, system=None, user=texts.D1_TASKS[topic][0],
                           fmt="chat", para=-1, emotion=""))
    return ex


# ---------------------------------------------------------------------------
# Leakage and sanity checks (used by tests and scripts/build_datasets.py)
# ---------------------------------------------------------------------------
INSTRUCTION_VOCAB = re.compile(
    r"\b(warm\w*|empath\w*|compassion\w*|caring|kind|gentle|heartfelt|tender\w*|emotion\w*|feelings?|"
    r"robotic|cold|detached|mechanical\w*|clinical|impersonal|flat|terse|sentiment|loving|supportive)\b", re.I)


def check_leakage(ex: list[dict]) -> dict:
    """Raise AssertionError on any leakage; return a dict of summary stats."""
    report = {}
    for ds in ("d1", "d2"):
        rows = [e for e in ex if e["dataset"] == ds]
        tr_topics = {e["topic"] for e in rows if e["split"] == "train"}
        te_topics = {e["topic"] for e in rows if e["split"] == "test"}
        assert not (tr_topics & te_topics), f"{ds}: topic in both splits: {tr_topics & te_topics}"
        tr_text = {e["user"] for e in rows if e["split"] == "train"}
        te_text = {e["user"] for e in rows if e["split"] == "test"}
        assert not (tr_text & te_text), f"{ds}: identical user text in train and test"
        tr_items = {e["item"] for e in rows if e["split"] == "train"}
        te_items = {e["item"] for e in rows if e["split"] == "test"}
        assert not (tr_items & te_items)
        report[ds] = dict(n_train=sum(e["split"] == "train" for e in rows),
                          n_test=sum(e["split"] == "test" for e in rows),
                          train_topics=sorted(tr_topics), test_topics=sorted(te_topics))
    # every D3 / neg example's item must have the same split as the source item
    split_of = {e["item"]: e["split"] for e in ex if e["dataset"] in ("d1", "d2")}
    for e in ex:
        if e["dataset"].startswith("d3_") or e["dataset"] in ("d1neg", "d1nosys"):
            assert split_of[e["item"]] == e["split"], e["id"]
    # D3 D2 rows exist only for test items
    assert all(e["split"] == "test" for e in ex if e["dataset"].startswith("d3_d2"))
    # steering prompts: new text, new topics
    used_text = {e["user"] for e in ex if e["dataset"] in ("d1", "d2", "calib")}
    used_topics = {e["topic"] for e in ex if e["dataset"] in ("d1", "d2")}
    steer = [e for e in ex if e["dataset"] == "steer"]
    assert not ({e["user"] for e in steer} & used_text), "steering prompt reused"
    assert not ({e["topic"] for e in steer} & used_topics), "steering topic overlaps"
    assert len({e["user"] for e in steer}) == len(steer) == 60
    # calibration prompts come from D1 train only
    d1_train_text = {e["user"] for e in ex if e["dataset"] == "d1" and e["split"] == "train"}
    assert all(e["user"] in d1_train_text for e in ex if e["dataset"] == "calib")
    # unique ids
    ids = [e["id"] for e in ex]
    assert len(ids) == len(set(ids)), "duplicate ids"
    report["steer_n"] = len(steer)
    return report


def label_alignment_ok(ex: list[dict]) -> bool:
    """D1 labels must match the system-prompt class; D2 labels must match emotion."""
    emp, cold = set(texts.EMPATHETIC_PROMPTS), set(texts.COLD_PROMPTS)
    for e in ex:
        if e["dataset"] == "d1" or e["dataset"].startswith("d3_d1"):
            assert (e["system"] in emp) == (e["label"] == 1) and (e["system"] in cold) == (e["label"] == 0), e["id"]
            assert (e["para"] < 5) == (e["label"] == 1), e["id"]
        if e["dataset"] == "d2" or e["dataset"].startswith("d3_d2"):
            assert (e["emotion"] == "neutral") == (e["label"] == 0), e["id"]
    return True


def d2_stats(ex: list[dict]) -> dict:
    rows = [e for e in ex if e["dataset"] == "d2"]
    by_item: dict[str, dict] = {}
    for e in rows:
        by_item.setdefault(e["item"], {})[e["label"]] = e
    diffs, emo_len, neu_len = [], [], []
    for item, d in by_item.items():
        a, b = len(d[1]["user"].split()), len(d[0]["user"].split())
        emo_len.append(a); neu_len.append(b); diffs.append(a - b)
    vocab_hits = {lab: sum(bool(INSTRUCTION_VOCAB.search(e["user"])) for e in rows if e["label"] == lab)
                  for lab in (0, 1)}
    emotions = {}
    for e in rows:
        if e["label"] == 1:
            emotions[e["emotion"]] = emotions.get(e["emotion"], 0) + 1
    n = len(by_item)
    return dict(n_pairs=n, mean_words_emotional=sum(emo_len) / n, mean_words_neutral=sum(neu_len) / n,
                mean_pair_word_diff=sum(diffs) / n, max_abs_pair_word_diff=max(abs(x) for x in diffs),
                instruction_vocab_hits=vocab_hits, emotion_counts=emotions)


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
