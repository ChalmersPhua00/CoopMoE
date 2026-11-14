#!/usr/bin/env python3
"""Labeled requests from standard public datasets: one JSON file of (request, category) pairs.

    python make_requests.py --out data/requests.json                 # 200 per category
    python make_requests.py --out data/requests.json --per-category 500 --categories code,math,general

Each category draws from one or more Hugging Face datasets (all ungated). Every dataset is
streamed and shuffled with a fixed seed, so a rerun writes the same file. Each record keeps the
dataset's own answer as "reference": the benchmark feeds it back as conversation history, and
`coopmoe trace-hf` traces it as the reply.

    [{"id": "code-0000", "category": "code", "source": "ise-uiuc/Magicoder-OSS-Instruct-75K",
      "request": "...", "reference": "..."},
     ...]

The categories are finer than the two boxes on purpose. router.box_categories in the run config
decides which box serves which categories, so the partition can change without regenerating
this file. `coopmoe split` shows which partition separates the experts best.

Needs `pip install datasets` (in requirements.txt).
"""
from __future__ import annotations

import argparse
import ast
import html
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------- cleaning
def _join(*parts: Optional[str]) -> str:
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>", "\n", text or "")
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def detok(text: str) -> str:
    """Undo WritingPrompts' tokenized text: "do n't" -> "don't", "end ." -> "end.",
    ``quoted'' -> "quoted", spaced <newline> markers -> line breaks."""
    text = text.replace("<newline>", "\n")
    text = re.sub(r"``\s*", '"', text)
    text = re.sub(r"\s*''", '"', text)
    text = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)
    text = re.sub(r" ([,.!?;:)])", r"\1", text)
    text = re.sub(r"([(]) ", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r" *\n *", "\n", text).strip()


def gsm8k(r: dict) -> tuple[str, str]:
    answer = re.sub(r"<<[^>]*>>", "", r["answer"])           # calculator annotations
    answer = re.sub(r"\n?#### *(.+)$", r"\nThe answer is \1.", answer.strip())
    return r["question"], answer


def law_stackexchange(r: dict) -> tuple[str, str]:
    answers = r.get("answers") or []
    if isinstance(answers, str):
        answers = ast.literal_eval(answers)
    best = max(answers, key=lambda a: int(a.get("score") or 0), default=None)
    return _join(r["question_title"], strip_html(r["question_body"])), strip_html(best["body"]) if best else ""


def writing_prompt(r: dict) -> tuple[str, str]:
    prompt = detok(re.sub(r"^\s*(\[\s*\w+\s*\]\s*)+", "", r["prompt"]))
    return f"Write a short story based on this prompt: {prompt}", detok(r["story"])


# ---------------------------------------------------------------------- sources
# The Hub's Parquet conversion of a dataset. camel-ai ships a zip of 20,000 small JSON files,
# which streams one HTTP request per file; its Parquet conversion streams in seconds.
PARQUET = "refs/convert/parquet"


@dataclass
class Source:
    repo: str
    extract: Callable[[dict], tuple[str, str]]   # row -> (request, reference)
    config: Optional[str] = None
    split: str = "train"
    keep: Optional[Callable[[dict], bool]] = None
    revision: Optional[str] = None


def _dolly(*kinds: str) -> Source:
    return Source("databricks/databricks-dolly-15k", lambda r: (_join(r["instruction"], r["context"]), r["response"]),
                  keep=lambda r: r["category"] in kinds)


SOURCES: dict[str, list[Source]] = {
    "code": [
        Source("ise-uiuc/Magicoder-OSS-Instruct-75K", lambda r: (r["problem"], r["solution"])),
        Source("nickrosh/Evol-Instruct-Code-80k-v1", lambda r: (r["instruction"], r["output"])),
    ],
    "math": [
        Source("openai/gsm8k", gsm8k, config="main"),
        Source("meta-math/MetaMathQA", lambda r: (r["query"], r["response"])),
    ],
    "science": [
        Source(f"camel-ai/{s}", lambda r: (r["message_1"], r["message_2"]), revision=PARQUET)
        for s in ("physics", "chemistry", "biology")
    ],
    "medicine": [
        Source("lavita/ChatDoctor-HealthCareMagic-100k", lambda r: (r["input"], r["output"])),
        Source("keivalya/MedQuad-MedicalQnADataset", lambda r: (r["Question"], r["Answer"])),
    ],
    "finance": [
        Source("gbharti/finance-alpaca", lambda r: (_join(r["instruction"], r["input"]), r["output"])),
        Source("FinGPT/fingpt-fiqa_qa", lambda r: (r["input"], r["output"])),
    ],
    "law": [
        Source("ymoslem/Law-StackExchange", law_stackexchange),
    ],
    "writing": [
        Source("euclaise/writingprompts", writing_prompt),
        _dolly("creative_writing"),
    ],
    "general": [
        _dolly("open_qa", "general_qa", "brainstorming"),
    ],
    # A language, not a topic. Qwen models often split their experts by language more sharply
    # than by subject, so `coopmoe split` gets the chance to show whether that is the better axis.
    "chinese": [
        Source("shibing624/alpaca-zh", lambda r: (_join(r["instruction"], r["input"]), r["output"])),
        Source("silk-road/alpaca-data-gpt4-chinese", lambda r: (_join(r["instruction_zh"], r["input_zh"]), r["output_zh"])),
    ],
}


# ---------------------------------------------------------------------- sampling
def sample_source(src: Source, n: int, seed: int, min_chars: int, max_chars: int, seen: set, log) -> list[dict]:
    """n cleaned records from one dataset: shuffled stream, requests longer than max_chars
    skipped (cutting them would break the question), references cut to max_chars."""
    from datasets import load_dataset

    ds = load_dataset(src.repo, src.config, split=src.split, streaming=True, revision=src.revision)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    out, scanned = [], 0
    for row in ds:
        scanned += 1
        if scanned > 200 * n + 10_000:
            break
        if src.keep and not src.keep(row):
            continue
        try:
            request, reference = src.extract(row)
        except (KeyError, ValueError, SyntaxError, TypeError):
            continue
        request, reference = (request or "").strip(), (reference or "").strip()
        if not (min_chars <= len(request) <= max_chars) or len(reference) < min_chars or request in seen:
            continue
        seen.add(request)
        out.append({"source": src.repo, "request": request, "reference": reference[:max_chars]})
        if len(out) >= n:
            break
    if len(out) < n:
        log(f"  {src.repo}: only {len(out)} of {n} usable records")
    return out


def sample_category(category: str, n: int, seed: int, min_chars: int, max_chars: int, log) -> list[dict]:
    """n records split evenly over the category's sources; a source that fails or runs short
    leaves its share to the ones after it."""
    sources = SOURCES[category]
    out: list[dict] = []
    seen: set = set()
    for i, src in enumerate(sources):
        quota = (n - len(out)) // (len(sources) - i) + ((n - len(out)) % (len(sources) - i) > 0)
        t0 = time.perf_counter()
        try:
            got = sample_source(src, quota, seed, min_chars, max_chars, seen, log)
            out += got
            log(f"  {src.repo}: {len(got)} records ({time.perf_counter() - t0:.0f} s)")
        except Exception as exc:  # network, renamed dataset, changed schema: keep the other sources
            log(f"  {src.repo}: skipped ({type(exc).__name__}: {exc})")
    return out


def write_requests(path: str, records: list[dict]) -> None:
    """A JSON array with one record per line: valid JSON that still greps and diffs well."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n]\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/requests.json")
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--categories", default=",".join(SOURCES), help=f"comma-separated subset of: {', '.join(SOURCES)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=16)
    ap.add_argument("--max-chars", type=int, default=3000)
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = [c for c in categories if c not in SOURCES]
    if unknown:
        raise SystemExit(f"unknown categories {unknown}; known: {', '.join(SOURCES)}")
    records = []
    for category in categories:
        log(f"{category}: {', '.join(s.repo for s in SOURCES[category])}")
        got = sample_category(category, args.per_category, args.seed, args.min_chars, args.max_chars, log)
        records += [dict(id=f"{category}-{i:04d}", category=category, **r) for i, r in enumerate(got)]
    write_requests(args.out, records)

    print(f"{args.out}: {len(records)} requests")
    print(f"{'category':<10} {'requests':>8} {'median request chars':>21} {'median reference chars':>23}  sources")
    for category in categories:
        rs = [r for r in records if r["category"] == category]
        if not rs:
            print(f"{category:<10} {0:>8}")
            continue
        per_src = {}
        for r in rs:
            per_src[r["source"]] = per_src.get(r["source"], 0) + 1
        print(f"{category:<10} {len(rs):>8} {statistics.median(len(r['request']) for r in rs):>21.0f} "
              f"{statistics.median(len(r['reference']) for r in rs):>23.0f}  "
              + ", ".join(f"{s} {k}" for s, k in per_src.items()))


if __name__ == "__main__":
    main()
