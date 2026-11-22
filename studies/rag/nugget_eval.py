"""Nugget and support evaluation, following the TREC 2025 RAG overview.

Nuggets: post-edited RAG25 nuggets, listwise assignment (support / partial_support / not_support),
AutoNuggetizer prompt, batches of 10. The official assessor is GPT-4.1; ours is a Qwen3.5 judge.
Support: first citation of each sentence, full / partial / no support, weights 1 / 0.5 / 0.
The official assessor is GPT-OSS-120B on 17 topics; we score all judged topics.
"""

from __future__ import annotations

import ast
import re
import statistics

NUGGET_SYSTEM = ("You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of atomic nuggets "
                 "based on if they are captured by a given passage.")
NUGGET_TEMPLATE = (
    "Based on the query and passage, label each of the {count} nuggets either as support, partial_support, or "
    "not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as "
    "support. A nugget that is partially captured in the passage should be labeled as partial_support. If the nugget "
    "is not captured at all, label it as not_support. Return the list of labels in a Pythonic list format "
    "(type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for "
    "each nugget.\n\nSearch Query: {query}\nPassage: {passage}\nNugget List: {nuggets}\n\n"
    "Only return the list of labels (List[str]). Do not explain.\nLabels:"
)
NUGGET_LABELS = ("support", "partial_support", "not_support")

SUPPORT_SYSTEM = "You judge whether a sentence is supported by a passage."
SUPPORT_TEMPLATE = (
    "Passage: {passage}\n\nSentence: {sentence}\n\n"
    "Is the sentence fully supported, partially supported, or not supported by the passage? "
    "Reply with exactly one of: full, partial, none."
)
SUPPORT_WEIGHTS = {"full": 1.0, "partial": 0.5, "none": 0.0}
NUGGET_BATCH_SIZE = 10


def nugget_prompts(query: str, answer: str, nuggets: list[dict]) -> list[list[dict[str, str]]]:
    prompts = []
    for start in range(0, len(nuggets), NUGGET_BATCH_SIZE):
        batch = [nugget["text"] for nugget in nuggets[start : start + NUGGET_BATCH_SIZE]]
        content = NUGGET_TEMPLATE.format(count=len(batch), query=query, passage=answer, nuggets=batch)
        prompts.append([{"role": "system", "content": NUGGET_SYSTEM}, {"role": "user", "content": content}])
    return prompts


def parse_nugget_labels(text: str, count: int) -> list[str]:
    """Missing or unparsable labels count as not_support."""
    match = re.search(r"\[.*?\]", text, re.S)
    labels = []
    if match:
        try:
            labels = [str(label).strip().lower() for label in ast.literal_eval(match.group())]
        except (ValueError, SyntaxError):
            labels = re.findall(r"partial_support|not_support|support", match.group())
    labels = [label if label in NUGGET_LABELS else "not_support" for label in labels[:count]]
    return labels + ["not_support"] * (count - len(labels))


def support_prompt(sentence: str, passage: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SUPPORT_SYSTEM},
            {"role": "user", "content": SUPPORT_TEMPLATE.format(passage=passage, sentence=sentence)}]


def parse_support(text: str) -> str:
    lowered = text.lower()
    for label in ("partial", "full", "none"):
        if label in lowered:
            return label
    return "none"


def normalize_subnarrative(name: str) -> str:
    return name.strip().strip('"').strip().lower()


def nugget_scores(nuggets: list[dict], labels: list[str]) -> dict[str, float]:
    """strict_vital: vital nuggets fully supported. vital: partial counts 0.5.
    subnarrative_coverage: sub-narratives with at least one fully supported mapped nugget."""
    vital = [label for nugget, label in zip(nuggets, labels) if nugget["importance"] == "vital"]
    covered: dict[str, bool] = {}
    for nugget, label in zip(nuggets, labels):
        key = normalize_subnarrative(nugget.get("mapped_sub_narrative") or "")
        if key:
            covered[key] = covered.get(key, False) or label == "support"
    return {
        "strict_vital": sum(label == "support" for label in vital) / len(vital) if vital else float("nan"),
        "vital": (sum({"support": 1.0, "partial_support": 0.5}.get(label, 0.0) for label in vital) / len(vital)
                  if vital else float("nan")),
        "subnarrative_coverage": sum(covered.values()) / len(covered) if covered else float("nan"),
    }


def support_scores(sentence_labels: list[str | None]) -> dict[str, float]:
    """sentence_labels: support label of each sentence's first citation, None if uncited."""
    if not sentence_labels:
        return {"support_precision": 0.0, "support_recall": 0.0}
    cited = [SUPPORT_WEIGHTS[label] for label in sentence_labels if label is not None]
    return {"support_precision": statistics.fmean(cited) if cited else 0.0,
            "support_recall": sum(cited) / len(sentence_labels)}
