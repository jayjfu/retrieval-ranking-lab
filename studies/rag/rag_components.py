"""Prompts, citation parsing and TREC RAG 2025 export."""

from __future__ import annotations

import json
import re
from pathlib import Path

SYSTEM_PROMPT = (
    "You write grounded answers using only the numbered passages provided. "
    "After every sentence, cite the passage(s) that support it in square brackets, e.g. [1] or [2][3]."
)
USER_TEMPLATE = (
    "Passages:\n{context}\n\nInformation need: {query}\n\n"
    "Write a well-organised answer of at most 400 words that covers every part of the information need. "
    "Every sentence must carry at least one citation. Use plain sentences, no headings or lists."
)

CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def build_messages(query: str, passages: list[str]) -> list[dict[str, str]]:
    context = "\n\n".join(f"[{number}] {passage}" for number, passage in enumerate(passages, start=1))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(context=context, query=query)}]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", CITATION_PATTERN.sub("", text)).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


def parse_cited_sentences(answer: str, num_passages: int) -> list[dict]:
    """Sentences with 0-based cited passage indices. Out-of-range citations are dropped."""
    sentences = []
    for raw_sentence in SENTENCE_BOUNDARY.split(re.sub(r"\s*\n+\s*", " ", answer.strip())):
        citations = []
        for match in CITATION_PATTERN.finditer(raw_sentence):
            for number in match.group(1).split(","):
                passage_index = int(number) - 1
                if 0 <= passage_index < num_passages and passage_index not in citations:
                    citations.append(passage_index)
        text = clean_text(raw_sentence)
        if text:
            sentences.append({"text": text, "citations": citations})
        elif sentences:  # citation-only fragment belongs to the previous sentence
            sentences[-1]["citations"] += [c for c in citations if c not in sentences[-1]["citations"]]
    return sentences


def truncate_to_words(sentences: list[dict], max_words: int = 400) -> list[dict]:
    kept, total = [], 0
    for sentence in sentences:
        total += len(sentence["text"].split())
        if total > max_words:
            break
        kept.append(sentence)
    return kept


def answer_text(sentences: list[dict]) -> str:
    return " ".join(sentence["text"] for sentence in sentences)


def to_trec_rag_2025(record: dict, team_id: str, run_id: str, run_type: str = "automatic") -> dict:
    """AG/RAG task format, citations index into references."""
    return {
        "metadata": {"team_id": team_id, "run_id": run_id, "type": run_type,
                     "narrative_id": record["query_id"], "narrative": record["query"]},
        "references": record["context_ids"][:100],
        "answer": [{"text": sentence["text"], "citations": [c for c in sentence["citations"] if c < 100]}
                   for sentence in record["sentences"]],
    }


def export_trec_rag(records: list[dict], path: str | Path, team_id: str, run_id: str) -> None:
    with open(path, "w") as file:
        for record in records:
            file.write(json.dumps(to_trec_rag_2025(record, team_id, run_id), ensure_ascii=False) + "\n")
