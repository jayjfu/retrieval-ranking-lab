"""Reference scores from the TREC 2025 RAG overview (arXiv:2603.09891), 22 NIST-judged topics."""

# Table 5, R task, NIST qrels: nDCG@30, nDCG@100, Recall@100
RETRIEVAL_NIST = {
    "[overview] best: 4method_merge": {"ndcg@30": 0.6934, "ndcg@100": 0.6134, "recall@100": 0.2331},
    "[overview] rag25_test_rankqwen3": {"ndcg@30": 0.6462, "ndcg@100": 0.5844, "recall@100": 0.2274},
    "[overview] splade-v3-arctic-l": {"ndcg@30": 0.6009, "ndcg@100": 0.5683, "recall@100": 0.2274},
    "[overview] rag25_test_splade-v3": {"ndcg@30": 0.5957, "ndcg@100": 0.5387, "recall@100": 0.2103},
    "[overview] rag25_test_arctic-l": {"ndcg@30": 0.4991, "ndcg@100": 0.4183, "recall@100": 0.1592},
    "[overview] anserini_bm25_top100": {"ndcg@30": 0.3468, "ndcg@100": 0.2900, "recall@100": 0.1107},
}

# Table 8, post-edited nuggets + automatic (GPT-4.1) assignment: strict vital score, sub-narrative coverage
NUGGETS_AUTO_ASSIGNMENT = {
    "[overview] best: LAS-agentic-RAG-agent": {"strict_vital": 0.50, "subnarrative_coverage": 0.80},
    "[overview] rag25_qwen3_50_ag": {"strict_vital": 0.47, "subnarrative_coverage": 0.80},
    "[overview] rag25_qwen3_20_ag": {"strict_vital": 0.45, "subnarrative_coverage": 0.78},
    "[overview] rag25_test_qwen3_20": {"strict_vital": 0.45, "subnarrative_coverage": 0.79},
}

# Table 12, support with GPT-OSS-120B assessor, 17 topics: weighted precision, weighted recall
SUPPORT_AUTO = {
    "[overview] best: no-llm-refined": {"support_precision": 0.9522, "support_recall": 0.9522},
    "[overview] rag25_qwen3_20_ag": {"support_precision": 0.5675, "support_recall": 0.5079},
    "[overview] rag25_qwen3_50_ag": {"support_precision": 0.4684, "support_recall": 0.3730},
}
