from __future__ import annotations

import torch

from retrieval_ranking_lab.rerank.base import Reranker

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER, batch_size: int = 128, max_length: int = 512,
                 device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False).tolist()
