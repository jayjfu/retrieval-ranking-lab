"""Text generation backends for RAG and LLM judging.

Spec: "vllm:<hf model id>" or "anthropic:<claude model id>".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

Messages = list[dict[str, str]]


class Generator(ABC):
    name: str

    @abstractmethod
    def generate(self, conversations: list[Messages]) -> list[str]: ...


class VLLMGenerator(Generator):
    """Batched local generation, greedy by default."""

    def __init__(self, model_name: str, max_tokens: int = 512, temperature: float = 0.0, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.85, tensor_parallel_size: int = 1):
        from vllm import LLM, SamplingParams

        self.name = model_name
        self.engine = LLM(model=model_name, max_model_len=max_model_len,
                          gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size,
                          limit_mm_per_prompt={"image": 0, "video": 0})  # text only, skips Qwen3.5 vision encoder
        self.sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    def generate(self, conversations: list[Messages]) -> list[str]:
        # enable_thinking=False only affects Qwen3 / Qwen3.5 templates
        outputs = self.engine.chat(conversations, self.sampling_params, use_tqdm=True,
                                   chat_template_kwargs={"enable_thinking": False})
        return [output.outputs[0].text.strip() for output in outputs]

    def shutdown(self):
        import gc

        import torch

        del self.engine
        gc.collect()
        torch.cuda.empty_cache()


class AnthropicGenerator(Generator):
    """Claude API. Refused requests are retried server-side on a fallback model."""

    def __init__(self, model_name: str = "claude-opus-5-5", max_tokens: int = 4096, effort: str = "low",
                 concurrency: int = 8):
        import anthropic

        self.name = model_name
        self.client = anthropic.Anthropic()
        self.max_tokens, self.effort, self.concurrency = max_tokens, effort, concurrency

    def generate_one(self, conversation: Messages) -> str:
        system = "\n\n".join(message["content"] for message in conversation if message["role"] == "system")
        messages = [message for message in conversation if message["role"] != "system"]
        extra = {"system": system} if system else {}
        response = self.client.beta.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            messages=messages,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **extra,
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def generate(self, conversations: list[Messages]) -> list[str]:
        with ThreadPoolExecutor(self.concurrency) as executor:
            return list(executor.map(self.generate_one, conversations))


def make_generator(spec: str, **options) -> Generator:
    backend, _, model_name = spec.partition(":")
    if backend == "vllm":
        return VLLMGenerator(model_name, **options)
    if backend == "anthropic":
        options.pop("max_model_len", None)  # vLLM only
        return AnthropicGenerator(model_name or "claude-opus-5-5", **options)
    raise ValueError(f"unknown generator backend {backend!r}, use vllm:<model> or anthropic:<model>")
