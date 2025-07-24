"""
vLLM API interface for LLMs (Qwen, etc.)
"""

import requests
import httpx
import asyncio
from openevolve.llm.base import LLMInterface

class VLLMLLM(LLMInterface):
    def __init__(self, model_cfg):
        self.api_base = model_cfg.api_base.rstrip("/")
        self.model = model_cfg.name
        self.temperature = getattr(model_cfg, "temperature", 0.9)
        self.top_p = getattr(model_cfg, "top_p", 0.8)
        self.max_tokens = getattr(model_cfg, "max_tokens", 80)
        self.top_k = getattr(model_cfg, "top_k", 20)
        self.retries = getattr(model_cfg, "retries", 3)
        self.retry_delay = getattr(model_cfg, "retry_delay", 5)

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=None,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(self, system_message, messages, **kwargs):
        # Prepend system message if present
        prompt = ""
        if system_message:
            prompt += system_message + "\n"
        prompt += "\n".join([m["content"] for m in messages if m["role"] == "user"])
        payload = {
            "model": self.model,
            "prompt": [prompt],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_k": kwargs.get("top_k", self.top_k),
            "top_p": kwargs.get("top_p", self.top_p),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        url = f"{self.api_base}/completions"
        for attempt in range(self.retries + 1):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["text"]
            except Exception as e:
                if attempt < self.retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise 