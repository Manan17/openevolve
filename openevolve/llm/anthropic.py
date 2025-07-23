"""
Anthropic API interface for LLMs
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import anthropic
import httpx

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)

class AnthropicLLM(LLMInterface):
    """LLM interface using Anthropic's Claude API"""

    def __init__(self, model_cfg: Optional[dict] = None):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_key = model_cfg.api_key
        self.api_base = model_cfg.api_base or "https://api.anthropic.com"
        self.random_seed = getattr(model_cfg, "random_seed", None)

        # Set up API client
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()
        if self.model not in logger._initialized_models:
            logger.info(f"Initialized Anthropic LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        # Anthropic expects a single user message and optional system prompt
        # We'll concatenate all user messages for simplicity
        user_content = "\n".join([m["content"] for m in messages if m["role"] == "user"])
        # Anthropic's API: system prompt is passed as 'system', user prompt as 'messages'
        params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "system": system_message,
            "messages": [
                {"role": "user", "content": user_content}
            ],
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                # Check for HTTP 529 or 429
                status_code = None
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    status_code = e.response.status_code
                elif isinstance(e, httpx.HTTPStatusError):
                    status_code = e.response.status_code
                # Fallback: check error string
                is_rate_limited = (
                    status_code in (429, 529)
                    or "529" in str(e)
                    or "429" in str(e)
                )
                if is_rate_limited:
                    wait_time = max(retry_delay * (2 ** attempt), 10)  # Exponential backoff, min 10s
                    logger.warning(
                        f"Rate limit (HTTP {status_code}) on attempt {attempt + 1}/{retries + 1}: {str(e)}. "
                        f"Waiting {wait_time:.1f}s before retry..."
                    )
                    await asyncio.sleep(wait_time)
                elif attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        loop = asyncio.get_event_loop()
        # anthropic SDK: client.messages.create(**params)
        response = await loop.run_in_executor(
            None, lambda: self.client.messages.create(**params)
        )
        logger.debug(f"Anthropic API parameters: {params}")
        logger.debug(f"Anthropic API response: {response.content}")
        # response.content is a list of content blocks; join them if needed
        if isinstance(response.content, list):
            return "".join([c.text for c in response.content if hasattr(c, "text")])
        return str(response.content) 