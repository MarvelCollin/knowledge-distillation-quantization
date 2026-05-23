import json
import math
import time
from pathlib import Path

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError


_RESTRICTED_PROVIDERS = {"konektika"}


class TeacherModel:
    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str,
        max_tokens: int,
        temperature: float,
        top_logprobs: int,
        headers: dict = None,
        provider: str = "openrouter",
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            default_headers=headers or {},
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_logprobs = top_logprobs
        self.provider = provider

    _SYSTEM = "You are a coding assistant. Output ONLY a raw Python function definition. No explanation, no markdown, no triple backticks. Start directly with def."

    def _build_create_kwargs(self, messages: list) -> dict:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.provider not in _RESTRICTED_PROVIDERS:
            kwargs["temperature"] = self.temperature
        return kwargs

    def get_response_with_logprobs(self, prompt: str) -> dict:
        messages = [
            {"role": "system", "content": self._SYSTEM},
            {"role": "user", "content": prompt},
        ]
        delays = [10, 30, 60]
        for attempt, delay in enumerate([0] + delays):
            if delay:
                print(f"  Retry {attempt}/{len(delays)} after {delay}s...")
                time.sleep(delay)
            try:
                response = self.client.chat.completions.create(
                    **self._build_create_kwargs(messages)
                )
                return self._parse_response(response)
            except RateLimitError:
                if attempt == len(delays):
                    raise
                print(f"  Rate limited.")
            except (APIConnectionError, APITimeoutError) as exc:
                if attempt == len(delays):
                    raise
                print(f"  {type(exc).__name__}: {exc}")

    def _parse_response(self, response) -> dict:
        choice = response.choices[0]
        tokens = []
        logprobs_per_token = []

        if choice.logprobs and choice.logprobs.content:
            for token_logprob in choice.logprobs.content:
                tokens.append(token_logprob.token)
                top_k = {item.token: item.logprob for item in token_logprob.top_logprobs}
                logprobs_per_token.append(top_k)

        text = choice.message.content

        return {
            "text": text,
            "tokens": tokens,
            "logprobs": logprobs_per_token,
        }

    def precompute_and_cache(self, prompts: list, cache_dir: str) -> None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        def _is_valid(idx: int, prompt: str) -> bool:
            file_path = cache_path / f"{idx}.json"
            if not file_path.exists():
                return False
            try:
                with open(file_path) as f:
                    cached = json.load(f)
                return cached.get("prompt", "") == prompt
            except Exception:
                return False

        pending = [(idx, prompt) for idx, prompt in enumerate(prompts)
                   if not _is_valid(idx, prompt)]

        if not pending:
            print(f"All {len(prompts)} teacher responses already cached.")
            return

        print(f"Caching {len(pending)}/{len(prompts)} teacher responses (resumable)...")
        for i, (idx, prompt) in enumerate(pending):
            file_path = cache_path / f"{idx}.json"
            print(f"  [{idx + 1}/{len(prompts)}] Requesting teacher response...")
            try:
                result = self.get_response_with_logprobs(prompt)
                preview = (result["text"] or "").replace("\n", " ").strip()[:120]
                print(f"  ✓ Response: {preview}...")
                with open(file_path, "w") as f:
                    json.dump({"prompt": prompt, **result}, f)
            except Exception as e:
                print(f"  WARNING: prompt {idx} failed ({e}). Skipping — will retry next run.")
            if i < len(pending) - 1:
                time.sleep(30)

    @staticmethod
    def load_cached(cache_dir: str, idx: int) -> dict | None:
        file_path = Path(cache_dir) / f"{idx}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r") as f:
            return json.load(f)
