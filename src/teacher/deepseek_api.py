import json
import math
from pathlib import Path

from openai import OpenAI


class DeepSeekTeacher:
    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str,
        max_tokens: int,
        temperature: float,
        top_logprobs: int,
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_logprobs = top_logprobs

    def get_response_with_logprobs(self, prompt: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )
        return self._parse_response(response)

    def _parse_response(self, response) -> dict:
        choice = response.choices[0]
        tokens = []
        logprobs_per_token = []

        if choice.logprobs and choice.logprobs.content:
            for token_logprob in choice.logprobs.content:
                tokens.append(token_logprob.token)
                top_k = {item.token: item.logprob for item in token_logprob.top_logprobs}
                logprobs_per_token.append(top_k)

        return {
            "text": choice.message.content,
            "tokens": tokens,
            "logprobs": logprobs_per_token,
        }

    def precompute_and_cache(self, prompts: list, cache_dir: str) -> None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        for idx, prompt in enumerate(prompts):
            file_path = cache_path / f"{idx}.json"
            if file_path.exists():
                continue
            result = self.get_response_with_logprobs(prompt)
            with open(file_path, "w") as f:
                json.dump({"prompt": prompt, **result}, f)

    @staticmethod
    def load_cached(cache_dir: str, idx: int) -> dict | None:
        file_path = Path(cache_dir) / f"{idx}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r") as f:
            return json.load(f)
