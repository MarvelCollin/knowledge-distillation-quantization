import json
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


class LocalTeacherModel:
    def __init__(self, model_path: str, max_tokens: int, temperature: float, top_logprobs: int):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_logprobs = top_logprobs

        print(f"Loading local teacher from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
        print(f"Local teacher loaded.  GPU mem used: {gpu_mem:.1f} GB")

    _SYSTEM = "You are a coding assistant. Output ONLY a raw Python function definition. No explanation, no markdown, no triple backticks. Start directly with def."

    def get_response_with_logprobs(self, prompt: str) -> dict:
        messages = [
            {"role": "system", "content": self._SYSTEM},
            {"role": "user", "content": prompt},
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )

        generated_ids = generated[0][input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        full_ids = generated[0].unsqueeze(0)
        with torch.no_grad():
            logits = self.model(full_ids).logits

        gen_logits = logits[0, input_length - 1: input_length - 1 + len(generated_ids)]

        tokens = []
        logprobs_per_token = []
        for i, token_id in enumerate(generated_ids):
            token = self.tokenizer.decode([token_id.item()])
            tokens.append(token)
            log_probs = F.log_softmax(gen_logits[i].float(), dim=-1)
            top_k = torch.topk(log_probs, k=min(self.top_logprobs, log_probs.shape[-1]))
            top_k_dict = {
                self.tokenizer.decode([idx.item()]): val.item()
                for idx, val in zip(top_k.indices, top_k.values)
            }
            logprobs_per_token.append(top_k_dict)

        return {"text": text, "tokens": tokens, "logprobs": logprobs_per_token}

    def precompute_and_cache(self, prompts: list, cache_dir: str) -> None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        def _is_valid(idx: int, prompt: str) -> bool:
            f = cache_path / f"{idx}.json"
            if not f.exists():
                return False
            try:
                d = json.loads(f.read_text())
                return d.get("prompt", "") == prompt
            except Exception:
                return False

        pending = [(idx, p) for idx, p in enumerate(prompts) if not _is_valid(idx, p)]

        if not pending:
            print(f"All {len(prompts)} teacher responses already cached.")
            return

        print(f"Caching {len(pending)}/{len(prompts)} teacher responses (local model)...")
        total_start = time.time()

        for i, (idx, prompt) in enumerate(pending):
            print(f"\n  [{i + 1}/{len(pending)}] prompt #{idx} — generating...", flush=True)
            t0 = time.time()
            try:
                result = self.get_response_with_logprobs(prompt)
                elapsed = time.time() - t0
                done = i + 1
                avg = (time.time() - total_start) / done
                eta = avg * (len(pending) - done)
                print(
                    f"\n  ✓ {len(result['tokens'])} tokens  {elapsed:.1f}s"
                    f"  |  ETA {eta / 60:.0f}m {eta % 60:.0f}s",
                    flush=True,
                )
                with open(cache_path / f"{idx}.json", "w") as f:
                    json.dump({"prompt": prompt, **result}, f)
            except Exception as e:
                print(f"\n  WARNING: prompt {idx} failed ({e}). Skipping.", flush=True)

    @staticmethod
    def load_cached(cache_dir: str, idx: int) -> dict | None:
        file_path = Path(cache_dir) / f"{idx}.json"
        if not file_path.exists():
            return None
        with open(file_path) as f:
            return json.load(f)
