import gc
import json
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


class LocalTeacherModel:
    def __init__(
        self,
        model_path: str,
        max_tokens: int,
        temperature: float,
        top_logprobs: int,
        student_tokenizer=None,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_logprobs = top_logprobs
        self.student_tokenizer = student_tokenizer

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
        if student_tokenizer is not None:
            print(f"  Student tokenizer wired for token-id top-k cache.")

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
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = output.sequences[0][input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        student_vocab = (
            self.student_tokenizer.vocab_size if self.student_tokenizer is not None else 0
        )

        tokens = []
        logprobs_per_token = []
        top_k_ids_per_token = []
        top_k_vals_per_token = []
        for token_id, scores in zip(generated_ids, output.scores):
            token = self.tokenizer.decode([token_id.item()])
            tokens.append(token)
            log_probs = F.log_softmax(scores[0].float(), dim=-1)
            top_k = torch.topk(log_probs, k=min(self.top_logprobs, log_probs.shape[-1]))

            top_k_dict = {}
            student_ids = []
            student_vals = []
            for idx, val in zip(top_k.indices, top_k.values):
                idx_int = idx.item()
                val_f = val.item()
                tok_str = self.tokenizer.decode([idx_int])
                top_k_dict[tok_str] = val_f

                if self.student_tokenizer is None or not tok_str:
                    continue
                re_ids = self.student_tokenizer.encode(tok_str, add_special_tokens=False)
                if len(re_ids) == 1 and 0 <= re_ids[0] < student_vocab:
                    student_ids.append(re_ids[0])
                    student_vals.append(val_f)

            logprobs_per_token.append(top_k_dict)
            top_k_ids_per_token.append(student_ids)
            top_k_vals_per_token.append(student_vals)

        result = {"text": text, "tokens": tokens, "logprobs": logprobs_per_token}
        if self.student_tokenizer is not None:
            result["top_k_ids"] = top_k_ids_per_token
            result["top_k_vals"] = top_k_vals_per_token
        return result

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
            result = None
            for attempt in range(2):
                try:
                    result = self.get_response_with_logprobs(prompt)
                    break
                except torch.cuda.OutOfMemoryError:
                    print(f"\n  OOM on attempt {attempt + 1}, clearing GPU cache...", flush=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"\n  WARNING: prompt {idx} failed ({e}). Saving empty stub.", flush=True)
                    break

            if result is not None:
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
            else:
                print(f"\n  Saving empty stub for prompt {idx} (skipped during distillation).", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                with open(cache_path / f"{idx}.json", "w") as f:
                    json.dump({"prompt": prompt, "text": "", "tokens": [], "logprobs": []}, f)

    @staticmethod
    def load_cached(cache_dir: str, idx: int) -> dict | None:
        file_path = Path(cache_dir) / f"{idx}.json"
        if not file_path.exists():
            return None
        with open(file_path) as f:
            return json.load(f)
