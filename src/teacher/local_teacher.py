import gc
import json
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.utils.reasoning import SYSTEM_PROMPT, extract_code
from src.evaluation.evaluator import run_test_cases


def _gpu_mem_used_gb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()[0]
        return int(out) / 1024.0
    except Exception:
        return 0.0


class LocalTeacherModel:
    def __init__(
        self,
        model_path: str,
        max_tokens: int,
        temperature: float,
        top_logprobs: int,
        student_tokenizer=None,
        top_p: float = 0.95,
        load_in_8bit: bool = False,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 10240,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_logprobs = top_logprobs
        self.student_tokenizer = student_tokenizer

        if load_in_8bit:
            print("  WARNING: load_in_8bit=True ignored (vLLM uses GPTQ/AWQ, not bitsandbytes). Using bf16.")

        from vllm import LLM, SamplingParams

        print(f"Loading vLLM teacher from {model_path}...")
        print(f"  gpu_memory_utilization={gpu_memory_utilization}  max_model_len={max_model_len}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=False,
        )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=top_logprobs,
            repetition_penalty=1.05,
        )

        print(f"  vLLM teacher loaded.  GPU used (nvidia-smi): {_gpu_mem_used_gb():.1f} GB")
        if student_tokenizer is not None:
            print(f"  Student tokenizer wired for token-id top-k cache.")

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _extract_result(self, vllm_output) -> dict:
        gen = vllm_output.outputs[0]
        token_ids = list(gen.token_ids)
        logprobs_list = gen.logprobs or []

        student_vocab = (
            self.student_tokenizer.vocab_size if self.student_tokenizer is not None else 0
        )

        tokens = []
        logprobs_per_token = []
        top_k_ids_per_token = []
        top_k_vals_per_token = []

        for t_id, lp_dict in zip(token_ids, logprobs_list):
            token = self.tokenizer.decode([t_id])
            tokens.append(token)

            top_k_dict = {}
            student_ids = []
            student_vals = []
            if lp_dict is not None:
                for tok_id, lp_obj in lp_dict.items():
                    val_f = float(lp_obj.logprob)
                    tok_str = self.tokenizer.decode([int(tok_id)])
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

        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        result = {"text": text, "tokens": tokens, "logprobs": logprobs_per_token}
        if self.student_tokenizer is not None:
            result["top_k_ids"] = top_k_ids_per_token
            result["top_k_vals"] = top_k_vals_per_token
        return result

    def get_responses_batch(self, prompts: list) -> list:
        formatted = [self._format_prompt(p) for p in prompts]
        outputs = self.llm.generate(formatted, self.sampling_params, use_tqdm=True)
        return [self._extract_result(o) for o in outputs]

    def precompute_and_cache(self, prompts: list, cache_dir: str,
                             test_cases_per_prompt: list = None,
                             chunk_size: int = 64) -> None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        def _is_valid(idx: int, prompt: str) -> bool:
            f = cache_path / f"{idx}.json"
            if not f.exists():
                return False
            try:
                d = json.loads(f.read_text())
                if d.get("prompt", "") != prompt:
                    return False
                tokens = d.get("tokens") or []
                if not tokens:
                    return False
                cached_max_tokens = d.get("max_tokens", 0)
                if cached_max_tokens < self.max_tokens:
                    test_passed = d.get("test_passed")
                    test_total = d.get("test_total")
                    if (
                        test_total is None
                        or test_passed is None
                        or test_total == 0
                        or test_passed < test_total
                    ):
                        return False
                return True
            except Exception:
                return False

        pending = [(idx, p) for idx, p in enumerate(prompts) if not _is_valid(idx, p)]

        if not pending:
            print(f"All {len(prompts)} teacher responses already cached.")
            return

        print(f"Caching {len(pending)}/{len(prompts)} teacher responses via vLLM continuous batching...")
        print(f"  chunk_size={chunk_size}  (save progress every chunk)")
        total_start = time.time()
        pass_count = 0
        fail_count = 0

        for chunk_start in range(0, len(pending), chunk_size):
            chunk = pending[chunk_start:chunk_start + chunk_size]
            chunk_prompts = [p for _, p in chunk]

            chunk_idx = chunk_start // chunk_size + 1
            total_chunks = (len(pending) + chunk_size - 1) // chunk_size
            chunk_t0 = time.time()
            print(f"\n[chunk {chunk_idx}/{total_chunks}] generating {len(chunk)} prompts via vLLM...")

            results = self.get_responses_batch(chunk_prompts)

            for (idx, prompt), result in zip(chunk, results):
                test_passed = None
                test_total = None
                code = ""
                if test_cases_per_prompt is not None:
                    tcs = test_cases_per_prompt[idx]
                    try:
                        code = extract_code(result.get("text", ""))
                        r = run_test_cases(code, tcs)
                        test_passed = r["passed"]
                        test_total = r["total"]
                    except Exception as exc:
                        print(f"  test exec error idx={idx}: {type(exc).__name__}: {exc}")

                if test_total is not None:
                    if test_total > 0 and test_passed == test_total:
                        pass_count += 1
                    else:
                        fail_count += 1

                payload = {"prompt": prompt, "max_tokens": self.max_tokens, **result}
                if test_total is not None:
                    payload["test_passed"] = test_passed
                    payload["test_total"] = test_total
                with open(cache_path / f"{idx}.json", "w") as f:
                    json.dump(payload, f)

            chunk_elapsed = time.time() - chunk_t0
            done = chunk_start + len(chunk)
            total_elapsed = time.time() - total_start
            avg_per_prompt = total_elapsed / done
            eta_sec = avg_per_prompt * (len(pending) - done)
            eta_m, eta_s = divmod(int(eta_sec), 60)
            eta_h, eta_m = divmod(eta_m, 60)

            tested = pass_count + fail_count
            rate = pass_count / max(tested, 1)
            print(
                f"  chunk done in {chunk_elapsed:.1f}s  "
                f"({chunk_elapsed / len(chunk):.1f}s/prompt avg)  "
                f"|  pass {pass_count}/{tested} ({rate:.1%})  "
                f"|  ETA {eta_h}h {eta_m:02d}m {eta_s:02d}s  "
                f"|  GPU {_gpu_mem_used_gb():.1f}GB"
            )

        if test_cases_per_prompt is not None:
            tested = pass_count + fail_count
            rate = pass_count / max(tested, 1)
            total_min = (time.time() - total_start) / 60
            print(f"\n  Cache build complete in {total_min:.1f} min: "
                  f"{pass_count}/{tested} pass all tests ({rate:.1%}).")

    def shutdown(self) -> None:
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception as exc:
            print(f"  vLLM cleanup warning: {exc}")
        if hasattr(self, "llm"):
            del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        gpu_after = _gpu_mem_used_gb()
        print(f"  vLLM shutdown complete. GPU used (nvidia-smi): {gpu_after:.1f} GB")

    @staticmethod
    def load_cached(cache_dir: str, idx: int) -> dict | None:
        from src.data.dataset import clean_teacher_cache
        file_path = Path(cache_dir) / f"{idx}.json"
        if not file_path.exists():
            return None
        with open(file_path) as f:
            data = json.load(f)
        return clean_teacher_cache(data)
