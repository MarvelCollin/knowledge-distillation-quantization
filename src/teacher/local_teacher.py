import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.evaluation.evaluator import run_test_cases
from src.utils.gpu import gpu_used_gb
from src.utils.reasoning import SYSTEM_PROMPT, extract_code


class LocalTeacherModel:
    def __init__(
        self,
        model_path: str,
        max_tokens: int,
        temperature: float,
        top_logprobs: int,
        student_tokenizer=None,
        top_p: float = 0.95,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 10240,
        enforce_eager: bool = False,
        enable_chunked_prefill: bool = False,
        max_num_batched_tokens: int = None,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_logprobs = top_logprobs
        self.student_tokenizer = student_tokenizer
        self.max_model_len = max_model_len

        from vllm import LLM, SamplingParams

        gc.collect()
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        safety_bytes = 1024 ** 3
        free_util = max(0.1, (free_bytes - safety_bytes) / total_bytes)
        gpu_memory_utilization = min(gpu_memory_utilization, free_util)

        print(f"Loading vLLM teacher from {model_path}...")
        print(f"  free {free_bytes / 1024**3:.1f} GB / total {total_bytes / 1024**3:.1f} GB")
        print(f"  gpu_memory_utilization={gpu_memory_utilization:.3f}  max_model_len={max_model_len}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        llm_kwargs = dict(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            enable_prefix_caching=True,
        )
        if enable_chunked_prefill:
            llm_kwargs["enable_chunked_prefill"] = True
            if max_num_batched_tokens is not None:
                llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        self.llm = LLM(**llm_kwargs)

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=top_logprobs,
        )

        print(f"  vLLM teacher loaded.  GPU used (nvidia-smi): {gpu_used_gb():.1f} GB")
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
        return self._extract_one(vllm_output.outputs[0])

    def _topk_to_student(self, lp_dict) -> tuple:
        student_vocab = (
            self.student_tokenizer.vocab_size if self.student_tokenizer is not None else 0
        )
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
        return top_k_dict, student_ids, student_vals

    def _extract_one(self, gen) -> dict:
        token_ids = list(gen.token_ids)
        logprobs_list = gen.logprobs or []

        tokens = []
        logprobs_per_token = []
        top_k_ids_per_token = []
        top_k_vals_per_token = []

        for t_id, lp_dict in zip(token_ids, logprobs_list):
            tokens.append(self.tokenizer.decode([t_id]))
            top_k_dict, student_ids, student_vals = self._topk_to_student(lp_dict)
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

    def _generate_raw(self, prompts: list) -> list:
        """Generate one trajectory per prompt, returning raw vLLM outputs.

        Logprob->student extraction is deferred to _extract_one so it runs only on
        the trajectory finally kept for each prompt (see precompute_and_cache)."""
        formatted = [self._format_prompt(p) for p in prompts]
        outputs = self.llm.generate(formatted, self.sampling_params, use_tqdm=True)
        return [o.outputs[0] for o in outputs]

    def sample_candidates_batch(self, prompts: list, n: int,
                                temperature: float, top_p: float) -> list:
        from vllm import SamplingParams

        sp = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=self.max_tokens,
            logprobs=self.top_logprobs,
            repetition_penalty=1.05,
        )
        formatted = [self._format_prompt(p) for p in prompts]
        outputs = self.llm.generate(formatted, sp, use_tqdm=True)
        return [list(o.outputs) for o in outputs]

    @staticmethod
    def _fully_passed(test_passed, test_total) -> bool:
        return test_total is not None and test_total > 0 and test_passed == test_total

    def _score_result(self, result: dict, test_cases: list) -> tuple:
        return self._score_text(result.get("text", ""), test_cases)

    def _score_gen(self, gen, test_cases: list) -> tuple:
        """Score a raw vLLM output by decoding only its text (no logprob extraction)."""
        return self._score_text(
            self.tokenizer.decode(list(gen.token_ids), skip_special_tokens=True),
            test_cases,
        )

    def _score_text(self, text: str, test_cases: list) -> tuple:
        try:
            code = extract_code(text)
            r = run_test_cases(code, test_cases)
            return r["passed"], r["total"]
        except Exception as exc:
            print(f"  test exec error: {type(exc).__name__}: {exc}")
            return None, None

    def precompute_and_cache(self, prompts: list, cache_dir: str,
                             test_cases_per_prompt: list = None,
                             chunk_size: int = 64,
                             rejection_samples: int = 0,
                             rejection_wave: int = 2,
                             rejection_temperature: float = 1.0,
                             rejection_top_p: float = 0.95) -> None:
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
                if rejection_samples > 0 and not self._fully_passed(
                        d.get("test_passed"), d.get("test_total")):
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

            results = self._generate_raw(chunk_prompts)

            scored = []
            if test_cases_per_prompt is not None:
                with ThreadPoolExecutor(max_workers=min(16, len(chunk))) as ex:
                    score_out = list(ex.map(
                        lambda ir: self._score_gen(ir[1], test_cases_per_prompt[ir[0]]),
                        [(idx, gen) for (idx, _), gen in zip(chunk, results)]))
                for (idx, prompt), gen, (test_passed, test_total) in zip(chunk, results, score_out):
                    scored.append([idx, prompt, gen, test_passed, test_total])
            else:
                for (idx, prompt), gen in zip(chunk, results):
                    scored.append([idx, prompt, gen, None, None])

            if rejection_samples > 0 and test_cases_per_prompt is not None:
                remaining = [s for s in scored
                             if s[4] is not None and not self._fully_passed(s[3], s[4])]
                failed_count = len(remaining)
                if remaining:
                    wave = max(1, rejection_wave)
                    print(f"  rejection sampling {failed_count} failed prompts "
                          f"(n={rejection_samples}, wave={wave}, temp={rejection_temperature})...")
                    recovered = 0
                    generated = 0
                    while remaining and generated < rejection_samples:
                        n_this = min(wave, rejection_samples - generated)
                        candidate_lists = self.sample_candidates_batch(
                            [s[1] for s in remaining], n_this,
                            rejection_temperature, rejection_top_p)
                        generated += n_this
                        tasks = []
                        for ri, (s, candidates) in enumerate(zip(remaining, candidate_lists)):
                            tcs = test_cases_per_prompt[s[0]]
                            for ci, candidate in enumerate(candidates):
                                tasks.append((ri, ci, candidate, tcs))
                        with ThreadPoolExecutor(max_workers=min(16, max(1, len(tasks)))) as ex:
                            task_scores = list(ex.map(
                                lambda t: self._score_result(t[2], t[3]), tasks))
                        by_prompt = {}
                        for (ri, ci, candidate, _), (cand_passed, cand_total) in zip(tasks, task_scores):
                            by_prompt.setdefault(ri, []).append((ci, candidate, cand_passed, cand_total))
                        still = []
                        for ri, s in enumerate(remaining):
                            hit = False
                            for ci, candidate, cand_passed, cand_total in sorted(by_prompt.get(ri, []), key=lambda x: x[0]):
                                if self._fully_passed(cand_passed, cand_total):
                                    s[2], s[3], s[4] = candidate, cand_passed, cand_total
                                    recovered += 1
                                    hit = True
                                    break
                            if not hit:
                                still.append(s)
                        remaining = still
                    print(f"  rejection recovered {recovered}/{failed_count} prompts "
                          f"(generated up to {generated}/{rejection_samples} per prompt).")

            for idx, prompt, result, test_passed, test_total in scored:
                if test_total is not None:
                    if self._fully_passed(test_passed, test_total):
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
                f"|  GPU {gpu_used_gb():.1f}GB"
            )

        if test_cases_per_prompt is not None:
            tested = pass_count + fail_count
            rate = pass_count / max(tested, 1)
            total_min = (time.time() - total_start) / 60
            print(f"\n  Cache build complete in {total_min:.1f} min: "
                  f"{pass_count}/{tested} pass all tests ({rate:.1%}).")

    def _rescore_one(self, full_token_ids: list, prompt_logprobs: list, prompt_len: int) -> dict:
        tokens = []
        logprobs_per_token = []
        top_k_ids_per_token = []
        top_k_vals_per_token = []
        for pos in range(prompt_len, len(full_token_ids)):
            tokens.append(self.tokenizer.decode([full_token_ids[pos]]))
            lp_dict = prompt_logprobs[pos] if pos < len(prompt_logprobs) else None
            top_k_dict, student_ids, student_vals = self._topk_to_student(lp_dict)
            logprobs_per_token.append(top_k_dict)
            top_k_ids_per_token.append(student_ids)
            top_k_vals_per_token.append(student_vals)
        text = self.tokenizer.decode(full_token_ids[prompt_len:], skip_special_tokens=True)
        result = {"text": text, "tokens": tokens, "logprobs": logprobs_per_token}
        if self.student_tokenizer is not None:
            result["top_k_ids"] = top_k_ids_per_token
            result["top_k_vals"] = top_k_vals_per_token
        return result

    def rescore_and_cache(self, src_cache_dir: str, dst_cache_dir: str, chunk_size: int = 8) -> None:
        from vllm import SamplingParams

        src = Path(src_cache_dir)
        dst = Path(dst_cache_dir)
        dst.mkdir(parents=True, exist_ok=True)

        files = sorted(src.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 1 << 30)
        pending = []
        for f in files:
            d = json.loads(f.read_text())
            if d.get("prompt") and d.get("text") and d.get("tokens"):
                pending.append((int(f.stem), d))

        print(f"Rescoring {len(pending)} cached trajectories: {src_cache_dir} -> {dst_cache_dir}")
        print(f"  prompt_logprobs={self.top_logprobs}  temperature=1.0  top_p=1.0  (true teacher distribution)")

        score_params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=1, prompt_logprobs=self.top_logprobs,
        )
        cap = self.max_model_len - 1
        total_start = time.time()

        for chunk_start in range(0, len(pending), chunk_size):
            chunk = pending[chunk_start:chunk_start + chunk_size]
            token_prompts = []
            metas = []
            for idx, d in chunk:
                p_ids = self.tokenizer(d["prompt"], add_special_tokens=False).input_ids
                r_ids = self.tokenizer(d["text"], add_special_tokens=False).input_ids
                full_ids = (p_ids + r_ids)[:cap]
                token_prompts.append({"prompt_token_ids": full_ids})
                metas.append((idx, d, len(p_ids)))

            outputs = self.llm.generate(token_prompts, score_params, use_tqdm=False)

            for (idx, d, prompt_len), out in zip(metas, outputs):
                result = self._rescore_one(out.prompt_token_ids, out.prompt_logprobs or [], prompt_len)
                payload = {"prompt": d["prompt"], "max_tokens": d.get("max_tokens", self.max_tokens), **result}
                if d.get("test_total") is not None:
                    payload["test_passed"] = d.get("test_passed")
                    payload["test_total"] = d.get("test_total")
                with open(dst / f"{idx}.json", "w") as fh:
                    json.dump(payload, fh)

            done = min(chunk_start + chunk_size, len(pending))
            elapsed = time.time() - total_start
            eta = elapsed / done * (len(pending) - done)
            eta_m, eta_s = divmod(int(eta), 60)
            print(f"  rescored {done}/{len(pending)}  ({elapsed / done:.2f}s/problem)  ETA {eta_m}m {eta_s:02d}s  GPU {gpu_used_gb():.1f}GB")

        print(f"\n  Rescore complete in {(time.time() - total_start) / 60:.1f} min -> {dst_cache_dir}")

    def shutdown(self) -> None:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.llm.llm_engine.model_executor
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        gpu_after = gpu_used_gb()
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
