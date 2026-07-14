import gc
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.data.problems import DIFFICULTY_ORDER
from src.data.teacher_cache import PASSED_PATTERN, TOTAL_PATTERN, fully_passed, tail_counts, tail_fully_passed
from src.evaluation.evaluator import failing_cases, run_test_cases
from src.utils.gpu import gpu_used_gb
from src.utils.reasoning import SYSTEM_PROMPT, extract_code
from src.utils.runlog import RunLog, eta_str, show_progress_bars


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
        kv_cache_dtype: str = "auto",
        max_num_seqs: int = None,
        enable_prefix_caching: bool = True,
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
        min_required_gb = 17.0
        free_gb = free_bytes / 1024 ** 3
        if free_gb - 1.0 < min_required_gb:
            raise RuntimeError(
                f"GPU busy: only {free_gb:.1f} GB free of "
                f"{total_bytes / 1024**3:.1f} GB, teacher needs ~{min_required_gb:.0f} GB. "
                f"Another run is holding the GPU — wait for it to finish or stop it first."
            )
        free_util = (free_bytes - safety_bytes) / total_bytes
        gpu_memory_utilization = min(gpu_memory_utilization, free_util)

        print(f"Loading vLLM teacher from {model_path}...")
        print(f"  free {free_bytes / 1024**3:.1f} GB / total {total_bytes / 1024**3:.1f} GB")
        print(f"  gpu_memory_utilization={gpu_memory_utilization:.3f}  max_model_len={max_model_len}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._decode_cache = {}
        self._student_id_cache = {}

        llm_kwargs = dict(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            kv_cache_dtype=kv_cache_dtype,
            enable_prefix_caching=enable_prefix_caching and kv_cache_dtype == "auto",
        )
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        if enable_chunked_prefill:
            llm_kwargs["enable_chunked_prefill"] = True
            llm_kwargs["disable_async_output_proc"] = True
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

    def _tok_str(self, tok_id: int) -> str:
        cached = self._decode_cache.get(tok_id)
        if cached is None:
            cached = self.tokenizer.decode([tok_id])
            self._decode_cache[tok_id] = cached
        return cached

    def _student_id(self, tok_id: int, tok_str: str) -> int | None:
        if tok_id in self._student_id_cache:
            return self._student_id_cache[tok_id]
        student_id = None
        if tok_str:
            re_ids = self.student_tokenizer.encode(tok_str, add_special_tokens=False)
            if len(re_ids) == 1 and 0 <= re_ids[0] < self.student_tokenizer.vocab_size:
                student_id = re_ids[0]
        self._student_id_cache[tok_id] = student_id
        return student_id

    def _topk_to_student(self, lp_dict) -> tuple:
        top_k_dict = {}
        student_ids = []
        student_vals = []
        if lp_dict is not None:
            for tok_id, lp_obj in lp_dict.items():
                val_f = float(lp_obj.logprob)
                tok_id = int(tok_id)
                tok_str = self._tok_str(tok_id)
                top_k_dict[tok_str] = val_f

                if self.student_tokenizer is None:
                    continue
                student_id = self._student_id(tok_id, tok_str)
                if student_id is not None:
                    student_ids.append(student_id)
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
            tokens.append(self._tok_str(int(t_id)))
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
        outputs = self.llm.generate(formatted, self.sampling_params, use_tqdm=show_progress_bars())
        return [self._extract_result(o) for o in outputs]

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
        outputs = self.llm.generate(formatted, sp, use_tqdm=show_progress_bars())
        return [[self._extract_one(g) for g in o.outputs] for o in outputs]

    def _score_result(self, result: dict, test_cases: list) -> dict:
        try:
            return run_test_cases(extract_code(result.get("text", "")), test_cases)
        except Exception as exc:
            print(f"  test exec error: {type(exc).__name__}: {exc}")
            return {"passed": None, "total": None, "details": [], "errors": [], "categories": []}

    @staticmethod
    def _failure_modes(score: dict) -> str:
        cats = [c for c in (score.get("categories") or []) if c and c != "pass"]
        if not cats:
            return "n/a"
        return ", ".join(f"{c}×{cats.count(c)}" for c in dict.fromkeys(cats))

    def _log_failures(self, failed: list, prefix: str = "  ") -> None:
        agg = {}
        for label, score in failed:
            modes = self._failure_modes(score)
            passed, total = score.get("passed"), score.get("total")
            print(f"{prefix}✗ {label}: {passed}/{total} tests | modes: {modes}")
            for cat, err in failing_cases(score, limit=2):
                snippet = " ".join((err or "").split())[:300] or "(no error message)"
                print(f"{prefix}    [{cat}] {snippet}")
            for c in (score.get("categories") or []):
                if c and c != "pass":
                    agg[c] = agg.get(c, 0) + 1
        if agg:
            hist = ", ".join(f"{c}={n}" for c, n in sorted(agg.items(), key=lambda x: -x[1]))
            print(f"{prefix}failure modes (aggregate): {hist}")

    def precompute_and_cache(self, prompts: list, cache_dir: str,
                             test_cases_per_prompt: list = None,
                             difficulty_per_prompt: list = None,
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
                prefix = ('{"prompt": ' + json.dumps(prompt) + ', "max_tokens": ').encode()
                with open(f, "rb") as fh:
                    head = fh.read(len(prefix) + 64)
                    size = fh.seek(0, 2)
                    fh.seek(max(0, size - 400))
                    tail = fh.read()
                if not head.startswith(prefix):
                    return False
                if b'"tokens": []' in head:
                    return False
                m = re.match(rb"\d+", head[len(prefix):])
                if m is None:
                    return False
                cached_max_tokens = int(m.group(0))
                mp = PASSED_PATTERN.search(tail)
                mt = TOTAL_PATTERN.search(tail)
                passed_all = (
                    mp is not None and mt is not None
                    and int(mt.group(1)) > 0
                    and int(mp.group(1)) == int(mt.group(1))
                )
                if rejection_samples > 0 and not passed_all:
                    return False
                if cached_max_tokens < self.max_tokens and not passed_all:
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
        runlog = RunLog("logs/cache_build.jsonl")
        runlog.event("build_start", pending=len(pending), total=len(prompts), chunk_size=chunk_size)
        total_start = time.time()
        pass_count = 0
        fail_count = 0

        from threading import Thread

        bg_thread = None
        bg_error = [None]

        diff_pass = Counter()
        diff_fail = Counter()
        diff_causes = {}

        def _get_diff(idx):
            if difficulty_per_prompt and idx < len(difficulty_per_prompt):
                return difficulty_per_prompt[idx] or "?"
            return "?"

        def _test_and_save(chunk, results, chunk_idx, total_chunks, chunk_t0):
            nonlocal pass_count, fail_count
            try:
                scored = []
                if test_cases_per_prompt is not None:
                    with ThreadPoolExecutor(max_workers=min(16, len(chunk))) as ex:
                        score_out = list(ex.map(
                            lambda ir: self._score_result(ir[1], test_cases_per_prompt[ir[0]]),
                            [(idx, result) for (idx, _), result in zip(chunk, results)]))
                    for (idx, prompt), result, score in zip(chunk, results, score_out):
                        scored.append([idx, prompt, result, score["passed"], score["total"], score])
                else:
                    for (idx, prompt), result in zip(chunk, results):
                        scored.append([idx, prompt, result, None, None, None])

                if rejection_samples > 0 and test_cases_per_prompt is not None:
                    remaining = [s for s in scored
                                 if s[4] is not None and not fully_passed(s[3], s[4])]
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
                            for (ri, ci, candidate, _), score in zip(tasks, task_scores):
                                by_prompt.setdefault(ri, []).append((ci, candidate, score))
                            still = []
                            for ri, s in enumerate(remaining):
                                hit = False
                                for ci, candidate, score in sorted(by_prompt.get(ri, []), key=lambda x: x[0]):
                                    if fully_passed(score["passed"], score["total"]):
                                        s[2], s[3], s[4], s[5] = candidate, score["passed"], score["total"], score
                                        recovered += 1
                                        hit = True
                                        break
                                if not hit:
                                    best = max(by_prompt.get(ri, []),
                                               key=lambda x: (x[2].get("passed") or 0), default=None)
                                    if best is not None and (best[2].get("passed") or 0) > (s[5].get("passed") or 0):
                                        s[5] = best[2]
                                    still.append(s)
                            remaining = still
                        print(f"  rejection recovered {recovered}/{failed_count} prompts "
                              f"(generated up to {generated}/{rejection_samples} per prompt).")

                chunk_pass = chunk_fail = 0
                chunk_diff_pass = Counter()
                chunk_diff_fail = Counter()
                chunk_fail_causes = {}
                for idx, prompt, result, test_passed, test_total, score in scored:
                    diff = _get_diff(idx)
                    new_pass = fully_passed(test_passed, test_total)
                    if test_total is not None and not new_pass and tail_fully_passed(cache_path / f"{idx}.json"):
                        pass_count += 1
                        chunk_pass += 1
                        diff_pass[diff] += 1
                        chunk_diff_pass[diff] += 1
                        continue

                    if test_total is not None:
                        if new_pass:
                            pass_count += 1
                            chunk_pass += 1
                            diff_pass[diff] += 1
                            chunk_diff_pass[diff] += 1
                        else:
                            fail_count += 1
                            chunk_fail += 1
                            diff_fail[diff] += 1
                            chunk_diff_fail[diff] += 1
                            if score is not None:
                                non_pass = [c for c in score.get("categories", []) if c != "pass"]
                                cause = Counter(non_pass).most_common(1)[0][0] if non_pass else "unknown"
                                chunk_fail_causes.setdefault(diff, Counter())[cause] += 1
                                diff_causes.setdefault(diff, Counter())[cause] += 1

                    payload = {"prompt": prompt, "max_tokens": self.max_tokens, **result}
                    if test_total is not None:
                        payload["test_passed"] = test_passed
                        payload["test_total"] = test_total
                        if score is not None and not new_pass:
                            non_pass = [c for c in score.get("categories", []) if c != "pass"]
                            payload["fail_cause"] = Counter(non_pass).most_common(1)[0][0] if non_pass else "unknown"
                    with open(cache_path / f"{idx}.json", "w") as f:
                        json.dump(payload, f)

                chunk_elapsed = time.time() - chunk_t0
                done_count = min(chunk_idx * chunk_size, len(pending))
                total_elapsed = time.time() - total_start
                avg_per_prompt = total_elapsed / max(done_count, 1)
                eta_sec = avg_per_prompt * (len(pending) - done_count)

                tested = pass_count + fail_count
                rate = pass_count / max(tested, 1)

                print(
                    f"  chunk {chunk_idx}/{total_chunks} done in {chunk_elapsed:.1f}s  "
                    f"({chunk_elapsed / len(chunk):.1f}s/prompt)  "
                    f"|  pass {chunk_pass}/{len(chunk)}  "
                    f"|  total {pass_count}/{tested} ({rate:.1%})  "
                    f"|  ETA {eta_str(eta_sec)}  "
                    f"|  GPU {gpu_used_gb():.1f}GB"
                )
                all_diffs = sorted(set(list(chunk_diff_pass) + list(chunk_diff_fail)),
                                   key=lambda d: DIFFICULTY_ORDER.get(d, 3))
                for d in all_diffs:
                    cp = chunk_diff_pass.get(d, 0)
                    cf = chunk_diff_fail.get(d, 0)
                    tp = diff_pass.get(d, 0)
                    tf = diff_fail.get(d, 0)
                    causes_str = ""
                    if d in chunk_fail_causes:
                        causes_str = "  " + ", ".join(
                            f"{c}={n}" for c, n in chunk_fail_causes[d].most_common())
                    print(
                        f"    {d:<8} {cp}/{cp+cf} solved  "
                        f"(total {tp}/{tp+tf}  {tp/max(tp+tf,1):.0%})"
                        f"{causes_str}"
                    )

                runlog.event(
                    "chunk", idx=chunk_idx, of=total_chunks, secs=round(chunk_elapsed, 1),
                    s_per_prompt=round(chunk_elapsed / len(chunk), 1),
                    chunk_pass=chunk_pass, chunk_fail=chunk_fail,
                    total_pass=pass_count, tested=tested, pass_rate=round(rate, 4),
                    by_difficulty={
                        d: {"pass": diff_pass.get(d, 0), "fail": diff_fail.get(d, 0),
                            "causes": dict(diff_causes.get(d, {}))}
                        for d in set(list(diff_pass) + list(diff_fail))
                    },
                    gpu_gb=round(gpu_used_gb(), 1), eta_s=int(eta_sec),
                )
            except Exception as e:
                bg_error[0] = e

        for chunk_start in range(0, len(pending), chunk_size):
            chunk = pending[chunk_start:chunk_start + chunk_size]
            chunk_prompts = [p for _, p in chunk]

            chunk_idx = chunk_start // chunk_size + 1
            total_chunks = (len(pending) + chunk_size - 1) // chunk_size
            chunk_t0 = time.time()
            print(f"\n[chunk {chunk_idx}/{total_chunks}] generating {len(chunk)} prompts via vLLM...")

            results = self.get_responses_batch(chunk_prompts)

            if bg_thread is not None:
                bg_thread.join()
                if bg_error[0] is not None:
                    raise bg_error[0]

            bg_thread = Thread(
                target=_test_and_save,
                args=(chunk, results, chunk_idx, total_chunks, chunk_t0),
            )
            bg_thread.start()

        if bg_thread is not None:
            bg_thread.join()
            if bg_error[0] is not None:
                raise bg_error[0]

        if test_cases_per_prompt is not None:
            tested = pass_count + fail_count
            rate = pass_count / max(tested, 1)
            total_min = (time.time() - total_start) / 60
            print(f"\n  Cache build complete in {total_min:.1f} min: "
                  f"{pass_count}/{tested} pass all tests ({rate:.1%}).")
            runlog.event("build_done", minutes=round(total_min, 1),
                         total_pass=pass_count, tested=tested, pass_rate=round(rate, 4))

    def _rescore_one(self, full_token_ids: list, prompt_logprobs: list, prompt_len: int) -> dict:
        tokens = []
        logprobs_per_token = []
        top_k_ids_per_token = []
        top_k_vals_per_token = []
        for pos in range(prompt_len, len(full_token_ids)):
            tokens.append(self._tok_str(int(full_token_ids[pos])))
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
        resumed = 0
        for f in files:
            if (dst / f.name).exists():
                resumed += 1
                continue
            d = json.loads(f.read_text())
            if d.get("prompt") and d.get("text") and d.get("tokens"):
                pending.append((int(f.stem), d))

        print(f"Rescoring {len(pending)} cached trajectories: {src_cache_dir} -> {dst_cache_dir}"
              + (f"  (resuming — {resumed} already done)" if resumed else ""))
        print(f"  prompt_logprobs={self.top_logprobs}  temperature=1.0  top_p=1.0  (true teacher distribution)")

        difficulties = {}
        plist = sorted(Path("cache").glob("problems_*.json"))
        if plist:
            probs = json.loads(plist[0].read_text())
            difficulties = {i: (p.get("difficulty") or "?") for i, p in enumerate(probs)}
        diff_pass, diff_done = Counter(), Counter()
        for f in dst.glob("*.json"):
            tp, tt = tail_counts(f, 300)
            dname = difficulties.get(int(f.stem), "?")
            diff_done[dname] += 1
            if fully_passed(tp, tt):
                diff_pass[dname] += 1

        score_params = SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=1, prompt_logprobs=self.top_logprobs,
        )
        cap = self.max_model_len - 1
        total_start = time.time()

        for chunk_start in range(0, len(pending), chunk_size):
            chunk = pending[chunk_start:chunk_start + chunk_size]
            token_prompts = []
            metas = []
            enc = self.student_tokenizer or self.tokenizer
            for idx, d in chunk:
                p_ids = enc(d["prompt"], add_special_tokens=False).input_ids
                r_ids = enc(d["text"], add_special_tokens=False).input_ids
                full_ids = (p_ids + r_ids)[:cap]
                token_prompts.append({"prompt_token_ids": full_ids})
                metas.append((idx, d, len(p_ids)))

            outputs = None
            last_err = None
            for attempt in (1, 2):
                try:
                    outputs = self.llm.generate(token_prompts, score_params, use_tqdm=False)
                    break
                except Exception as err:
                    last_err = err
                    print(f"  ⚠ rescore attempt {attempt} failed on idx {[m[0] for m in metas]}: {type(err).__name__}: {err}")
            if outputs is None:
                for idx, d, _ in metas:
                    with open(dst / f"{idx}.json", "w") as fh:
                        json.dump(d, fh)
                    print(f"  ⚠ idx {idx}: original logprobs kept, rescore skipped for this entry")
                raise RuntimeError(f"engine unhealthy after retry, restart resumes past idx {metas[-1][0]}") from last_err

            for (idx, d, prompt_len), out in zip(metas, outputs):
                try:
                    result = self._rescore_one(out.prompt_token_ids, out.prompt_logprobs or [], prompt_len)
                    payload = {"prompt": d["prompt"], "max_tokens": d.get("max_tokens", self.max_tokens), **result}
                    if d.get("test_total") is not None:
                        payload["test_passed"] = d.get("test_passed")
                        payload["test_total"] = d.get("test_total")
                except Exception as err:
                    print(f"  ⚠ idx {idx}: extraction failed ({type(err).__name__}: {err}), original logprobs kept")
                    payload = d
                with open(dst / f"{idx}.json", "w") as fh:
                    json.dump(payload, fh)
                dname = difficulties.get(idx, "?")
                diff_done[dname] += 1
                tp_, tt_ = payload.get("test_passed"), payload.get("test_total")
                if tp_ is not None and tt_ and tp_ == tt_:
                    diff_pass[dname] += 1

            done = min(chunk_start + chunk_size, len(pending))
            elapsed = time.time() - total_start
            eta = elapsed / done * (len(pending) - done)
            eta_m, eta_s = divmod(int(eta), 60)
            n_pass, n_done = sum(diff_pass.values()), sum(diff_done.values())
            tally = "  ".join(f"{dn} {diff_pass[dn]}/{diff_done[dn]}" for dn in ("Easy", "Medium", "Hard"))
            print(f"  rescored {done + resumed}/{len(pending) + resumed}  ({elapsed / done:.2f}s/problem)  ETA {eta_m}m {eta_s:02d}s  GPU {gpu_used_gb():.1f}GB")
            print(f"    passing {n_pass}/{n_done} ({n_pass * 100 // max(n_done, 1)}%)  [{tally}]")

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
