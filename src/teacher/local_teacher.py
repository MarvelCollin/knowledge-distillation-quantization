import gc
import json
import subprocess
import sys
import threading
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.reasoning import SYSTEM_PROMPT, extract_code
from src.evaluation.evaluator import run_test_cases, _format_error


class LocalTeacherModel:
    def __init__(
        self,
        model_path: str,
        max_tokens: int,
        temperature: float,
        top_logprobs: int,
        student_tokenizer=None,
        top_p: float = 0.95,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
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

    def get_response_with_logprobs(self, prompt: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.05,
                no_repeat_ngram_size=6,
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

    def precompute_and_cache(self, prompts: list, cache_dir: str,
                             test_cases_per_prompt: list = None) -> None:
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

        def _gpu_stats() -> str:
            try:
                raw = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                temp, mem_used, mem_total, util = raw.split(", ")
                return f"{temp}°C  {int(mem_used)/1024:.1f}/{int(mem_total)/1024:.0f}GB  util {util}%"
            except Exception:
                return "GPU n/a"

        print(f"Caching {len(pending)}/{len(prompts)} teacher responses (local model)...")
        total_start = time.time()
        pass_count = 0
        fail_count = 0

        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        def _run_with_spinner(fn, *args):
            container = {"result": None, "error": None}
            stop_event = threading.Event()

            def _worker():
                try:
                    container["result"] = fn(*args)
                except Exception as exc:
                    container["error"] = exc
                finally:
                    stop_event.set()

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            frame = 0
            gen_start = time.time()
            while not stop_event.is_set():
                elapsed = time.time() - gen_start
                sys.stdout.write(f"\r  {spinner_frames[frame % len(spinner_frames)]}  reasoning... {elapsed:.0f}s")
                sys.stdout.flush()
                stop_event.wait(timeout=0.12)
                frame += 1
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()
            t.join()
            if container["error"] is not None:
                raise container["error"]
            return container["result"]

        for i, (idx, prompt) in enumerate(pending):
            print(f"\n[{i + 1}/{len(pending)}] prompt #{idx}", flush=True)
            t0 = time.time()
            result = None
            for attempt in range(2):
                try:
                    result = _run_with_spinner(self.get_response_with_logprobs, prompt)
                    break
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM on attempt {attempt + 1}, clearing GPU cache...", flush=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"  WARNING: prompt {idx} failed ({e}). Saving empty stub.", flush=True)
                    break

            test_passed = None
            test_total = None
            test_errors = []
            test_details = []
            code = ""
            if result is not None and test_cases_per_prompt is not None:
                tcs = test_cases_per_prompt[idx]
                try:
                    code = extract_code(result.get("text", ""))
                    r = run_test_cases(code, tcs)
                    test_passed = r["passed"]
                    test_total = r["total"]
                    test_errors = r["errors"]
                    test_details = r["details"]
                except Exception as exc:
                    print(f"  ⚠ test exec error: {type(exc).__name__}: {exc}", flush=True)

            if result is not None:
                elapsed = time.time() - t0
                done = i + 1
                avg = (time.time() - total_start) / done
                eta = avg * (len(pending) - done)
                eta_m, eta_s = divmod(int(eta), 60)

                if test_total is not None:
                    if test_total > 0 and test_passed == test_total:
                        pass_count += 1
                        verdict = f"✓ PASSED {test_passed}/{test_total}"
                    else:
                        fail_count += 1
                        verdict = f"✗ FAILED {test_passed}/{test_total}"
                else:
                    verdict = "no tests"

                print(
                    f"  tokens : {len(result['tokens'])}",
                    flush=True,
                )
                print(
                    f"  time   : {elapsed:.1f}s  |  ETA {eta_m}m {eta_s:02d}s",
                    flush=True,
                )
                print(
                    f"  gpu    : {_gpu_stats()}",
                    flush=True,
                )
                print(
                    f"  result : {verdict}  |  running {pass_count}/{pass_count + fail_count} passed",
                    flush=True,
                )
                code_lines = code.strip().splitlines() if code.strip() else []
                if code_lines:
                    print("  --- extracted code ---", flush=True)
                    for line in code_lines:
                        print(f"  {line}", flush=True)
                    print("  ----------------------", flush=True)
                else:
                    print("  code   : (no code extracted)", flush=True)
                if test_details and test_passed is not None and test_passed < (test_total or 0):
                    print("  --- failure details ---", flush=True)
                    for j, (detail, err) in enumerate(zip(test_details, test_errors)):
                        if detail != "pass" and err:
                            print(f"  test {j}: {detail} — {_format_error(err)}", flush=True)
                    print("  -----------------------", flush=True)
                payload = {"prompt": prompt, "max_tokens": self.max_tokens, **result}
                if test_total is not None:
                    payload["test_passed"] = test_passed
                    payload["test_total"] = test_total
                with open(cache_path / f"{idx}.json", "w") as f:
                    json.dump(payload, f)
            else:
                fail_count += 1
                print(f"  skipped — saving empty stub", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                with open(cache_path / f"{idx}.json", "w") as f:
                    json.dump({"prompt": prompt, "text": "", "tokens": [], "logprobs": [], "max_tokens": self.max_tokens}, f)

        if test_cases_per_prompt is not None:
            tested = pass_count + fail_count
            rate = pass_count / max(tested, 1)
            print(f"\n  Cache build complete: {pass_count}/{tested} prompts pass all tests ({rate:.1%}).")

    @staticmethod
    def load_cached(cache_dir: str, idx: int) -> dict | None:
        from src.data.dataset import clean_teacher_cache
        file_path = Path(cache_dir) / f"{idx}.json"
        if not file_path.exists():
            return None
        with open(file_path) as f:
            data = json.load(f)
        return clean_teacher_cache(data)
