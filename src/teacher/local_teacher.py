import gc

import torch
from transformers import AutoTokenizer

from src.utils.gpu import gpu_used_gb
from src.utils.reasoning import SYSTEM_PROMPT
from src.utils.runlog import show_progress_bars


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

    @classmethod
    def from_config(cls, config: dict, student_tokenizer=None, **overrides):
        teacher = config["teacher"]
        kwargs = dict(
            model_path=teacher["local_model_path"],
            max_tokens=teacher["max_tokens"],
            temperature=teacher["temperature"],
            top_p=teacher.get("top_p", 0.95),
            top_logprobs=teacher["top_logprobs"],
            student_tokenizer=student_tokenizer,
            gpu_memory_utilization=teacher.get("gpu_memory_utilization", 0.85),
            max_model_len=teacher.get("max_model_len", 10240),
            kv_cache_dtype=teacher.get("kv_cache_dtype", "auto"),
            enable_chunked_prefill=teacher.get("enable_chunked_prefill", False),
            max_num_batched_tokens=teacher.get("max_num_batched_tokens", None),
            max_num_seqs=teacher.get("max_num_seqs", None),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

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

    def _extract_tokens(self, token_ids: list, logprobs_list: list) -> tuple:
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
        return tokens, logprobs_per_token, top_k_ids_per_token, top_k_vals_per_token

    def _build_result(self, text: str, tokens: list, logprobs: list,
                      top_k_ids: list, top_k_vals: list) -> dict:
        result = {"text": text, "tokens": tokens, "logprobs": logprobs}
        if self.student_tokenizer is not None:
            result["top_k_ids"] = top_k_ids
            result["top_k_vals"] = top_k_vals
        return result

    def _extract_one(self, gen) -> dict:
        token_ids = list(gen.token_ids)
        tokens, logprobs, top_k_ids, top_k_vals = self._extract_tokens(token_ids, gen.logprobs or [])
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return self._build_result(text, tokens, logprobs, top_k_ids, top_k_vals)

    def rescore_tokens(self, full_token_ids: list, prompt_logprobs: list, prompt_len: int) -> dict:
        resp_ids = list(full_token_ids[prompt_len:])
        lp_slice = [prompt_logprobs[pos] if pos < len(prompt_logprobs) else None
                    for pos in range(prompt_len, len(full_token_ids))]
        tokens, logprobs, top_k_ids, top_k_vals = self._extract_tokens(resp_ids, lp_slice)
        text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
        return self._build_result(text, tokens, logprobs, top_k_ids, top_k_vals)

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
