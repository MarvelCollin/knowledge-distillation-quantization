import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from src.data.problems import PROMPT_TEMPLATE
from src.evaluation.evaluator import extract_signature
from src.utils.reasoning import (
    SYSTEM_PROMPT,
    THINK_END_TAG,
    build_signature_user_content,
    extract_code,
    extract_fn_name,
)

THINK_BUDGET_RATIO = 0.5

CP_PROMPT = (
    "\n\nRead the problem statement fully and restate what is being asked. "
    "List the constraints and identify the edge cases: empty input, a single element, duplicates, "
    "negative values, and the maximum input sizes. Analyze which algorithms fit the constraints and "
    "choose the SIMPLEST correct approach; confirm its time and space complexity fits the limits "
    "before committing. Plan the solution step by step, then write clean, scalable Python using "
    "EXACTLY the given function signature and parameter names. Define every helper and variable "
    "before you use it. No comments, no fallback logic, no prints, no placeholders. Think deeply: "
    "trace your approach against the given examples and the edge cases you listed, and only write "
    "the code once the logic survives that check. Before closing the code block, verify there are "
    "no syntax errors and the code matches your plan."
)


def _build_code_primer(has_think_end: bool, signature: str) -> str:
    head = "\n```python\n" if has_think_end else "\n</think>\n```python\n"
    if signature:
        return f"{head}{signature}\n"
    return head


class _StopOnSequence(StoppingCriteria):
    def __init__(self, sequence_ids: list, prompt_len: int):
        self.sequence_ids = sequence_ids
        self.prompt_len = prompt_len
        self.seq_len = len(sequence_ids)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if input_ids.shape[1] - self.prompt_len < self.seq_len:
            return False
        tail = input_ids[0, -self.seq_len:].tolist()
        return tail == self.sequence_ids


def generate_with_thinking_cap(model, tokenizer, prompt_text: str,
                                max_new_tokens: int,
                                code_primer_signature: str = "",
                                **gen_kwargs) -> tuple[str, bool]:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id

    think_budget = max(1, int(max_new_tokens * THINK_BUDGET_RATIO))
    code_budget = max(1, max_new_tokens - think_budget)

    think_end_ids = tokenizer.encode(THINK_END_TAG, add_special_tokens=False)
    stop = StoppingCriteriaList([_StopOnSequence(think_end_ids, prompt_len)])

    with torch.no_grad():
        phase1 = model.generate(
            **inputs,
            max_new_tokens=think_budget,
            pad_token_id=eos_id,
            stopping_criteria=stop,
            **gen_kwargs,
        )

    phase1_gen = phase1[0][prompt_len:]
    if phase1_gen.numel() > 0 and phase1_gen[-1].item() == eos_id:
        return tokenizer.decode(phase1_gen, skip_special_tokens=True), False

    phase1_text = tokenizer.decode(phase1_gen, skip_special_tokens=False)
    primer = _build_code_primer(THINK_END_TAG in phase1_text, code_primer_signature)
    suffix_ids = tokenizer.encode(primer, add_special_tokens=False, return_tensors="pt").to(model.device)
    primed = torch.cat([phase1, suffix_ids], dim=1)
    primed_len = primed.shape[1]
    attn = torch.ones_like(primed)

    with torch.no_grad():
        phase2 = model.generate(
            primed,
            attention_mask=attn,
            max_new_tokens=code_budget,
            pad_token_id=eos_id,
            **gen_kwargs,
        )

    gen_tokens = phase2[0][prompt_len:]
    truncated = (phase2.shape[1] - primed_len) >= code_budget
    raw = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return raw, truncated


def build_student_prompt(tokenizer, prompt, test_cases):
    expected = extract_fn_name(test_cases)
    signature = extract_signature(test_cases[0], expected) if test_cases else ""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return fmt, signature


def generate_student_solution(student, prompt, test_cases, max_new_tokens,
                              do_sample=False, temperature=1.0, top_p=1.0):
    fmt, signature = build_student_prompt(student.tokenizer, prompt, test_cases)
    student.eval()
    gen_kwargs = {"do_sample": do_sample}
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)
    raw, _ = generate_with_thinking_cap(
        student.model, student.tokenizer, fmt, max_new_tokens,
        code_primer_signature=signature, **gen_kwargs,
    )
    return raw, extract_code(raw)


def build_eval_prompts(problems, tokenizer, budget_tokens=None, strict_naming=False,
                       cp_prompt=False):
    formatted = []
    signatures = []
    for prob in problems:
        prompt = PROMPT_TEMPLATE.format(text=prob["text"])
        expected = prob.get("entry_point") or extract_fn_name(prob["test_cases"])
        signature = prob.get("signature") or (
            extract_signature(prob["test_cases"][0], expected) if prob["test_cases"] else ""
        )
        user_content = build_signature_user_content(prompt, signature)
        if cp_prompt:
            user_content += CP_PROMPT
        if budget_tokens:
            user_content += (f"\n\nYour entire output is capped at {budget_tokens} tokens. "
                             f"Keep your reasoning brief enough to finish the complete function "
                             f"well before the cap.")
        if strict_naming:
            user_content += ("\n\nIMPORTANT: throughout your code, refer to the function and its "
                             "parameters by EXACTLY the names in the given signature. Never rename "
                             "them or switch to snake_case, and define every helper or variable "
                             "before you use it.")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted.append(fmt)
        signatures.append(signature)
    return formatted, signatures


def free_generate(llm, prompts, num_samples,
                  temperature, top_p, max_new_tokens, seed=1234,
                  repetition_penalty=1.05):
    from vllm import SamplingParams

    do_sample = num_samples > 1
    temp = temperature if do_sample else 0.0
    tp = top_p if do_sample else 1.0

    all_prompts = []
    all_params = []
    sample_index = []
    for i, prompt in enumerate(prompts):
        for j in range(num_samples):
            all_prompts.append(prompt)
            all_params.append(SamplingParams(
                temperature=temp, top_p=tp, max_tokens=max_new_tokens, n=1,
                repetition_penalty=repetition_penalty, seed=seed + j,
            ))
            sample_index.append((i, j))

    outputs = llm.generate(all_prompts, all_params, use_tqdm=True)

    grid = [[None] * num_samples for _ in range(len(prompts))]
    for (i, j), out in zip(sample_index, outputs):
        gen = out.outputs[0]
        truncated = gen.finish_reason == "length"
        grid[i][j] = (gen.text, truncated)
    return grid


def budget_forced_generate(llm, prompts, signatures, num_samples,
                           temperature, top_p, max_new_tokens,
                           think_ratio=0.5, seed=1234,
                           repetition_penalty=1.05):
    from vllm import SamplingParams

    do_sample = num_samples > 1
    temp = temperature if do_sample else 0.0
    tp = top_p if do_sample else 1.0
    think_budget = max(1, int(max_new_tokens * think_ratio))
    code_budget = max(1, max_new_tokens - think_budget)

    think_prompts = []
    think_params = []
    sample_index = []
    for i, prompt in enumerate(prompts):
        for j in range(num_samples):
            think_prompts.append(prompt)
            think_params.append(SamplingParams(
                temperature=temp, top_p=tp, max_tokens=think_budget, n=1,
                stop=[THINK_END_TAG], include_stop_str_in_output=True, repetition_penalty=repetition_penalty, seed=seed + j,
            ))
            sample_index.append((i, j))
    phase1 = llm.generate(think_prompts, think_params, use_tqdm=True)

    cont_prompts = []
    code_params = []
    meta = []
    for (i, j), out in zip(sample_index, phase1):
        sig = signatures[i] if i < len(signatures) else ""
        think_text = out.outputs[0].text
        closed = think_text if THINK_END_TAG in think_text else think_text + "\n" + THINK_END_TAG
        primer = "\n```python\n" + (f"{sig}\n" if sig else "")
        cont_prompts.append(prompts[i] + closed + primer)
        code_params.append(SamplingParams(
            temperature=temp, top_p=tp, max_tokens=code_budget, n=1,
            stop=["```"], include_stop_str_in_output=False, repetition_penalty=repetition_penalty, seed=seed + j,
        ))
        meta.append((i, j, closed, primer))

    phase2 = llm.generate(cont_prompts, code_params, use_tqdm=True)

    grid = [[None] * num_samples for _ in range(len(prompts))]
    for (i, j, closed, primer), out in zip(meta, phase2):
        gen = out.outputs[0]
        truncated = gen.finish_reason == "length"
        final_text = closed + primer + gen.text + "\n```"
        grid[i][j] = (final_text, truncated)
    return grid
