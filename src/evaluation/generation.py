from src.data.dataset import PROMPT_TEMPLATE
from src.evaluation.evaluator import extract_signature
from src.utils.reasoning import (
    SYSTEM_PROMPT,
    THINK_END_TAG,
    build_signature_user_content,
    extract_code,
    extract_fn_name,
    generate_with_thinking_cap,
)


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


def build_eval_prompts(problems, tokenizer):
    formatted = []
    signatures = []
    for prob in problems:
        prompt = PROMPT_TEMPLATE.format(text=prob["text"])
        expected = prob.get("entry_point") or extract_fn_name(prob["test_cases"])
        signature = extract_signature(prob["test_cases"][0], expected) if prob["test_cases"] else ""
        user_content = build_signature_user_content(prompt, signature)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted.append(fmt)
        signatures.append(signature)
    return formatted, signatures


def budget_forced_generate(llm, prompts, signatures, num_samples,
                           temperature, top_p, max_new_tokens,
                           think_ratio=0.75, seed=1234):
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
                stop=[THINK_END_TAG], include_stop_str_in_output=True, seed=seed + j,
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
        primer = "\n```python\n" + (f"{sig}\n    " if sig else "")
        cont_prompts.append(prompts[i] + closed + primer)
        code_params.append(SamplingParams(
            temperature=temp, top_p=tp, max_tokens=code_budget, n=1,
            stop=["```"], include_stop_str_in_output=False, seed=seed + j,
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
