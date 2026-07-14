import json
import time
from collections import Counter
from pathlib import Path

from src.data.teacher_cache import fully_passed, tail_counts
from src.utils.gpu import gpu_used_gb


def rescore_and_cache(teacher, src_cache_dir: str, dst_cache_dir: str, chunk_size: int = 8) -> None:
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
    print(f"  prompt_logprobs={teacher.top_logprobs}  temperature=1.0  top_p=1.0  (true teacher distribution)")

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
        temperature=1.0, top_p=1.0, max_tokens=1, prompt_logprobs=teacher.top_logprobs,
    )
    cap = teacher.max_model_len - 1
    total_start = time.time()

    for chunk_start in range(0, len(pending), chunk_size):
        chunk = pending[chunk_start:chunk_start + chunk_size]
        token_prompts = []
        metas = []
        enc = teacher.student_tokenizer or teacher.tokenizer
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
                outputs = teacher.llm.generate(token_prompts, score_params, use_tqdm=False)
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
                result = teacher.rescore_tokens(out.prompt_token_ids, out.prompt_logprobs or [], prompt_len)
                payload = {"prompt": d["prompt"], "max_tokens": d.get("max_tokens", teacher.max_tokens), **result}
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
