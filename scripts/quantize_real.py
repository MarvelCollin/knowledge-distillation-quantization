import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from probe_quant_grid import probe as probe_quant_grid
from src.config import load_config
from src.data.problems import build_user_content, load_problems
from src.data.teacher_cache import load_passing_responses
from src.utils.reasoning import SYSTEM_PROMPT

def build_calibration(config: dict, tokenizer, n_samples: int, max_len: int) -> list:
    problems = load_problems(config)
    cache_dir = config["data"]["teacher_cache_dir"]
    n_train = int(len(problems) * config["data"]["train_ratio"])
    passing = load_passing_responses(cache_dir, n_train)

    texts, skipped = [], 0
    for idx in sorted(passing):
        if len(texts) >= n_samples:
            break
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(problems[idx])},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        seq = prompt + passing[idx]["text"]
        if len(tokenizer(seq)["input_ids"]) > max_len:
            skipped += 1
            continue
        texts.append(seq)

    if skipped:
        print(f"calibration: skipped {skipped} traces longer than --calib-max-len {max_len} "
              "(kept only complete prompt+reasoning+code sequences)")
    if not texts:
        raise SystemExit(
            f"No passing teacher traces found under {cache_dir} for the first {n_train} "
            "problems. Calibration needs them; check the cache path in config."
        )
    print(f"calibration: {len(texts)} sequences from the train split "
          f"(requested {n_samples}, {len(passing)} passing traces available)")
    return texts

def build_recipe(method: str, bits: int, group_size: int, ignore: list):
    scheme = f"W{bits}A16"
    if method == "gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier
        return GPTQModifier(targets="Linear", scheme=scheme, ignore=ignore)
    if method == "awq":
        try:
            from llmcompressor.modifiers.awq import AWQModifier
        except ImportError as e:
            raise SystemExit(
                "AWQModifier not found in the installed llm-compressor. Upgrade it, or run "
                "--method gptq first -- one calibrated method is enough to test the claim."
            ) from e
        return AWQModifier(targets="Linear", scheme=scheme, ignore=ignore)
    from llmcompressor.modifiers.quantization import QuantizationModifier
    return QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore)

def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9

_QUANT_SUFFIXES = (
    "weight_scale", "weight_zero_point", "weight_g_idx", "weight_shape", "weight_packed",
    "input_scale", "input_zero_point",
)

def strip_quant_params(out: Path) -> int:
    from safetensors.torch import load_file, save_file

    removed = 0
    for shard in sorted(out.glob("*.safetensors")):
        tensors = load_file(str(shard))
        keep = {k: v for k, v in tensors.items() if not k.endswith(_QUANT_SUFFIXES)}
        if len(keep) != len(tensors):
            removed += len(tensors) - len(keep)
            save_file(keep, str(shard), metadata={"format": "pt"})

    index = out / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text())
        idx["weight_map"] = {k: v for k, v in idx["weight_map"].items()
                             if not k.endswith(_QUANT_SUFFIXES)}
        index.write_text(json.dumps(idx, indent=2))
    return removed

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Source checkpoint (HF id or local dir).")
    ap.add_argument("--out", required=True, help="Output dir for the compressed checkpoint.")
    ap.add_argument("--method", required=True, choices=["gptq", "awq", "rtn"])
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--group-size", type=int, default=128,
                    help="Informational: the W4A16/W8A16 presets are group-128 symmetric, "
                         "matching quantize_int8.py. A different value needs a custom scheme.")
    ap.add_argument("--calib-samples", type=int, default=256)
    ap.add_argument("--calib-max-len", type=int, default=2048)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--save-mode", choices=["fake", "compressed"], default="fake",
                    help="fake (default): dequantize back to bf16 and save a plain checkpoint")
    args = ap.parse_args()

    if args.group_size != 128:
        print(f"WARNING: --group-size {args.group_size} is not applied; the {args.bits}-bit "
              "preset is group-128. Comparability with the RTN grid requires 128.")

    try:
        from llmcompressor import oneshot
    except ImportError:
        from llmcompressor.transformers import oneshot
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    texts = build_calibration(config, tokenizer, args.calib_samples, args.calib_max_len)
    # attention_mask is required: llm-compressor's run_calibration_forward calls
    # apply_pad_mask_to_batch, which zeroes padded positions via input_ids * attention_mask
    # and raises KeyError without it.
    ds = Dataset.from_list([
        {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        for enc in (tokenizer(t, max_length=args.calib_max_len, truncation=True) for t in texts)
    ])

    print(f"loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map={"": 0}, trust_remote_code=True)

    recipe = build_recipe(args.method, args.bits, args.group_size, ignore=["lm_head"])
    print(f"running {args.method.upper()} W{args.bits}A16 oneshot "
          f"({len(ds)} calibration sequences, max_len={args.calib_max_len})...")
    oneshot_kwargs = dict(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.calib_max_len,
        num_calibration_samples=len(ds),
    )
    if args.save_mode == "compressed":
        oneshot_kwargs["output_dir"] = args.out
    oneshot(**oneshot_kwargs)

    if args.save_mode == "fake":
        print("saving dequantized (fake-quant) bf16 checkpoint...")
        model.save_pretrained(args.out, save_compressed=False)
    tokenizer.save_pretrained(args.out)
    try:
        AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True, use_fast=False
        ).save_vocabulary(str(args.out))
        print("saved legacy vocab.json/merges.txt (needed by tools that force a slow "
              "tokenizer, e.g. evalplus)")
    except Exception as e:
        print(f"WARNING: could not save legacy tokenizer vocab files ({e}); tools "
              "requiring a slow tokenizer may fail")

    out = Path(args.out)
    template = getattr(tokenizer, "chat_template", None)
    tok_cfg_path = out / "tokenizer_config.json"
    if template and tok_cfg_path.exists():
        tok_cfg = json.loads(tok_cfg_path.read_text())
        if not tok_cfg.get("chat_template"):
            tok_cfg["chat_template"] = template
            tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2))
            print("embedded chat_template into tokenizer_config.json for the older eval stack")
    elif not template:
        print("WARNING: source tokenizer has no chat_template; eval prompts will not build.")

    cfg_path = out / "config.json"
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text())
        if args.save_mode == "fake":
            if raw.pop("quantization_config", None) is not None:
                cfg_path.write_text(json.dumps(raw, indent=2))
                print("removed stale quantization_config; checkpoint is plain bf16")
            n = strip_quant_params(out)
            if n:
                print(f"stripped {n} quantization-parameter tensors (scales, zero points)")
        else:
            sparsity = raw.get("quantization_config", {}).get("sparsity_config")
            if isinstance(sparsity, dict) and not sparsity.get("format"):
                raw["quantization_config"].pop("sparsity_config")
                cfg_path.write_text(json.dumps(raw, indent=2))
                print("dropped empty sparsity_config from config.json for the older eval stack")

    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    qcfg = cfg.get("quantization_config", {})
    print("\n=== verification ===")
    print(f"method            {args.method.upper()} W{args.bits}A16, lm_head excluded")
    print(f"save mode         {args.save_mode}")
    print(f"output            {out}")
    print(f"on-disk size      {dir_size_gb(out):.2f} GB")
    if args.save_mode == "fake":
        print("quant config      absent by design (dequantized bf16, loads as a normal model)")
    else:
        print(f"quant config      {'present' if qcfg else 'MISSING -- checkpoint is not compressed'}")
    if qcfg:
        print(f"  format          {qcfg.get('format')}")
        print(f"  quant_method    {qcfg.get('quant_method')}")

    # A fake-quant checkpoint is stored as bf16, so nothing about its dtype or config proves
    # the weights were actually rounded. Count distinct values inside one quantization group
    # and fail loudly if they are not on the expected grid. QuantizationModifier (--method rtn)
    # only attaches scales and defers rounding to save_compressed=True, so under --save-mode
    # fake it silently emits the unquantized input; that produced a plausible-looking but
    # invalid eval on 2026-08-23 before this gate existed.
    if args.save_mode == "fake":
        grid_group = 128
        max_expected = 2 ** args.bits
        if max_expected < grid_group:
            _, _, counts = probe_quant_grid(str(out), rows=6, group=grid_group)
            print(f"grid check        uniq/group={counts} (expected <= {max_expected})")
            if max(counts) > max_expected:
                raise SystemExit(
                    f"\nFAILED: {out} is not on a W{args.bits} group-{grid_group} grid.\n"
                    f"Found up to {max(counts)} distinct values per group, expected at most "
                    f"{max_expected}.\nThe weights were never quantized, so any evaluation of "
                    f"this checkpoint is invalid.\n"
                    f"Known cause: --method rtn with --save-mode fake is a no-op on current "
                    f"llm-compressor.\nUse scripts/quantize_int8.py --bits 4 --group-size 128 "
                    f"for a simulated RTN checkpoint instead.")
            print("grid check        PASS (weights really are on the low-bit grid)")

    print("\nNext: gate this checkpoint before evaluating it.")
    print(f"  docker compose run --rm compare_eval \\")
    print(f"    python scripts/sanity_generate.py --model {out} \\")
    print(f"    --dtype bfloat16 --max-tokens 800 --chars 1500")

if __name__ == "__main__":
    main()
