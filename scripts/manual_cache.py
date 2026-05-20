import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import yaml
from datasets import load_dataset
from src.data.dataset import PROMPT_TEMPLATE

CYAN   = "\033[0;36m"
GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
RED    = "\033[0;31m"
BOLD   = "\033[1m"
NC     = "\033[0m"


def load_config() -> dict:
    with open("/workspace/config/config.yaml") as f:
        return yaml.safe_load(f)


def build_prompts(config: dict) -> list:
    dataset_name = config["data"]["dataset_name"]
    max_samples = config["data"]["max_samples"]

    raw = load_dataset(dataset_name, split="train")
    raw = raw.select(range(min(max_samples, len(raw))))

    result = []
    for item in raw:
        test_cases = [t for t in item.get("test_list", []) if isinstance(t, str) and t.strip()]
        if not test_cases or not item.get("code", "").strip():
            continue
        result.append({
            "idx": len(result),
            "text": item["text"].strip(),
            "prompt": PROMPT_TEMPLATE.format(text=item["text"].strip()),
        })
    return result


def is_cached_valid(cache_dir: Path, idx: int, prompt: str) -> bool:
    f = cache_dir / f"{idx}.json"
    if not f.exists():
        return False
    try:
        d = json.loads(f.read_text())
        return d.get("prompt", "") == prompt
    except Exception:
        return False


def save_entry(cache_dir: Path, idx: int, prompt: str, text: str) -> None:
    entry = {
        "prompt": prompt,
        "text": text,
        "tokens": [],
        "logprobs": [],
    }
    (cache_dir / f"{idx}.json").write_text(json.dumps(entry, indent=2))


def read_multiline() -> str:
    print(f"  {YELLOW}Paste your answer. Type {BOLD}---END---{NC}{YELLOW} on its own line when done:{NC}")
    print()
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "---END---":
            break
        lines.append(line)
    return "\n".join(lines)


def print_divider():
    print(f"{BOLD}{CYAN}{'─' * 60}{NC}")


def print_prompt(item: dict) -> None:
    print()
    print_divider()
    print(f"  {BOLD}Prompt #{item['idx']}{NC}  —  {item['text'][:70]}")
    print_divider()
    print(item["prompt"])
    print_divider()
    print()


def select_target(uncached: list) -> list:
    print(f"\n  {BOLD}Uncached prompts:{NC}\n")
    for i, item in enumerate(uncached):
        print(f"    {i + 1:>3}) [#{item['idx']:>3}]  {item['text'][:65]}")
    print()
    sel = input("  Enter number: ").strip()
    try:
        return [uncached[int(sel) - 1]]
    except (ValueError, IndexError):
        print(f"\n  {RED}Invalid selection.{NC}\n")
        return []


def main():
    config = load_config()
    cache_dir = Path(config["data"]["teacher_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    print()
    print_divider()
    print("  Loading dataset...")

    prompts = build_prompts(config)
    uncached = [p for p in prompts if not is_cached_valid(cache_dir, p["idx"], p["prompt"])]

    print(f"  {len(uncached)}/{len(prompts)} prompts not yet cached.")
    print_divider()
    print()

    if not uncached:
        print(f"  {GREEN}All prompts are already cached.{NC}\n")
        return

    print(f"  {BOLD}Options:{NC}")
    print("  a) Enter answers for all uncached prompts")
    print("  s) Select a specific prompt by number")
    print("  q) Quit")
    print()

    choice = input("  Select: ").strip().lower()

    if choice == "q":
        return

    if choice == "a":
        targets = uncached
    elif choice == "s":
        targets = select_target(uncached)
        if not targets:
            return
    else:
        print(f"\n  {RED}Invalid option.{NC}\n")
        return

    saved = 0
    for item in targets:
        print_prompt(item)
        text = read_multiline()
        print()

        if not text.strip():
            print(f"  {YELLOW}Skipped (empty input).{NC}\n")
            continue

        save_entry(cache_dir, item["idx"], item["prompt"], text)
        saved += 1
        print(f"  {GREEN}✓ Saved #{item['idx']} → {cache_dir}/{item['idx']}.json{NC}\n")

    print_divider()
    print(f"  {GREEN}Done. {saved} response(s) cached.{NC}")
    print_divider()
    print()


if __name__ == "__main__":
    main()
