import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import yaml
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

MODELS = [
    {
        "name": "DeepSeek V4 Flash (OpenRouter)",
        "model": "deepseek/deepseek-v4-flash:free",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "headers": {
            "HTTP-Referer": "https://github.com/MarvelCollin/knowledge-distillation-quantization",
            "X-Title": "Knowledge Distillation Quantization",
        },
    },
    {
        "name": "Google Gemini 2.0 Flash",
        "model": "gemini-2.0-flash",
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "TEACHER_API_KEY",
        "headers": {},
    },
    {
        "name": "Google Gemini 2.0 Flash Lite",
        "model": "gemini-2.0-flash-lite",
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "TEACHER_API_KEY",
        "headers": {},
    },
]

CYAN  = "\033[0;36m"
GREEN = "\033[0;32m"
RED   = "\033[0;31m"
YELLOW = "\033[1;33m"
BOLD  = "\033[1m"
NC    = "\033[0m"


def test_model(info: dict) -> tuple[bool, str]:
    api_key = os.environ.get(info["api_key_env"])
    if not api_key:
        return False, f"API key not set ({info['api_key_env']})"
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=info["api_base"],
            default_headers=info["headers"],
        )
        resp = client.chat.completions.create(
            model=info["model"],
            messages=[{"role": "user", "content": "Reply with only the word OK."}],
            max_tokens=5,
        )
        return True, resp.choices[0].message.content.strip()
    except RateLimitError:
        return False, "Rate limited (quota exhausted)"
    except Exception as e:
        msg = str(e)
        return False, msg[:100]


def update_config(info: dict) -> None:
    config_path = Path("/workspace/config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["teacher"]["model"]   = info["model"]
    config["teacher"]["api_base"] = info["api_base"]
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def main():
    load_dotenv()

    print(f"\n{BOLD}{CYAN}{'─'*60}{NC}")
    print(f"{BOLD}  Checking available teacher models...{NC}")
    print(f"{BOLD}{CYAN}{'─'*60}{NC}\n")

    working = []
    for info in MODELS:
        print(f"  Testing {BOLD}{info['name']}{NC}")
        print(f"  Model : {info['model']}")
        ok, msg = test_model(info)
        if ok:
            print(f"  {GREEN}✓ WORKING{NC}  response: {msg}\n")
            working.append(info)
        else:
            print(f"  {RED}✗ FAILED {NC}  {msg}\n")

    print(f"{BOLD}{CYAN}{'─'*60}{NC}")
    print(f"  {GREEN}{len(working)}{NC}/{len(MODELS)} models available.\n")

    if not working:
        print(f"  {RED}No working models found. Check your API keys and quota.{NC}\n")
        sys.exit(1)

    if len(working) == 1:
        chosen = working[0]
        print(f"  Auto-selecting: {BOLD}{chosen['name']}{NC}\n")
    else:
        print(f"  {BOLD}Select a model to use as teacher:{NC}")
        for i, m in enumerate(working):
            print(f"    {i + 1}) {m['name']}  ({m['model']})")
        print()
        while True:
            try:
                choice = input("  Enter number: ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(working):
                    chosen = working[idx]
                    break
                print("  Invalid choice, try again.")
            except (ValueError, KeyboardInterrupt):
                print("\n  Cancelled.")
                sys.exit(1)

    print(f"\n  {GREEN}✓ Selected:{NC} {BOLD}{chosen['name']}{NC}")
    update_config(chosen)
    print(f"  config.yaml updated.\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
