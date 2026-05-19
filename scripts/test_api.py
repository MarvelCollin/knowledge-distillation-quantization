import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import yaml
from dotenv import load_dotenv

from src.teacher.deepseek_api import DeepSeekTeacher


def test_deepseek_api():
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")

    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set in .env")
        return False

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = config["teacher"]["model"]
    api_base = config["teacher"]["api_base"]

    print(f"Testing model: {model}")
    print(f"API base: {api_base}")
    print(f"Key: {api_key[:15]}...")

    teacher = DeepSeekTeacher(
        api_key=api_key,
        model=model,
        api_base=api_base,
        max_tokens=100,
        temperature=0.0,
        top_logprobs=5,
    )

    test_prompt = "Write a simple Python function to add two numbers."

    try:
        print(f"\nSending test prompt...")
        result = teacher.get_response_with_logprobs(test_prompt)
        print(f"✓ API call succeeded!")
        print(f"Response: {result['text'][:100]}...")
        print(f"Tokens with logprobs: {len(result['tokens'])}")
        return True
    except Exception as e:
        print(f"✗ API call failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_deepseek_api()
    exit(0 if success else 1)
