import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
from src.teacher.deepseek_api import DeepSeekTeacher

MODELS_TO_TRY = [
    "qwen/qwen3-coder:free",
    "deepseek/deepseek-v4-flash:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m2.5:free",
    "arcee-ai/trinity-large-thinking:free",
    "nvidia/nemotron-nano-9b-v2:free",
]

def test_model(model_name):
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set in .env")
        return False
    
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    teacher = DeepSeekTeacher(
        api_key=api_key,
        model=model_name,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=100,
        temperature=0.0,
        top_logprobs=5,
    )
    
    test_prompt = "Write a simple Python function to add two numbers."
    
    try:
        print(f"Sending test prompt...")
        result = teacher.get_response_with_logprobs(test_prompt)
        print(f"✓ SUCCESS! Model {model_name} works")
        print(f"Response: {result['text'][:100]}...")
        return True
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        return False

def update_config(model_name):
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        content = f.read()
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("model:") and not stripped.startswith("model_name:"):
            lines.append(f'  model: "{model_name}"')
        else:
            lines.append(line)
    with open(config_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✓ config.yaml updated with model: {model_name}")


if __name__ == "__main__":
    for model in MODELS_TO_TRY:
        if test_model(model):
            print(f"\n{'='*60}")
            print(f"WORKING MODEL FOUND: {model}")
            print(f"{'='*60}")
            update_config(model)
            break
    else:
        print(f"\n{'='*60}")
        print("ERROR: None of the models work!")
        print(f"{'='*60}")
        sys.exit(1)
        sys.exit(1)
