from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"Model {MODEL_NAME} downloaded successfully.")
