import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class StudentModel(nn.Module):
    def __init__(self, model_name: str, max_length: int):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return outputs.logits

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
