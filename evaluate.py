import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import yaml
import torch
import argparse

from tqdm import tqdm

from src.data.dataset import create_datasets
from src.evaluation.evaluator import compute_bleu, compute_rouge_l
from src.student.model import StudentModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_responses(
    student: StudentModel,
    prompts: list,
    max_new_tokens: int,
    device: torch.device,
) -> list:
    student.eval()
    responses = []

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Generating"):
            inputs = student.tokenizer(prompt, return_tensors="pt").to(device)
            output_ids = student.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
            )
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            responses.append(student.tokenizer.decode(generated, skip_special_tokens=True))

    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = StudentModel(
        model_name=args.checkpoint,
        max_length=config["student"]["max_length"],
    )
    student.to(device)

    _, val_dataset = create_datasets(config, student.tokenizer)

    prompts = [val_dataset.get_prompt(i) for i in range(len(val_dataset))]
    references = [val_dataset.get_reference(i) for i in range(len(val_dataset))]

    hypotheses = generate_responses(
        student,
        prompts,
        config["evaluation"]["max_new_tokens"],
        device,
    )

    bleu = compute_bleu(references, hypotheses)
    rouge_l = compute_rouge_l(references, hypotheses)

    print(f"BLEU:    {bleu:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")

    eval_dir = Path(config["evaluation"]["output_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    results_path = eval_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"bleu": bleu, "rouge_l": rouge_l, "num_samples": len(prompts)}, f, indent=2)

    print(f"Saved to {results_path}")


if __name__ == "__main__":
    main()
