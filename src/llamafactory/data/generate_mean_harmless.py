"""
Compute mean hidden-state representation (mu) for harmless prompts.

Input JSON format:
[
  {"instruction": "...", "category": null},
  ...
]

Output:
- mu tensor saved as .pt (shape: [H], dtype=float32)
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def masked_mean_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden:    [B, L, H]
    attention_mask: [B, L] (0/1)
    return:         [B, H]
    """
    mask_f = attention_mask.to(last_hidden.dtype).unsqueeze(-1)     # [B, L, 1]
    denom = mask_f.sum(dim=1).clamp_min(1.0)                        # [B, 1]
    return (last_hidden * mask_f).sum(dim=1) / denom                # [B, H]


@torch.no_grad()
def compute_mu(
    instructions: List[str],
    model_id_or_path: str,
    batch_size: int,
    max_length: int,
    device: str,
    use_bf16: bool,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    # 对于某些模型，pad_token 可能缺失；这里将其设为 eos_token 以便 batch padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map=None,
    ).to(device)
    model.eval()

    # 累加均值：sum_repr / count
    sum_repr = None
    count = 0

    for i in tqdm(range(0, len(instructions), batch_size)):
        batch_text = instructions[i : i + batch_size]

        enc = tokenizer(
            batch_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # last_hidden: [B, L, H]
        last_hidden = out.hidden_states[-1]
        # prompt repr: [B, H]
        batch_repr = masked_mean_last_hidden(last_hidden, attention_mask)  # [B, H]

        # 为稳定起见，用 float32 累加
        batch_repr_f32 = batch_repr.to(torch.float32).detach().cpu()       # [B, H]

        if sum_repr is None:
            sum_repr = batch_repr_f32.sum(dim=0)                           # [H]
        else:
            sum_repr += batch_repr_f32.sum(dim=0)                          # [H]
        count += batch_repr_f32.size(0)

    assert sum_repr is not None and count > 0
    mu = sum_repr / float(count)                                          # [H], float32 on CPU
    return mu

# src/llamafactory/data/generate_mean_harmless.py --model_id_or_path=/ds/models/llms/Llama-3.1-8B-Instruct --model_name=llama31-8b
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id_or_path", type=str, required=True, help="HF model path or local path")
    parser.add_argument("--model_name", type=str, required=True, help="model name")
    parser.add_argument("--input_dir", type=str, default="/netscratch/dshi/projects/Multilingual-Refusal/dataset/splits_multi", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="/netscratch/dshi/projects/MPO/data/mean_harmless", help="Dir to save mu tensor (.pt)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (recommended on A100/H100)")
    args = parser.parse_args()

    def compute(input_path, lan):
        data: List[Dict[str, Any]] = json.loads(input_path.read_text(encoding="utf-8"))

        instructions = []
        for ex in data:
            inst = ex.get("instruction", None)
            if isinstance(inst, str) and inst.strip():
                instructions.append(inst.strip())

        if len(instructions) == 0:
            raise ValueError("No valid 'instruction' strings found.")

        mu = compute_mu(
            instructions=instructions,
            model_id_or_path=args.model_id_or_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            use_bf16=args.bf16,
        )

        out_path = Path(f"{args.output_dir}/mu_{args.model_name}_{lan}.pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mu, out_path)
        print(f"[OK] Saved mu: shape={tuple(mu.shape)}, dtype={mu.dtype} to {out_path}")

    # en
    input_path = "/netscratch/dshi/projects/Multilingual-Refusal/dataset/splits/harmless_train_200_sampled.json"
    input_path = Path(input_path)
    compute(input_path, "en")

    # other languages
    all_languages = ['zh', 'ja', 'ko', 'ar', 'bn', 'sw']
    for lan in all_languages:
        input_path = f"{args.input_dir}/harmless_train_translated_{lan}.json"
        input_path = Path(input_path)
        compute(input_path, lan)


if __name__ == "__main__":
    main()
