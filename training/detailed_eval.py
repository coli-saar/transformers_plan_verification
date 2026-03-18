"""
Evaluate best OOD checkpoint on test_id and test_ood splits.

Usage:
    python -m final_model_training.detailed_eval --run_dir path/to/run --csv_path path/to/out.csv
"""

import argparse
import os
import types
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from training.config import load_config
from training.constants import VERDICT_CORRECT_TOKEN, VERDICT_INCORRECT_TOKEN
from training.dataset import build_dataset, load_splits, make_collator
from training.models import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLITS = ["test_id", "test_ood"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def load_model(ckpt_path: str, cfg, tokenizer):
    model_ns = types.SimpleNamespace(
        architecture=cfg.model.architecture,
        n_positions=int(cfg.model.n_positions),
        n_embd=int(cfg.model.n_embd),
        n_layer=int(cfg.model.n_layer),
        n_head=int(cfg.model.n_head),
        rotary_pct=float(cfg.model.rotary_pct),
        rope_theta=float(cfg.model.rope_theta),
        dropout=float(cfg.model.dropout),
        use_parallel_residual=bool(cfg.model.use_parallel_residual),
        model_name_or_path=ckpt_path,
        embd_pdrop=cfg.model.embd_pdrop,
        attn_pdrop=cfg.model.attn_pdrop,
        resid_pdrop=cfg.model.resid_pdrop,
    )
    training_ns = types.SimpleNamespace(gradient_checkpointing=False)
    compat_cfg = types.SimpleNamespace(model=model_ns, training=training_ns)
    return get_model(compat_cfg, tokenizer)


def get_verdict_ids(tokenizer):
    corr = tokenizer.encode(VERDICT_CORRECT_TOKEN, add_special_tokens=False)
    inc = tokenizer.encode(VERDICT_INCORRECT_TOKEN, add_special_tokens=False)
    assert len(corr) == 1 and len(inc) == 1
    return int(corr[0]), int(inc[0])


def evaluate_split(model, tokenizer, cfg, split_name, batch_size, correct_id, incorrect_id):
    candidate_ids = torch.tensor([correct_id, incorrect_id]).to(DEVICE)
    splits = load_splits(cfg.paths.splits_path)
    
    if split_name not in splits:
        print(f"Split '{split_name}' not found, skipping")
        return None

    eval_ds = build_dataset(
        cfg.paths.jsonl_path, tokenizer, split=split_name, splits_path=cfg.paths.splits_path,
        mode="sequence", label_mode="verdict_only", block_size=None, seed=42,
        include_metadata=True, validate_first_n=50,
    )
    collator = make_collator(tokenizer)

    results = []
    with torch.inference_mode():
        for start in range(0, len(eval_ds), batch_size):
            end = min(start + batch_size, len(eval_ds))
            batch_items = [eval_ds[i] for i in range(start, end)]
            batch = collator(batch_items)
            
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            batch_idx, pos = (labels != -100).nonzero(as_tuple=True)
            gold = labels[batch_idx, pos]
            step_logits = logits[batch_idx, pos - 1]
            
            cand_logits = step_logits.index_select(dim=-1, index=candidate_ids)
            probs = torch.softmax(cand_logits, dim=-1)
            pred_idx = cand_logits.argmax(dim=-1)
            pred = candidate_ids[pred_idx]
            
            for i, item in enumerate(batch_items):
                label_is_correct = (gold[i].item() == correct_id)
                pred_correct = (pred[i].item() == correct_id)
                confidence = probs[i, 0].item() if pred_correct else probs[i, 1].item()
                
                results.append({
                    "split": split_name,
                    "plan_length": item.get("plan_len", -1),
                    "label": "correct" if label_is_correct else "incorrect",
                    "correctness": int(pred[i].item() == gold[i].item()),
                    "prediction_confidence": confidence,
                })

    print(f"{split_name}: {sum(r['correctness'] for r in results)}/{len(results)}")
    return results


def main():
    args = parse_args()
    
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "best_val_ood"
    config_path = run_dir / "input_config.yaml"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    cfg = load_config(str(config_path))
    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.tokenizer_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    correct_id, incorrect_id = get_verdict_ids(tokenizer)
    
    print(f"Loading model from {ckpt_path}")
    model = load_model(str(ckpt_path), cfg, tokenizer).to(DEVICE)
    model.eval()
    
    all_results = []
    for split in SPLITS:
        results = evaluate_split(model, tokenizer, cfg, split, args.batch_size, correct_id, incorrect_id)
        if results:
            all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    df.to_csv(args.csv_path, index=False)
    print(f"Saved {len(df)} results to {args.csv_path}")


if __name__ == "__main__":
    main()
