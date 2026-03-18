from __future__ import annotations

import argparse
import os
import random
import shutil
import types
from pathlib import Path

from training.constants import VERDICT_CORRECT_TOKEN, VERDICT_INCORRECT_TOKEN, HF_HOME

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = HF_HOME

from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint, set_seed
from torch.utils.data import DataLoader, Subset

from training.config import load_config
from training.models import ComputeAccuracy, get_model
from training.dataset import (
    PackedLMDataset,
    RepackEachEpochCallback,
    build_dataset,
    build_eval_dataloader,
    make_collator,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GPT-NeoX on verdict-at-end JSONL using a YAML config")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return p.parse_args()


def _rank_from_env() -> int:
    """Best-effort rank detection before Trainer initializes distributed."""
    rank_str = os.environ.get("RANK") or os.environ.get("SLURM_PROCID")
    if rank_str is None:
        return 0
    try:
        return int(rank_str)
    except Exception:
        return 0


def _is_rank0() -> bool:
    return _rank_from_env() == 0


def _load_local_tokenizer(tokenizer_dir: str) -> AutoTokenizer:
    p = Path(tokenizer_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"Tokenizer directory not found: {p}. "
            "Note: if this path doesn't exist, HF will treat it like a Hub repo id."
        )
    tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        # Make padding explicit for Trainer + attention_mask logic.
        tok.pad_token = tok.eos_token
    return tok


def _verdict_token_ids(tokenizer) -> tuple[int, int]:
    corr = tokenizer.encode(VERDICT_CORRECT_TOKEN, add_special_tokens=False)
    inc = tokenizer.encode(VERDICT_INCORRECT_TOKEN, add_special_tokens=False)
    if len(corr) != 1 or len(inc) != 1:
        raise ValueError(
            "Verdict tokens must each be a single token. "
            f"Got correct={corr}, incorrect={inc}. "
            "Make sure your tokenizer includes ' correct' and ' incorrect' as single tokens."
        )
    return int(corr[0]), int(inc[0])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Seed
    set_seed(int(cfg.seed))

    # Ensure output_dir exists early (get_last_checkpoint expects it).
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional W&B
    if cfg.wandb.project:
        os.environ["WANDB_PROJECT"] = cfg.wandb.project
        if cfg.wandb.entity:
            os.environ["WANDB_ENTITY"] = cfg.wandb.entity

    # Tokenizer
    tokenizer = _load_local_tokenizer(cfg.paths.tokenizer_dir)
    correct_id, incorrect_id = _verdict_token_ids(tokenizer)
    if correct_id == incorrect_id:
        raise ValueError("Internal error: correct and incorrect verdict token IDs are identical")

    # Data
    if cfg.data.block_size is not None and int(cfg.model.n_positions) < int(cfg.data.block_size):
        raise ValueError(
            f"model.n_positions ({cfg.model.n_positions}) must be >= data.block_size ({cfg.data.block_size})"
        )

    train_ds = build_dataset(
        cfg.paths.jsonl_path,
        tokenizer,
        split="train",
        splits_path=cfg.paths.splits_path,
        mode=cfg.data.train_mode,
        label_mode=cfg.data.train_label_mode,
        block_size=cfg.data.block_size,
        seed=int(cfg.seed),
        include_metadata=False,
        validate_first_n=50,
        allow_cross_sample_attention=cfg.data.allow_cross_sample_attention,
        reset_pos_id=cfg.data.reset_pos_id,
    )

    collator = make_collator(tokenizer)

    # Eval dataloaders: build only on rank 0 to avoid extra work; other ranks will pause at barriers.
    eval_loaders = {}
    if _is_rank0():
        # Train subset accuracy (sequence mode + verdict_only => exactly one supervised token per sample)
        train_eval_ds = build_dataset(
            cfg.paths.jsonl_path,
            tokenizer,
            split="train",
            splits_path=cfg.paths.splits_path,
            mode="sequence",
            label_mode="verdict_only",
            block_size=None,
            seed=int(cfg.seed),
            include_metadata=False,
            validate_first_n=50,
        )
        subset_n = int(cfg.eval.train_subset_size)
        if subset_n > 0 and len(train_eval_ds) > subset_n:
            rng = random.Random(int(cfg.seed))
            idxs = list(range(len(train_eval_ds)))
            rng.shuffle(idxs)
            train_eval_ds = Subset(train_eval_ds, idxs[:subset_n])

        eval_loaders["train"] = DataLoader(
            train_eval_ds,
            batch_size=int(cfg.data.per_device_eval_batch_size),
            shuffle=False,
            num_workers=int(cfg.data.eval_num_workers),
            collate_fn=collator,
        )

        # Additional eval splits (ID/OOD etc) taken from cfg.eval.splits.
        eval_splits = list(cfg.eval.splits or [])
        if "train" in eval_splits:
            raise ValueError("eval.splits must not include 'train' (reserved for train_subset eval loader)")
        if len(set(eval_splits)) != len(eval_splits):
            raise ValueError(f"eval.splits contains duplicates: {eval_splits}")

        for split_name in eval_splits:
            eval_loaders[str(split_name)] = build_eval_dataloader(
                cfg.paths.jsonl_path,
                tokenizer,
                split=str(split_name),
                splits_path=cfg.paths.splits_path,
                label_mode="verdict_only",
                batch_size=int(cfg.data.per_device_eval_batch_size),
                num_workers=int(cfg.data.eval_num_workers),
            )

    # Model config object for `final_model_training.models.get_model`
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
        model_name_or_path=cfg.model.model_name_or_path,
        embd_pdrop=cfg.model.embd_pdrop,
        attn_pdrop=cfg.model.attn_pdrop,
        resid_pdrop=cfg.model.resid_pdrop,
    )
    training_ns = types.SimpleNamespace(gradient_checkpointing=bool(cfg.train.gradient_checkpointing))
    compat_cfg = types.SimpleNamespace(model=model_ns, training=training_ns)
    model = get_model(compat_cfg, tokenizer)

    # HF TrainingArguments
    report_to = ["wandb"] if cfg.wandb.project else []
    if cfg.train.bf16 and cfg.train.fp16:
        raise ValueError("train.bf16 and train.fp16 cannot both be true")

    training_args = TrainingArguments(
        output_dir=str(cfg.paths.output_dir),
        overwrite_output_dir=bool(cfg.train.overwrite_output_dir),
        per_device_train_batch_size=int(cfg.train.per_device_train_batch_size),
        per_device_eval_batch_size=int(cfg.data.per_device_eval_batch_size),
        gradient_accumulation_steps=int(cfg.train.gradient_accumulation_steps),
        optim=str(cfg.train.optim),
        num_train_epochs=float(cfg.train.num_train_epochs),
        max_steps=int(cfg.train.max_steps),
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        adam_beta1=float(cfg.train.adam_beta1),
        adam_beta2=float(cfg.train.adam_beta2),
        adam_epsilon=float(cfg.train.adam_epsilon),
        warmup_steps=int(cfg.train.warmup_steps),
        warmup_ratio=float(cfg.train.warmup_ratio),
        lr_scheduler_type=str(cfg.train.lr_scheduler_type),
        lr_scheduler_kwargs=(dict(cfg.train.lr_scheduler_kwargs) if cfg.train.lr_scheduler_kwargs else None),
        logging_steps=int(cfg.train.logging_steps),
        save_steps=int(cfg.train.save_steps),
        save_total_limit=int(cfg.train.save_total_limit),
        eval_strategy="no",
        fp16=bool(cfg.train.fp16),
        bf16=bool(cfg.train.bf16),
        gradient_checkpointing=bool(cfg.train.gradient_checkpointing),
        max_grad_norm=float(cfg.train.max_grad_norm),
        report_to=report_to,
        run_name=(cfg.wandb.run_name or None),
        seed=int(cfg.seed),
    )

    callbacks = [
        ComputeAccuracy(
            eval_loaders=eval_loaders,
            eval_interval=int(cfg.eval.eval_interval),
            candidate_token_ids=[correct_id, incorrect_id],
            output_dir=str(out_dir),
        )
    ]
    if isinstance(train_ds, PackedLMDataset):
        callbacks.append(RepackEachEpochCallback(train_ds))

    os.makedirs(str(out_dir), exist_ok=True)
    shutil.copy(args.config, out_dir / "input_config.yaml")
    
    resume_from_checkpoint = None
    if cfg.train.resume:
        ckpt = get_last_checkpoint(str(out_dir))
        if ckpt:
            resume_from_checkpoint = ckpt

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final artifacts
    if trainer.is_world_process_zero():
        trainer.save_model()
        tokenizer.save_pretrained(str(cfg.paths.output_dir))
        trainer.save_state()


if __name__ == "__main__":
    main()


