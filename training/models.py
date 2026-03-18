"""Model architectures for plan verification."""

import os
import types
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GPTNeoXConfig, GPTNeoXForCausalLM, TrainerCallback
import wandb  


def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_rank() -> int:
    return int(torch.distributed.get_rank()) if _dist_is_initialized() else 0


def _dist_world_size() -> int:
    return int(torch.distributed.get_world_size()) if _dist_is_initialized() else 1


def _dist_barrier() -> None:
    if _dist_is_initialized():
        torch.distributed.barrier()


def _get_model_device(model) -> torch.device:
    """Safely retrieve the device backing the (possibly wrapped) model."""
    return next(model.parameters()).device


def resize_model_embeddings(model, tokenizer):
    '''
    Resize model embeddings to match tokenizer vocab size.
    This should be called after adding new tokens to the tokenizer.
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        
    Returns:
        Model with resized embeddings
    '''
    if len(tokenizer) != model.config.vocab_size:
        print(f'Resizing model embeddings: {model.config.vocab_size} -> {len(tokenizer)}')
        model.resize_token_embeddings(len(tokenizer))
        # Update config
        model.config.vocab_size = len(tokenizer)
    
    return model


def _load_weights_if_available(model, ckpt_dir: str) -> bool:
    '''Load state_dict from a local checkpoint directory if present.'''
    bin_path = os.path.join(ckpt_dir, 'pytorch_model.bin')
    safetensors_path = os.path.join(ckpt_dir, 'model.safetensors')
    if os.path.isfile(bin_path):
        print(f'Loading weights from: {bin_path}')
        state_dict = torch.load(bin_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'Warning: missing keys: {len(missing)}')
            ape_missing = [k for k in missing if k.startswith('ape_')]
            if ape_missing:
                print('Warning: APE parameters not found in checkpoint; reinitializing:', ape_missing)
        if unexpected:
            print(f'Warning: unexpected keys: {len(unexpected)}')
        return True
    if os.path.isfile(safetensors_path):
        print(f'Loading weights from: {safetensors_path}')
        try:
            from safetensors.torch import load_file as safe_load_file
            state_dict = safe_load_file(safetensors_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f'Warning: missing keys: {len(missing)}')
                ape_missing = [k for k in missing if k.startswith('ape_')]
                if ape_missing:
                    print('Warning: APE parameters not found in checkpoint; reinitializing:', ape_missing)
            if unexpected:
                print(f'Warning: unexpected keys: {len(unexpected)}')
            return True
        except Exception as e:
            print(f'Failed to load safetensors: {e}')
            return False
    return False


def get_model(config, tokenizer):
    '''
    Create or load a model based on configuration.
    
    Uses GPT-NeoX as base architecture with different positional encodings:
        - gpt-neox-rope: GPT-NeoX with RoPE (Rotary Positional Encoding)
        - gpt-neox-ape: GPT-NeoX with APE (Absolute/Learned Positional Encoding)
        - gpt-neox-nope: GPT-NeoX with NoPE (No Positional Encoding)
    
    All variants use Pre-LayerNorm for better training stability.
    Set use_parallel_residual=False for sequential blocks (like Pre-LN GPT-2),
    or use_parallel_residual=True for parallel blocks (like GPT-J).
    
    Args:
        config: Configuration object
        tokenizer: Tokenizer (for vocab size)
        
    Returns:
        Model instance
    '''
    arch = config.model.architecture.lower()
    if arch.startswith('gpt-neox-'):
        # Always initialize the requested variant first (ensures hooks like APE are set)
        pos_encoding = arch[len("gpt-neox-"):]
        if pos_encoding not in {"rope", "ape", "nope"}:
            raise ValueError(f"Unknown GPT-NeoX variant: {arch}")
        model = create_gptneox_model(config, tokenizer, pos_encoding=pos_encoding)
        # Optionally load weights into this initialized model
        if config.model.model_name_or_path:
            loaded = _load_weights_if_available(model, config.model.model_name_or_path)
            if not loaded:
                print('No local weights file found; training from scratch with this architecture.')
    else:
        # Non-NeoX architectures (if any) can rely on HF loader
        if config.model.model_name_or_path is None:
            raise ValueError(f'Invalid model architecture: {config.model.architecture}. '
                             f'Supported: gpt-neox-rope, gpt-neox-ape, gpt-neox-nope')
        print(f'Loading pretrained model from: {config.model.model_name_or_path}')
        model = AutoModelForCausalLM.from_pretrained(config.model.model_name_or_path)

    model = resize_model_embeddings(model, tokenizer)
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print('Gradient checkpointing enabled')
    
    return model


def create_gptneox_model(config, tokenizer, pos_encoding: str):
    '''
    Minimal factory for GPT-NeoX with positional encoding variant.
    pos_encoding: 'rope' | 'ape' | 'nope'
    '''
    assert pos_encoding in {'rope', 'ape', 'nope'}
    # Handle potential None values (e.g., YAML `null`)
    use_parallel_residual = getattr(config.model, 'use_parallel_residual', None)
    use_parallel_residual = bool(use_parallel_residual) if use_parallel_residual is not None else False
    rotary_pct = getattr(config.model, 'rotary_pct', None)
    rotary_pct = float(rotary_pct) if rotary_pct is not None else 0.25
    rope_theta = getattr(config.model, 'rope_theta', None)
    rope_theta = float(rope_theta) if rope_theta is not None else 10000.0
    rope_scaling = getattr(config.model, 'rope_scaling', None)
    # Dropouts with backward-compatible fallbacks (handle None explicitly)
    dropout_fallback = float(getattr(config.model, 'dropout', 0.0) or 0.0)
    embd_pdrop = getattr(config.model, 'embd_pdrop', None)
    embd_pdrop = float(embd_pdrop) if embd_pdrop is not None else dropout_fallback
    attn_pdrop = getattr(config.model, 'attn_pdrop', None)
    attn_pdrop = float(attn_pdrop) if attn_pdrop is not None else dropout_fallback
    resid_pdrop = getattr(config.model, 'resid_pdrop', None)
    resid_pdrop = float(resid_pdrop) if resid_pdrop is not None else dropout_fallback

    rotary_pct_cfg = rotary_pct if pos_encoding == 'rope' else 0.0

    model_config_kwargs = dict(
        vocab_size=len(tokenizer),
        max_position_embeddings=config.model.n_positions,
        hidden_size=config.model.n_embd,
        num_hidden_layers=config.model.n_layer,
        num_attention_heads=config.model.n_head,
        intermediate_size=config.model.n_embd * 4,
        rotary_pct=rotary_pct_cfg,
        rotary_emb_base=rope_theta,
        use_parallel_residual=use_parallel_residual,
        attention_dropout=attn_pdrop,
        hidden_dropout=resid_pdrop,
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        use_cache=False,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
    )
    if rope_scaling is not None:
        model_config_kwargs["rope_scaling"] = rope_scaling
    model_config = GPTNeoXConfig(**model_config_kwargs)

    model = GPTNeoXForCausalLM(model_config)

    if pos_encoding == 'ape':
        # Minimal learned positional embeddings added at the model.forward level
        max_positions = config.model.n_positions
        hidden_size = config.model.n_embd
        pe = torch.nn.Embedding(max_positions, hidden_size)
        torch.nn.init.normal_(pe.weight, std=0.02)
        drop = torch.nn.Dropout(p=embd_pdrop)
        # register so it saves/loads with state_dict (attach at top level for clarity)
        model.ape_pe = pe
        model.ape_drop = drop

        orig_embed = model.get_input_embeddings()
        orig_forward = model.forward

        def forward_with_ape(self,
                             input_ids=None,
                             attention_mask=None,
                             position_ids=None,
                             inputs_embeds=None,
                             past_key_values=None,
                             **kw):
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError('need input_ids or inputs_embeds')
                inputs_embeds = orig_embed(input_ids)

            bsz, seqlen, _ = inputs_embeds.shape
            past_len = 0
            if past_key_values is not None and len(past_key_values) > 0:
                # NeoX: K shape is (bsz, num_heads, time, head_dim)
                past_len = past_key_values[0][0].shape[-2]

            if position_ids is None:
                if attention_mask is not None and getattr(attention_mask, "dim", lambda: 0)() == 2:
                    pos = (attention_mask.cumsum(-1) - 1).clamp(min=0).to(torch.long)
                    position_ids = pos[:, past_len:past_len + seqlen]
                else:
                    start = past_len
                    position_ids = torch.arange(
                        start, start + seqlen, device=inputs_embeds.device, dtype=torch.long
                    ).unsqueeze(0).expand(bsz, -1)
            else:
                position_ids = position_ids.to(torch.long)

            inputs_embeds = inputs_embeds + self.ape_pe(position_ids).to(inputs_embeds.dtype)
            inputs_embeds = self.ape_drop(inputs_embeds)

            return orig_forward(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kw,
            )

        model.forward = types.MethodType(forward_with_ape, model)

    rope_info = ""
    if pos_encoding == 'rope':
        rope_info = f" (rotary_pct={rotary_pct}"
        if rope_scaling is not None:
            rope_info += f", scaling={rope_scaling}"
        rope_info += ")"
    print(
        f"Created GPT-NeoX: {('Parallel' if use_parallel_residual else 'Sequential')} residual, "
        f"PE={pos_encoding.upper()}"
        + rope_info
        + f", params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    return model


class ComputeAccuracy(TrainerCallback):
    """
    Periodic evaluator for "classification-as-LM" datasets where each sequence has
    exactly one supervised token (the only label != -100).

    Prediction uses the logits from the *previous* position (standard causal LM shift)
    and can be restricted to a candidate set of label token IDs (K-way classification).
    
    Optionally saves best checkpoints for val_id and val_ood loaders independently.
    """

    def __init__(
        self,
        eval_loaders,
        eval_interval: int = 1000,
        candidate_token_ids: Optional[Sequence[int]] = None,  # possible token candidates for classification
        log_prefix: str = "val",
        output_dir: Optional[str] = None,  # if set, save best checkpoints here
        loss_loader_names: Optional[Sequence[str]] = ("train", "val_id", "val_ood"),  # loaders to compute CLM loss for
    ):
        self.eval_loaders = eval_loaders
        self.eval_interval = int(eval_interval)
        self.candidate_token_ids = list(candidate_token_ids) if candidate_token_ids is not None else None
        self.log_prefix = log_prefix
        self.output_dir = output_dir
        self.loss_loader_names = set(loss_loader_names) if loss_loader_names is not None else set()
        
        # Track best accuracies for checkpoint saving
        self.best_val_id_acc = -1.0
        self.best_val_ood_acc = -1.0

    def _save_checkpoint(self, model, save_path: str, step: int, acc: float):
        """Save model checkpoint to the given path."""
        os.makedirs(save_path, exist_ok=True)
        # Unwrap DDP/FSDP if needed
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(save_path)
        print(f"Saved best checkpoint to {save_path} (step={step}, acc={acc:.4f})")

    def on_step_end(self, args, state, control, model, **kwargs):
        step = int(round(state.global_step))
        if self.eval_interval <= 0 or (step % self.eval_interval) != 0:
            return control

        # DDP safety: keep all ranks in lockstep, but evaluate only on rank 0.
        ddp = _dist_world_size() > 1
        if ddp:
            _dist_barrier()
        if ddp and _dist_rank() != 0:
            # Wait for rank 0 to finish evaluation, then continue training.
            _dist_barrier()
            return control

        device = _get_model_device(model)
        was_training = model.training
        model.eval()

        cand = None
        if self.candidate_token_ids is not None:
            cand = torch.tensor(self.candidate_token_ids, dtype=torch.long, device=device)

        try:
            with torch.inference_mode():
                for loader_name, loader in self.eval_loaders.items():
                    all_preds = []
                    all_labels = []
                    
                    # Track loss if this loader is in loss_loader_names
                    compute_loss = loader_name in self.loss_loader_names
                    all_losses = []
                    total_tokens = 0

                    for batch in loader:
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch.get("attention_mask")
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(device)

                        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, T, V)

                        # Compute CLM loss if needed (before accuracy computation)
                        if compute_loss:
                            # Construct full CLM labels from input_ids (mask padding with -100)
                            full_labels = input_ids.clone()
                            if attention_mask is not None:
                                full_labels[attention_mask == 0] = -100
                            
                            # Standard causal LM loss: predict token[i+1] from logits[i]
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = full_labels[..., 1:].contiguous()
                            
                            # Count non-masked tokens for proper averaging
                            num_tokens = (shift_labels != -100).sum().item()
                            if num_tokens > 0:
                                batch_loss = F.cross_entropy(
                                    shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1),
                                    ignore_index=-100,
                                    reduction="sum"
                                )
                                all_losses.append(batch_loss.item())
                                total_tokens += num_tokens

                        # One non-(-100) label per sample (for accuracy).
                        batch_idx, pos = (labels != -100).nonzero(as_tuple=True)
                        if batch_idx.numel() != labels.shape[0]:
                            raise ValueError(
                                f"Expected exactly one supervised token per sample, but got "
                                f"{batch_idx.numel()} labels for batch size {labels.shape[0]} "
                                f"(loader={loader_name})."
                            )
                        if (pos == 0).any():
                            raise ValueError(
                                f"Supervised token at position 0 can't be predicted by a causal LM "
                                f"(loader={loader_name})."
                            )

                        gold = labels[batch_idx, pos]  # (B,)
                        step_logits = logits[batch_idx, pos - 1]  # (B, V)

                        if cand is None:
                            pred = step_logits.argmax(dim=-1)
                        else:
                            cand_logits = step_logits.index_select(dim=-1, index=cand)  # (B, K)
                            pred = cand[cand_logits.argmax(dim=-1)]

                        all_preds.append(pred.detach().cpu())
                        all_labels.append(gold.detach().cpu())

                    # Compute accuracy
                    if not all_labels:
                        acc = float("nan")
                    else:
                        preds = torch.cat(all_preds)
                        golds = torch.cat(all_labels)
                        acc = (preds == golds).to(torch.float32).mean().item()

                    # Compute average loss
                    if compute_loss and total_tokens > 0:
                        avg_loss = sum(all_losses) / total_tokens
                    else:
                        avg_loss = float("nan") if compute_loss else None

                    # Log accuracy
                    print(f"{loader_name} accuracy: {acc:.3f}")
                    metrics = {f"{self.log_prefix}/{loader_name}/accuracy": acc, "step": step}
                    
                    # Log loss if computed
                    if avg_loss is not None:
                        print(f"{loader_name} loss: {avg_loss:.4f}")
                        metrics[f"{self.log_prefix}/{loader_name}/loss"] = avg_loss
                    
                    state.log_history.append(metrics)
                    if wandb is not None and getattr(wandb, "run", None) is not None:
                        wandb.log(metrics)
                    
                    # Save best checkpoint for val_id and val_ood
                    if self.output_dir is not None and not (acc != acc):  # skip if NaN
                        if loader_name == "val_id" and acc > self.best_val_id_acc:
                            self.best_val_id_acc = acc
                            save_path = os.path.join(self.output_dir, "best_val_id")
                            self._save_checkpoint(model, save_path, step, acc)
                        elif loader_name == "val_ood" and acc > self.best_val_ood_acc:
                            self.best_val_ood_acc = acc
                            save_path = os.path.join(self.output_dir, "best_val_ood")
                            self._save_checkpoint(model, save_path, step, acc)
        finally:
            if was_training:
                model.train()
            if ddp:
                # Always release nonzero ranks, even if eval fails (avoid deadlocks).
                _dist_barrier()

        return control

