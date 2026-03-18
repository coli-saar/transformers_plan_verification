from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer, TrainerCallback
from datasets import Dataset

from training.constants import VERDICT_CORRECT_TOKEN, VERDICT_INCORRECT_TOKEN


def load_splits(splits_path: str | Path) -> Dict[str, List[int]]:
    """Load splits from a JSON file. Each key with a list value is treated as a split."""
    splits_path = Path(splits_path)
    if not splits_path.exists():
        raise FileNotFoundError(f"Split file not found: {splits_path}")
    data_dict = json.loads(splits_path.read_text(encoding="utf-8"))
    splits = {}
    for k, v in data_dict.items():
        if isinstance(v, list):
            splits[str(k)] = [int(x) for x in v]
    return splits


def load_jsonl_dataset(path: str | Path) -> Dataset:
    """Stream a JSONL file into a HF Dataset without loading it fully into RAM."""
    from datasets import Dataset  # local import so split-only utilities don't require datasets at import time

    path = str(path)

    def gen() -> Iterable[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "input_ids" not in rec:
                    raise ValueError(f"{path}:{line_no}: missing required key 'input_ids'")
                yield rec

    ds = Dataset.from_generator(gen)
    if "input_ids" not in ds.column_names:
        raise ValueError(f"Loaded dataset from {path} but no 'input_ids' column found")
    return ds


def _get_verdict_token_ids(tokenizer: PreTrainedTokenizer) -> Tuple[int, int]:
    """Return (correct_id, incorrect_id) and verify they are single tokens."""
    corr = tokenizer.encode(VERDICT_CORRECT_TOKEN, add_special_tokens=False)
    inc = tokenizer.encode(VERDICT_INCORRECT_TOKEN, add_special_tokens=False)
    if len(corr) != 1 or len(inc) != 1:
        raise ValueError(
            "Verdict tokens must each be a single token. "
            f"Got correct={corr}, incorrect={inc}. "
            "Make sure your tokenizer has ' correct' and ' incorrect' as special tokens."
        )
    return int(corr[0]), int(inc[0])


def _validate_input_ids(
    input_ids: Sequence[int],
    *,
    vocab_size: Optional[int],
    require_verdict_at_end: bool,
    verdict_ids: Optional[set[int]],
) -> None:
    if not isinstance(input_ids, (list, tuple)):
        raise ValueError(f"input_ids must be a list/tuple, got {type(input_ids)}")
    if len(input_ids) == 0:
        raise ValueError("input_ids must be non-empty")
    for t in input_ids:
        if not isinstance(t, int):
            raise ValueError(f"input_ids must contain ints, got element {t!r} ({type(t)})")
        if t < 0:
            raise ValueError(f"input_ids contains negative id: {t}")
        if vocab_size is not None and t >= vocab_size:
            raise ValueError(f"input_ids contains out-of-range id {t} (vocab_size={vocab_size})")
    if require_verdict_at_end and verdict_ids is not None:
        if input_ids[-1] not in verdict_ids:
            raise ValueError(f"Expected final token to be a verdict id {sorted(verdict_ids)}, got {input_ids[-1]}")


def _labels_autoregressive(input_ids: Sequence[int]) -> List[int]:
    return list(input_ids)


def _labels_verdict_only_sequence(input_ids: Sequence[int], verdict_ids: set[int]) -> List[int]:
    # loss should only be applied to the verdict token (final token in the sequence)
    labels = [-100] * len(input_ids)
    if input_ids[-1] not in verdict_ids:
        raise ValueError(f"Expected last token to be verdict id {sorted(verdict_ids)}, got {input_ids[-1]}")
    labels[-1] = int(input_ids[-1])
    return labels


def _labels_verdict_only_packed(input_ids: Sequence[int], verdict_ids: set[int]) -> List[int]:
    # loss applied wherever a verdict token appears in the packed block
    labels = [-100] * len(input_ids)
    for i, tid in enumerate(input_ids):
        if int(tid) in verdict_ids:
            labels[i] = int(tid)
    return labels


def _build_causal_segment_mask(segment_ids: List[int], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Build 4D causal attention mask with sample isolation.
    Shape: (1, 1, seq_len, seq_len)
    Returns 0.0 for attend, large negative for mask (additive mask convention).
    """
    seq_len = len(segment_ids)
    seg = torch.tensor(segment_ids)
    
    # Same segment mask: (i,j) = True if token i and j in same segment
    same_seg = seg.unsqueeze(0) == seg.unsqueeze(1)  # (seq_len, seq_len)
    
    # Causal mask: (i,j) = True if i >= j (can attend to current and past)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    
    # Combined: can attend if same segment AND causal
    can_attend = same_seg & causal
    
    # Convert to additive attention mask format (0 = attend, large_neg = mask)
    mask = torch.where(can_attend, 0.0, torch.finfo(dtype).min).to(dtype)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


def _build_position_ids_reset(segment_ids: List[int]) -> List[int]:
    """Build position IDs that reset to 0 at the start of each segment."""
    position_ids = []
    current_pos = 0
    prev_seg = -1
    for seg_id in segment_ids:
        if seg_id != prev_seg:
            current_pos = 0
            prev_seg = seg_id
        position_ids.append(current_pos)
        current_pos += 1
    return position_ids


def _build_position_ids_rand_offset(
    segment_ids: List[int], 
    max_position_id: int,
    rng: random.Random
) -> List[int]:
    """
    Build position IDs with random offset per segment.
    Each segment starts from a random position in [0, max_offset].
    """
    # First pass: compute segment lengths
    seg_lengths: Dict[int, int] = {}
    for seg_id in segment_ids:
        seg_lengths[seg_id] = seg_lengths.get(seg_id, 0) + 1
    
    position_ids = []
    prev_seg = -1
    current_offset = 0
    current_pos = 0
    
    for seg_id in segment_ids:
        if seg_id != prev_seg:
            seg_len = seg_lengths[seg_id]
            max_offset = max(0, max_position_id - seg_len)
            current_offset = rng.randint(0, max_offset)
            current_pos = 0
            prev_seg = seg_id
        position_ids.append(current_offset + current_pos)
        current_pos += 1
    return position_ids


class InstanceSequenceDataset(TorchDataset):
    """Map-style dataset that yields one sample per JSONL record."""

    def __init__(
        self,
        base: Dataset,
        tokenizer: PreTrainedTokenizer,
        label_mode: str = "autoregressive",
        require_verdict_at_end: bool = True,
        include_metadata: bool = False,
        validate_first_n: int = 50,
    ):
        self.base = base
        self.tokenizer = tokenizer
        self.label_mode = label_mode
        self.require_verdict_at_end = bool(require_verdict_at_end)
        if self.label_mode == "stepwise_correctness":
            self.require_verdict_at_end = False
        self.include_metadata = bool(include_metadata)

        correct_id, incorrect_id = _get_verdict_token_ids(tokenizer)
        self.verdict_ids: set[int] = {correct_id, incorrect_id}
        self.vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else None

        if "input_ids" not in base.column_names:
            raise ValueError("Base dataset must have an 'input_ids' column")

        n = min(int(validate_first_n), len(base))
        for i in range(n):
            _validate_input_ids(
                base[int(i)]["input_ids"],
                vocab_size=self.vocab_size,
                require_verdict_at_end=self.require_verdict_at_end,
                verdict_ids=self.verdict_ids,
            )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.base[int(idx)]
        input_ids = rec["input_ids"]
        _validate_input_ids(
            input_ids,
            vocab_size=self.vocab_size,
            require_verdict_at_end=self.require_verdict_at_end,
            verdict_ids=self.verdict_ids,
        )

        if self.label_mode == "autoregressive":
            labels = _labels_autoregressive(input_ids)
        elif self.label_mode == "verdict_only":
            labels = _labels_verdict_only_sequence(input_ids, self.verdict_ids)
        elif self.label_mode == "stepwise_correctness":
            if "is_correct" not in rec:
                raise ValueError("is_correct column missing for stepwise_correctness")
            is_correct = rec["is_correct"]
            
            # We want input_ids[i] to predict is_correct[i].
            # HF Causal LM shifts labels right by 1 (predicts labels[i+1] from input[i]).
            # So we set labels[i+1] = is_correct[i], which implies labels = [-100] + is_correct.
            if len(is_correct) != len(input_ids):
                raise ValueError(
                    f"is_correct length {len(is_correct)} != input_ids length {len(input_ids)} "
                    f"for example {rec.get('example_id', 'unknown')}"
                )
            labels = [-100] + list(is_correct[:-1])
        else:
            raise ValueError(f"Unknown label_mode={self.label_mode}")

        item: Dict[str, Any] = {
            "input_ids": list(input_ids),
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }
        if self.include_metadata:
            for k in ("example_id", "problem_idx", "plan_len", "is_correct"):
                if k in rec:
                    item[k] = rec[k]
        return item


class PackedLMDataset(TorchDataset):
    """
    Epoch-wise packed dataset: randomly concatenates sequences then slices into fixed-length blocks.

    Call set_epoch(epoch) each epoch to reshuffle + re-pack.
    
    Args:
        allow_cross_sample_attention: If False, generate 4D attention mask to block cross-sample attention.
        reset_pos_id: None (natural growth), 'reset' (reset per sample), 'rand_offset' (random start per sample).
    """

    def __init__(
        self,
        base: Dataset,
        tokenizer: PreTrainedTokenizer,
        block_size: int,
        seed: int = 42,
        label_mode: str = "autoregressive",
        validate_first_n: int = 50,
        allow_cross_sample_attention: bool = False,
        reset_pos_id: Optional[str] = None,
    ):
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        self.base = base
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.label_mode = label_mode
        self.allow_cross_sample_attention = allow_cross_sample_attention
        self.reset_pos_id = reset_pos_id

        correct_id, incorrect_id = _get_verdict_token_ids(tokenizer)
        self.verdict_ids: set[int] = {correct_id, incorrect_id}
        self.vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else None

        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer.eos_token_id must be set for packing mode")
        self.eos_token_id = int(eos)

        if "input_ids" not in base.column_names:
            raise ValueError("Base dataset must have an 'input_ids' column")
        
        if self.label_mode == "stepwise_correctness" and "is_correct" not in base.column_names:
             raise ValueError("Base dataset must have 'is_correct' column for stepwise_correctness mode")

        require_verdict = True
        if self.label_mode == "stepwise_correctness":
            require_verdict = False

        n = min(int(validate_first_n), len(base))
        for i in range(n):
            _validate_input_ids(
                base[int(i)]["input_ids"],
                vocab_size=self.vocab_size,
                require_verdict_at_end=require_verdict,
                verdict_ids=self.verdict_ids,
            )

        # Cache all sequences in memory once.
        self._cached_seqs_eos: List[List[int]] = []
        self._cached_is_correct: Optional[List[List[int]]] = None
        if self.label_mode == "stepwise_correctness":
            self._cached_is_correct = []

        for i in range(len(base)):
            # Cache inputs
            seq = base[int(i)]["input_ids"]
            seq_list = [int(t) for t in seq]
            seq_list.append(self.eos_token_id)
            self._cached_seqs_eos.append(seq_list)
            
            # Cache is_correct if needed
            if self._cached_is_correct is not None:
                # Expect is_correct to be same len as original input
                is_correct = base[int(i)]["is_correct"]
                if len(is_correct) != len(seq):
                    raise ValueError(
                        f"is_correct length {len(is_correct)} != input_ids length {len(seq)} "
                        f"at index {i}"
                    )
                # We don't append anything to is_correct, because labels will be shifted.
                # input: [t0, t1, EOS]
                # is_correct: [c0, c1]
                # labels: [-100, c0, c1]
                # len match: 3 vs 3.
                self._cached_is_correct.append([int(x) for x in is_correct])

        self._blocks: List[List[int]] = []
        self._blocks_segment_ids: List[List[int]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = random.Random(self.seed + int(epoch))
        
        # If stepwise, we need to shuffle is_correct in sync
        indices = list(range(len(self._cached_seqs_eos)))
        rng.shuffle(indices)

        # Flatten in shuffled order, tracking segment IDs
        all_tokens: List[int] = []
        all_segment_ids: List[int] = []
        seqs = self._cached_seqs_eos
        
        if self.label_mode == "stepwise_correctness":
            all_labels: List[int] = []
            cached_correct = self._cached_is_correct
            assert cached_correct is not None
            for seg_idx, i in enumerate(indices):
                s_in = seqs[int(i)]
                s_corr = cached_correct[int(i)]
                all_tokens.extend(s_in)
                all_segment_ids.extend([seg_idx] * len(s_in))
                # s_in has EOS. len = L+1. s_corr has len L.
                # labels = [-100] + s_corr. len = L+1.
                all_labels.extend([-100])
                all_labels.extend(s_corr)
        else:
            for seg_idx, i in enumerate(indices):
                s_in = seqs[int(i)]
                all_tokens.extend(s_in)
                all_segment_ids.extend([seg_idx] * len(s_in))

        n_blocks = len(all_tokens) // self.block_size
        if n_blocks <= 0:
            self._blocks = []
            self._blocks_segment_ids = []
            return
        cutoff = n_blocks * self.block_size
        self._blocks = [all_tokens[j : j + self.block_size] for j in range(0, cutoff, self.block_size)]
        self._blocks_segment_ids = [all_segment_ids[j : j + self.block_size] for j in range(0, cutoff, self.block_size)]
        
        # Store separate labels list if stepwise 
        if self.label_mode == "stepwise_correctness":
            self._blocks_labels = [all_labels[j : j + self.block_size] for j in range(0, cutoff, self.block_size)]

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        input_ids = self._blocks[int(idx)]
        segment_ids = self._blocks_segment_ids[int(idx)]

        if self.label_mode == "autoregressive":
            labels = _labels_autoregressive(input_ids)
        elif self.label_mode == "verdict_only":
            labels = _labels_verdict_only_packed(input_ids, self.verdict_ids)
        elif self.label_mode == "stepwise_correctness":
            labels = self._blocks_labels[int(idx)]
        else:
            raise ValueError(f"Unknown label_mode={self.label_mode}")

        result: Dict[str, Any] = {"input_ids": input_ids, "labels": labels}
        
        # Attention mask
        if self.allow_cross_sample_attention:
            result["attention_mask"] = [1] * self.block_size
        else:
            result["attention_mask"] = _build_causal_segment_mask(segment_ids)
        
        # Position IDs
        if self.reset_pos_id == "reset":
            result["position_ids"] = _build_position_ids_reset(segment_ids)
        elif self.reset_pos_id == "rand_offset":
            rng = random.Random(self.seed + idx)
            result["position_ids"] = _build_position_ids_rand_offset(segment_ids, self.block_size, rng)
        # If reset_pos_id is None, don't include position_ids (use default behavior)
        
        return result


class PackedLMDatasetWithoutBreaking(PackedLMDataset):
    """
    Packed dataset that never breaks instances. If an instance doesn't fit in the current block,
    starts a new block. Pads short blocks to block_size.
    Raises ValueError if an instance > block_size.
    """

    def set_epoch(self, epoch: int) -> None:
        rng = random.Random(self.seed + int(epoch))
        # Reuse the cached sequences from the base class
        indices = list(range(len(self._cached_seqs_eos)))
        rng.shuffle(indices)

        self._blocks = []  # type: ignore # We store tuples
        self._blocks_segment_ids = []

        # Determine pad token (prefer tokenizer's pad, else eos)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.eos_token_id

        curr_inputs: List[int] = []
        curr_labels: List[int] = []
        curr_mask: List[int] = []
        curr_segment_ids: List[int] = []
        segment_counter = 0

        cached_correct = self._cached_is_correct if self.label_mode == "stepwise_correctness" else None

        for idx in indices:
            seq_inputs = self._cached_seqs_eos[int(idx)]
            if len(seq_inputs) > self.block_size:
                raise ValueError(
                    f"Instance length {len(seq_inputs)} exceeds block_size {self.block_size}"
                )

            # Check if we need to flush the current block
            if len(curr_inputs) + len(seq_inputs) > self.block_size:
                # Pad and commit
                pad_len = self.block_size - len(curr_inputs)
                if pad_len > 0:
                    curr_inputs.extend([int(pad_id)] * pad_len)
                    curr_labels.extend([-100] * pad_len)
                    curr_mask.extend([0] * pad_len)
                    # Padding tokens get a unique segment ID (won't matter for attention anyway)
                    curr_segment_ids.extend([-1] * pad_len)
                self._blocks.append((curr_inputs, curr_labels, curr_mask))
                self._blocks_segment_ids.append(curr_segment_ids)
                curr_inputs = []
                curr_labels = []
                curr_mask = []
                curr_segment_ids = []

            # Add current instance
            curr_inputs.extend(seq_inputs)
            curr_mask.extend([1] * len(seq_inputs))
            curr_segment_ids.extend([segment_counter] * len(seq_inputs))
            segment_counter += 1

            # Generate labels for this segment
            if self.label_mode == "autoregressive":
                seq_labels = _labels_autoregressive(seq_inputs)
            elif self.label_mode == "verdict_only":
                seq_labels = _labels_verdict_only_packed(seq_inputs, self.verdict_ids)
            elif self.label_mode == "stepwise_correctness":
                assert cached_correct is not None
                s_corr = cached_correct[int(idx)]
                seq_labels = [-100] + list(s_corr)
            else:
                raise ValueError(f"Unknown label_mode={self.label_mode}")
            
            curr_labels.extend(seq_labels)

        # Flush final partial block
        if curr_inputs:
            pad_len = self.block_size - len(curr_inputs)
            if pad_len > 0:
                curr_inputs.extend([int(pad_id)] * pad_len)
                curr_labels.extend([-100] * pad_len)
                curr_mask.extend([0] * pad_len)
                curr_segment_ids.extend([-1] * pad_len)
            self._blocks.append((curr_inputs, curr_labels, curr_mask))
            self._blocks_segment_ids.append(curr_segment_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        input_ids, labels, orig_mask = self._blocks[int(idx)]  # type: ignore
        segment_ids = self._blocks_segment_ids[int(idx)]
        
        result: Dict[str, Any] = {"input_ids": input_ids, "labels": labels}
        
        # Attention mask
        if self.allow_cross_sample_attention:
            # Use original 1D mask (includes padding info)
            result["attention_mask"] = orig_mask
        else:
            # Build 4D mask with sample isolation
            # Note: padding positions are handled - they have segment_id=-1 so won't match any real segment
            result["attention_mask"] = _build_causal_segment_mask(segment_ids)
        
        # Position IDs
        if self.reset_pos_id == "reset":
            result["position_ids"] = _build_position_ids_reset(segment_ids)
        elif self.reset_pos_id == "rand_offset":
            rng = random.Random(self.seed + idx)
            result["position_ids"] = _build_position_ids_rand_offset(segment_ids, self.block_size, rng)
        
        return result



class RepackEachEpochCallback(TrainerCallback):
    """HF Trainer callback: re-pack a `PackedLMDataset` at the start of each epoch."""

    def __init__(self, packed_dataset: PackedLMDataset):
        self.packed_dataset = packed_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        epoch = int(state.epoch or 0)
        self.packed_dataset.set_epoch(epoch)
        return control


@dataclass
class SimpleCausalLMCollator:
    """
    Minimal collator that pads input_ids/labels and builds attention_mask.

    - Pads input_ids with tokenizer.pad_token_id (or eos_token_id if pad is missing)
    - Pads labels with -100 (ignore index)
    - Preserves pre-existing attention_mask (including 4D masks) and position_ids
    """

    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate empty batch")

        max_len = max(len(f["input_ids"]) for f in features)

        input_ids_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []
        
        # Check if features have pre-existing attention_mask
        has_custom_attn_mask = "attention_mask" in features[0] and features[0]["attention_mask"] is not None
        is_4d_mask = has_custom_attn_mask and isinstance(features[0]["attention_mask"], torch.Tensor)
        
        # Check if features have position_ids
        has_position_ids = "position_ids" in features[0] and features[0]["position_ids"] is not None

        attn_masks = []
        position_ids_batch = []

        for f in features:
            ids = list(f["input_ids"])
            pad = max_len - len(ids)
            input_ids_batch.append(ids + [self.pad_token_id] * pad)

            labels = f.get("labels")
            if labels is None:
                labels = list(ids)
            else:
                labels = list(labels)
            labels_batch.append(labels + [-100] * pad)

            # Handle attention_mask
            if has_custom_attn_mask:
                mask = f["attention_mask"]
                if is_4d_mask:
                    attn_masks.append(mask)
                else:
                    if isinstance(mask, torch.Tensor):
                        mask = mask.tolist()
                    attn_masks.append(list(mask) + [0] * pad)
            else:
                attn_masks.append([1] * len(ids) + [0] * pad)
            
            # Handle position_ids
            if has_position_ids:
                pos_ids = f["position_ids"]
                if isinstance(pos_ids, torch.Tensor):
                    pos_ids = pos_ids.tolist()
                # Pad position_ids (padding positions don't really matter)
                position_ids_batch.append(list(pos_ids) + [0] * pad)

        result = {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }
        
        # Handle attention_mask stacking
        if is_4d_mask:
            # Stack 4D masks: each is (1, 1, seq, seq), result is (B, 1, seq, seq)
            result["attention_mask"] = torch.cat(attn_masks, dim=0)
        else:
            result["attention_mask"] = torch.tensor(attn_masks, dtype=torch.long)
        
        # Add position_ids if present
        if has_position_ids:
            result["position_ids"] = torch.tensor(position_ids_batch, dtype=torch.long)
        
        return result


def make_collator(tokenizer: PreTrainedTokenizer) -> SimpleCausalLMCollator:
    pad = tokenizer.pad_token_id
    if pad is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
        pad = int(tokenizer.eos_token_id)
    return SimpleCausalLMCollator(pad_token_id=int(pad))


def build_dataset(
    jsonl_path: str | Path,
    tokenizer: PreTrainedTokenizer,
    split: Optional[str] = None,
    splits_path: str | Path | None = None,
    mode: str = "sequence",
    label_mode: str = "autoregressive",
    block_size: Optional[int] = None,
    seed: int = 42,
    include_metadata: bool = False,
    validate_first_n: int = 50,
    allow_cross_sample_attention: bool = False,
    reset_pos_id: Optional[str] = None,
) -> TorchDataset:
    """
    Convenience helper that loads the JSONL and returns a torch Dataset for training.
    
    Args:
        allow_cross_sample_attention: For packed modes, if False, generate 4D mask to block cross-sample attention.
        reset_pos_id: For packed modes, None (natural growth), 'reset' (reset per sample), 'rand_offset'.
    """
    # Persisted split selection (computed before loading into HF dataset)
    indices: Optional[List[int]] = None
    if split is not None:
        if splits_path is None:
            raise ValueError("splits_path is required when split is specified")
        splits = load_splits(splits_path)
        if str(split) not in splits:
            raise ValueError(f"Unknown split={split!r}; available={sorted(splits.keys())}")
        indices = splits[str(split)]

    base = load_jsonl_dataset(jsonl_path)
    if indices is not None:
        base = base.select(indices)
        if len(base) != len(indices):
            raise ValueError("Internal error: split selection size mismatch")
    if mode == "sequence":
        return InstanceSequenceDataset(
            base,
            tokenizer,
            label_mode=label_mode,
            include_metadata=include_metadata,
            validate_first_n=validate_first_n,
        )
    if mode == "lm":
        if block_size is None:
            raise ValueError("block_size is required for mode='lm'")
        return PackedLMDataset(
            base,
            tokenizer,
            block_size=int(block_size),
            seed=int(seed),
            label_mode=label_mode,
            validate_first_n=validate_first_n,
            allow_cross_sample_attention=allow_cross_sample_attention,
            reset_pos_id=reset_pos_id,
        )
    if mode == "lm_no_break":
        if block_size is None:
            raise ValueError("block_size is required for mode='lm_no_break'")
        return PackedLMDatasetWithoutBreaking(
            base,
            tokenizer,
            block_size=int(block_size),
            seed=int(seed),
            label_mode=label_mode,
            validate_first_n=validate_first_n,
            allow_cross_sample_attention=allow_cross_sample_attention,
            reset_pos_id=reset_pos_id,
        )
    raise ValueError(f"Unknown mode={mode}")


def build_eval_dataloader(
    jsonl_path: str | Path,
    tokenizer: PreTrainedTokenizer,
    split: str = "val",
    splits_path: str | Path | None = None,
    label_mode: str = "verdict_only",
    seed: int = 42,
    batch_size: int = 8,
    num_workers: int = 0,
) -> DataLoader:
    """Lightweight eval dataloader builder (no shuffle, persisted 8:1:1 split)."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    ds = build_dataset(
        jsonl_path,
        tokenizer,
        split=split,
        splits_path=splits_path,
        mode="sequence", 
        label_mode=label_mode,
        block_size=None, 
        seed=int(seed),
        include_metadata=False,
        validate_first_n=50,
    )
    if len(ds) == 0:
        raise ValueError(f"Eval dataset is empty for split={split}. Check split file.")

    collator = make_collator(tokenizer)
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collator,
    )
