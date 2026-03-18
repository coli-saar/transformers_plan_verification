from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

TRAIN_MODES = ["sequence", "lm", "lm_no_break"]
LABEL_MODES = ["autoregressive", "verdict_only", "stepwise_correctness"]
RESET_POS_ID_MODES = [None, "reset", "rand_offset"]


def _as_int(x: Any, *, name: str, default: Optional[int] = None) -> int:
    if x is None:
        if default is not None:
            return default
        raise ValueError(f"Expected int for {name}, got None")
    try:
        return int(x)
    except Exception as exc:
        raise ValueError(f"Expected int for {name}, got {x!r}") from exc


def _as_float(x: Any, *, name: str, default: Optional[float] = None) -> float:
    if x is None:
        if default is not None:
            return default
        raise ValueError(f"Expected float for {name}, got None")
    try:
        return float(x)
    except Exception as exc:
        raise ValueError(f"Expected float for {name}, got {x!r}") from exc


def _as_bool(x: Any, *, name: str, default: Optional[bool] = None) -> bool:
    if x is None:
        if default is not None:
            return default
        raise ValueError(f"Expected bool for {name}, got None")
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"true", "yes", "y", "1"}:
            return True
        if v in {"false", "no", "n", "0"}:
            return False
    raise ValueError(f"Expected bool for {name}, got {x!r}")


def _get_section(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    val = d.get(key, None)
    if val is None:
        return {}
    if not isinstance(val, dict):
        raise ValueError(f"Expected '{key}' to be a dict, got {type(val)}")
    return val


@dataclass(frozen=True)
class PathsConfig:
    jsonl_path: str
    tokenizer_dir: str
    output_dir: str
    splits_path: Optional[str] = None


@dataclass(frozen=True)
class DataConfig:
    train_mode: str = "lm"
    train_label_mode: str = "verdict_only"
    block_size: Optional[int] = 256  # required if train_mode='lm'
    per_device_eval_batch_size: int = 8
    eval_num_workers: int = 0
    allow_cross_sample_attention: bool = False  # if False, block attention across samples in packed mode
    reset_pos_id: Optional[str] = None  # None | 'reset' | 'rand_offset'


@dataclass(frozen=True)
class ModelConfig:
    architecture: str  # gpt-neox-rope|gpt-neox-ape|gpt-neox-nope
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int

    rotary_pct: float = 0.25
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None  # e.g., {"type": "linear", "factor": 2.0}
    dropout: float = 0.0
    use_parallel_residual: bool = False

    # Optional explicit dropouts (override `dropout` fallback if set)
    embd_pdrop: Optional[float] = None
    attn_pdrop: Optional[float] = None
    resid_pdrop: Optional[float] = None

    # Optional local checkpoint directory for weight init
    model_name_or_path: Optional[str] = None


@dataclass(frozen=True)
class TrainConfig:
    # Termination: prefer max_steps for packed training; otherwise epochs.
    max_steps: int = -1
    num_train_epochs: float = 1.0

    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1

    # Optimizer
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    warmup_steps: int = 0
    lr_scheduler_type: str = "cosine"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3

    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False

    resume: bool = True
    overwrite_output_dir: bool = False


@dataclass(frozen=True)
class EvalConfig:
    eval_interval: int = 500
    train_subset_size: int = 2048
    splits: List[str] = field(default_factory=lambda: ["val", "test"])


@dataclass(frozen=True)
class WandbConfig:
    project: str = ""
    entity: str = ""
    run_name: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig
    wandb: WandbConfig

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExperimentConfig":
        if not isinstance(d, dict):
            raise ValueError("Top-level YAML must be a mapping/dict")

        seed = _as_int(d.get("seed", 42), name="seed")

        paths_d = _get_section(d, "paths")
        for req in ("jsonl_path", "tokenizer_dir", "output_dir"):
            if req not in paths_d:
                raise ValueError(f"Missing required paths.{req}")
        paths = PathsConfig(
            jsonl_path=str(paths_d["jsonl_path"]),
            tokenizer_dir=str(paths_d["tokenizer_dir"]),
            output_dir=str(paths_d["output_dir"]),
            splits_path=str(paths_d["splits_path"]) if paths_d.get("splits_path") else None,
        )

        data_d = _get_section(d, "data")
        train_mode = str(data_d.get("train_mode") or "lm")
        if train_mode not in TRAIN_MODES:
            raise ValueError(f"data.train_mode must be one of: {TRAIN_MODES}")
        train_label_mode = str(data_d.get("train_label_mode") or "verdict_only")
        if train_label_mode not in LABEL_MODES:
            raise ValueError(f"data.train_label_mode must be one of: {LABEL_MODES}")
        # For lm mode, block_size is required. If key is missing, default to 256.
        # If key is present but set to null, that's an explicit user error -> raise.
        if "block_size" in data_d:
            block_size_raw = data_d["block_size"]
            if block_size_raw is None:
                if train_mode == "lm":
                    raise ValueError("data.block_size cannot be null when data.train_mode='lm'")
                block_size = None
            else:
                block_size = _as_int(block_size_raw, name="data.block_size")
        else:
            block_size = 256  # default when key is missing
        if train_mode == "lm" and (block_size is None or block_size <= 0):
            raise ValueError("data.block_size must be set and > 0 when data.train_mode='lm'")
        
        # Parse allow_cross_sample_attention
        allow_cross_sample_attention = _as_bool(
            data_d.get("allow_cross_sample_attention"), 
            name="data.allow_cross_sample_attention", 
            default=False
        )
        
        # Parse reset_pos_id
        reset_pos_id_raw = data_d.get("reset_pos_id", None)
        if reset_pos_id_raw is None:
            reset_pos_id = None
        else:
            reset_pos_id = str(reset_pos_id_raw)
            if reset_pos_id not in RESET_POS_ID_MODES:
                raise ValueError(f"data.reset_pos_id must be one of {RESET_POS_ID_MODES}, got {reset_pos_id!r}")
        
        data = DataConfig(
            train_mode=train_mode,  # type: ignore[arg-type]
            train_label_mode=train_label_mode,  # type: ignore[arg-type]
            block_size=block_size,
            per_device_eval_batch_size=_as_int(
                data_d.get("per_device_eval_batch_size"), name="data.per_device_eval_batch_size", default=8
            ),
            eval_num_workers=_as_int(data_d.get("eval_num_workers"), name="data.eval_num_workers", default=0),
            allow_cross_sample_attention=allow_cross_sample_attention,
            reset_pos_id=reset_pos_id,
        )

        model_d = _get_section(d, "model")
        for req in ("architecture", "n_embd", "n_layer", "n_head"):
            if req not in model_d:
                raise ValueError(f"Missing required model.{req}")
        arch = str(model_d["architecture"]).lower()
        if arch not in {"gpt-neox-rope", "gpt-neox-ape", "gpt-neox-nope"}:
            raise ValueError("model.architecture must be one of: gpt-neox-rope, gpt-neox-ape, gpt-neox-nope")
        n_positions_raw = model_d.get("n_positions", None)
        if n_positions_raw is None:
            if data.block_size is None:
                raise ValueError("model.n_positions is required when data.block_size is not set")
            n_positions = int(data.block_size)
        else:
            n_positions = _as_int(n_positions_raw, name="model.n_positions")

        def _opt_float(key: str) -> Optional[float]:
            if key not in model_d or model_d[key] is None:
                return None
            return _as_float(model_d[key], name=f"model.{key}")

        # Parse rope_scaling dict if present
        rope_scaling_raw = model_d.get("rope_scaling")
        rope_scaling: Optional[Dict[str, Any]] = None
        if rope_scaling_raw is not None and isinstance(rope_scaling_raw, dict):
            rope_scaling = {str(k): v for k, v in rope_scaling_raw.items()}

        model = ModelConfig(
            architecture=arch,
            n_positions=n_positions,
            n_embd=_as_int(model_d["n_embd"], name="model.n_embd"),
            n_layer=_as_int(model_d["n_layer"], name="model.n_layer"),
            n_head=_as_int(model_d["n_head"], name="model.n_head"),
            rotary_pct=_as_float(model_d.get("rotary_pct"), name="model.rotary_pct", default=0.25),
            rope_theta=_as_float(model_d.get("rope_theta"), name="model.rope_theta", default=10000.0),
            rope_scaling=rope_scaling,
            dropout=_as_float(model_d.get("dropout"), name="model.dropout", default=0.0),
            use_parallel_residual=_as_bool(model_d.get("use_parallel_residual"), name="model.use_parallel_residual", default=False),
            embd_pdrop=_opt_float("embd_pdrop"),
            attn_pdrop=_opt_float("attn_pdrop"),
            resid_pdrop=_opt_float("resid_pdrop"),
            model_name_or_path=str(model_d["model_name_or_path"]) if model_d.get("model_name_or_path") else None,
        )

        train_d = _get_section(d, "train")
        lr_scheduler_kwargs_raw = train_d.get("lr_scheduler_kwargs", None)
        if lr_scheduler_kwargs_raw is None:
            lr_scheduler_kwargs: Dict[str, Any] = {}
        else:
            if not isinstance(lr_scheduler_kwargs_raw, dict):
                raise ValueError("train.lr_scheduler_kwargs must be a dict if provided")
            lr_scheduler_kwargs = {str(k): v for k, v in lr_scheduler_kwargs_raw.items()}
        train = TrainConfig(
            max_steps=_as_int(train_d.get("max_steps"), name="train.max_steps", default=-1),
            num_train_epochs=_as_float(train_d.get("num_train_epochs"), name="train.num_train_epochs", default=1.0),
            per_device_train_batch_size=_as_int(
                train_d.get("per_device_train_batch_size"), name="train.per_device_train_batch_size", default=8
            ),
            gradient_accumulation_steps=_as_int(
                train_d.get("gradient_accumulation_steps"), name="train.gradient_accumulation_steps", default=1
            ),
            optim=str(train_d.get("optim") or "adamw_torch"),
            adam_beta1=_as_float(train_d.get("adam_beta1"), name="train.adam_beta1", default=0.9),
            adam_beta2=_as_float(train_d.get("adam_beta2"), name="train.adam_beta2", default=0.999),
            adam_epsilon=_as_float(train_d.get("adam_epsilon"), name="train.adam_epsilon", default=1e-8),
            max_grad_norm=_as_float(train_d.get("max_grad_norm"), name="train.max_grad_norm", default=1.0),
            learning_rate=_as_float(train_d.get("learning_rate"), name="train.learning_rate", default=3e-4),
            weight_decay=_as_float(train_d.get("weight_decay"), name="train.weight_decay", default=0.0),
            warmup_ratio=_as_float(train_d.get("warmup_ratio"), name="train.warmup_ratio", default=0.05),
            warmup_steps=_as_int(train_d.get("warmup_steps"), name="train.warmup_steps", default=0),
            lr_scheduler_type=str(train_d.get("lr_scheduler_type") or "cosine"),
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            logging_steps=_as_int(train_d.get("logging_steps"), name="train.logging_steps", default=10),
            save_steps=_as_int(train_d.get("save_steps"), name="train.save_steps", default=500),
            save_total_limit=_as_int(train_d.get("save_total_limit"), name="train.save_total_limit", default=3),
            bf16=_as_bool(train_d.get("bf16"), name="train.bf16", default=False),
            fp16=_as_bool(train_d.get("fp16"), name="train.fp16", default=False),
            gradient_checkpointing=_as_bool(
                train_d.get("gradient_checkpointing"), name="train.gradient_checkpointing", default=False
            ),
            resume=_as_bool(train_d.get("resume"), name="train.resume", default=True),
            overwrite_output_dir=_as_bool(train_d.get("overwrite_output_dir"), name="train.overwrite_output_dir", default=False),
        )

        eval_d = _get_section(d, "eval")
        splits_raw = eval_d.get("splits", None)
        if splits_raw is None:
            eval_splits = ["val", "test"]
        else:
            if not isinstance(splits_raw, list):
                raise ValueError("eval.splits must be a list of split names (e.g. ['val_id','val_ood'])")
            eval_splits = [str(s) for s in splits_raw]
            if not eval_splits:
                raise ValueError("eval.splits must be non-empty if provided")
            bad = [s for s in eval_splits if not s.strip()]
            if bad:
                raise ValueError(f"eval.splits contains empty split name(s): {bad!r}")
        eval_cfg = EvalConfig(
            eval_interval=_as_int(eval_d.get("eval_interval"), name="eval.eval_interval", default=500),
            train_subset_size=_as_int(eval_d.get("train_subset_size"), name="eval.train_subset_size", default=2048),
            splits=eval_splits,
        )

        wandb_d = _get_section(d, "wandb")
        wandb_cfg = WandbConfig(
            project=str(wandb_d.get("project", "")),
            entity=str(wandb_d.get("entity", "")),
            run_name=str(wandb_d.get("run_name", "")),
        )

        return ExperimentConfig(
            seed=seed,
            paths=paths,
            data=data,
            model=model,
            train=train,
            eval=eval_cfg,
            wandb=wandb_cfg,
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML config into an ExperimentConfig."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    config = yaml.safe_load(p.read_text(encoding="utf-8"))
    return {} if config is None else ExperimentConfig.from_dict(config)


__all__ = [
    "ExperimentConfig",
    "load_config",
]


