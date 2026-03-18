import os
import yaml
from pathlib import Path
from textwrap import dedent
from training.constants import (
    PROJECT_ROOT, CONDOR_LOGS_DIR,
    COLOR_TOKENIZER_DIR, COLOR_DATA_BASE,
)

# Output directories
BASE_DIR = Path(__file__).parent
GENERATED_DIR = BASE_DIR / "generated_sweep"
CONFIG_GEN_DIR = GENERATED_DIR / "configs"
SH_GEN_DIR = GENERATED_DIR / "scripts"
SUB_GEN_DIR = GENERATED_DIR / "condor"

for d in [CONFIG_GEN_DIR, SH_GEN_DIR, SUB_GEN_DIR]:
    d.mkdir(parents=True, exist_ok=True)


DATA_CONFIGS = {
    "wf": {
        "jsonl_path": f"{COLOR_DATA_BASE}/wf/tokenized.jsonl",
        "splits_path": f"{COLOR_DATA_BASE}/wf/tokenized.splits.json",
    },
    "nwf": {
        "jsonl_path": f"{COLOR_DATA_BASE}/nwf/tokenized.jsonl",
        "splits_path": f"{COLOR_DATA_BASE}/nwf/tokenized.splits.json",
    },
}


SEEDS = [0, 1, 2, 3]
EXPERIMENTS = []

for data in ["wf", "nwf"]:
    for seed in SEEDS:
        EXPERIMENTS.append({
            "data": data,
            "seed": seed,
        })

DEFAULTS = {
    "n_layer": 8,
    "n_embd": 768,
    "n_head": 12,
    "lr": 3e-4,
    "wd": 0.1,
    "batch_size": 8,
    "reset_pos_id": None,
    "max_grad_norm": 0.5,
    "seed": 0,
    "max_steps": 50000,
    "save_steps": 1000,
    "eval_interval": 1000,
    "block_size": 4096,
}


# =============================================================================
# Helper functions
# =============================================================================

def format_lr(lr: float) -> str:
    if lr >= 1e-3:
        return f"{round(lr * 1e3)}e-3"
    elif lr >= 1e-4:
        return f"{round(lr * 1e4)}e-4"
    return f"{round(lr * 1e5)}e-5"


def make_experiment(exp: dict) -> tuple[str, dict]:
    """Generate (name, config) for one experiment."""
    e = {**DEFAULTS, **exp}
    data_cfg = DATA_CONFIGS[e["data"]]

    # Build name
    name = f"{e['data']}_l{e['n_layer']}_h{e['n_embd']}_lr{format_lr(e['lr'])}_bs{e['batch_size']}_wd{e['wd']}_nope"

    # Add reset_pos_id to name if set
    if e["reset_pos_id"] is not None:
        name += f"_{e['reset_pos_id']}"

    name += f"_s{e['seed']}"

    config = {
        "seed": e["seed"],
        "paths": {
            "jsonl_path": data_cfg["jsonl_path"],
            "tokenizer_dir": COLOR_TOKENIZER_DIR,
            "output_dir": f"scratch_data/training_outputs/color_new_config/{name}",
            "splits_path": data_cfg["splits_path"],
        },
        "data": {
            "train_mode": "lm_no_break",
            "train_label_mode": "autoregressive",
            "block_size": e["block_size"],
            "per_device_eval_batch_size": 32,
            "eval_num_workers": 10,
            "allow_cross_sample_attention": False,
            "reset_pos_id": e["reset_pos_id"],
        },
        "model": {
            "architecture": "gpt-neox-nope",
            "n_positions": e["block_size"],
            "n_embd": e["n_embd"],
            "n_layer": e["n_layer"],
            "n_head": e["n_head"],
            "dropout": 0.0,
            "model_name_or_path": None,
        },
        "train": {
            "max_steps": e["max_steps"],
            "per_device_train_batch_size": e["batch_size"],
            "gradient_accumulation_steps": 1,
            "learning_rate": e["lr"],
            "weight_decay": e["wd"],
            "warmup_ratio": 0.1,
            "warmup_steps": 0,
            "lr_scheduler_type": "linear",
            "max_grad_norm": e["max_grad_norm"],
            "logging_steps": 10,
            "save_steps": e["save_steps"],
            "save_total_limit": 3,
            "bf16": True,
            "fp16": False,
            "gradient_checkpointing": False,
            "resume": True,
            "overwrite_output_dir": False,
        },
        "eval": {
            "eval_interval": e["eval_interval"],
            "train_subset_size": 2000,
            "splits": ["val_id", "val_ood", "test_id", "test_ood"],
        },
        "wandb": {
            "project": "plan-verification-color-new-config",
            "run_name": name,
        },
    }
    return name, config


def make_run_sh(config_path: str, name: str) -> str:
    config_rel = Path(config_path).relative_to(PROJECT_ROOT)
    content = dedent(f"""\
        #!/usr/bin/env bash
        cd {PROJECT_ROOT}
        source /nethome/yudo/condor_scripts/rename_gpus.sh
        nvidia-smi
        echo $CUDA_VISIBLE_DEVICES
        echo $HOSTNAME
        source "/scratch/yudo/miniconda3/etc/profile.d/conda.sh"
        conda activate llm_plan_verification_env
        export PYTHONPATH={PROJECT_ROOT}${{PYTHONPATH:+:$PYTHONPATH}}
        python -m final_model_training.train --config {config_rel}
    """)
    path = SH_GEN_DIR / f"{name}.sh"
    path.write_text(content)
    os.chmod(path, 0o755)
    return str(path)


def make_condor_sub(name: str, sh_path: str) -> str:
    sh_rel = Path(sh_path).relative_to(PROJECT_ROOT)
    condor_name = f"color_{name}"
    gpu_mem = 40000
    content = dedent(f"""\
        universe                = vanilla
        initialdir              = {PROJECT_ROOT}
        executable              = /bin/bash
        arguments               = "{sh_rel}"
        output                  = {CONDOR_LOGS_DIR}/{condor_name}.$(ClusterId).out
        error                   = {CONDOR_LOGS_DIR}/{condor_name}.$(ClusterId).err
        log                     = {CONDOR_LOGS_DIR}/{condor_name}.$(ClusterId).log
        request_CPUs            = 8
        request_memory          = 80G
        request_GPUs            = 1
        requirements            = (GPUs_GlobalMemoryMb >= {gpu_mem}) && (TARGET.UidDomain == "coli.uni-saarland.de")
        batch_name              = {condor_name}
        getenv                  = True
        +MaxWallTime            = 604800
        accounting_group        = pausable
        queue 1
    """)
    path = SUB_GEN_DIR / f"{condor_name}.sub"
    path.write_text(content)
    return str(path)


def main():
    if not EXPERIMENTS:
        print("No experiments defined.")
        return

    sub_paths = []

    for exp in EXPERIMENTS:
        name, config = make_experiment(exp)

        cfg_path = CONFIG_GEN_DIR / f"{name}.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        sh_path = make_run_sh(str(cfg_path), name)
        sub_path = make_condor_sub(name, sh_path)
        sub_paths.append(sub_path)
        print(f"Created: {name}")

    # Master submit script
    submit_all = GENERATED_DIR / "submit_all.sh"
    lines = ["#!/usr/bin/env bash", f"# Submit all sweep jobs from: {PROJECT_ROOT}", ""]
    lines += [f"condor_submit {Path(p).relative_to(PROJECT_ROOT)}" for p in sub_paths]
    submit_all.write_text("\n".join(lines) + "\n")
    os.chmod(submit_all, 0o755)

    print(f"\nGenerated {len(sub_paths)} experiments in {GENERATED_DIR}")
    print(f"Run: bash {submit_all.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
