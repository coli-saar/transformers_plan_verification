#!/usr/bin/env python3
"""Generate Condor jobs for detailed evaluation of all dataset variants."""

import os
from pathlib import Path
from textwrap import dedent
from training.constants import PROJECT_ROOT, CONDOR_LOGS_DIR

BASE_DIR = "scratch_data/training_outputs"
OUT_DIR = "scratch_data/eval_raw_results"
GENERATED_DIR = Path(__file__).parent / "generated_eval"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Dataset variants: (name, run_dir_template)
VARIANTS = [
    ("lights_out_wf", f"{BASE_DIR}/lights_out_exp_anonym_loc/incomplete_l8_h768_lr2e-5_bs8_wd0.1_ape_gn1_s{{s}}_ms50000_csa"),
    ("lights_out_nwf", f"{BASE_DIR}/lights_out_regular/default_l8_h768_lr2e-5_bs8_wd0.1_ape_s{{s}}_csa"),
    ("color_bag_wf", f"{BASE_DIR}/color_new_config/3_wf_l8_h768_lr3e-4_bs8_wd0.1_ape_s{{s}}"),
    ("color_bag_nwf", f"{BASE_DIR}/color_new_config/3_nwf_l8_h768_lr3e-4_bs8_wd0.1_ape_s{{s}}"),
    ("grippers_wf", f"{BASE_DIR}/grippers_new/3_wf_l8_h768_lr6e-4_bs8_wd0.1_ape_s{{s}}"),
    ("grippers_df", f"{BASE_DIR}/grippers_new/3_df_l8_h768_lr6e-4_bs8_wd0.1_ape_s{{s}}"),
]

SEEDS = [
    0, 
    1, 
    2, 
    3,
]


def make_run_sh(name: str, run_dir_template: str) -> str:
    """Generate shell script that runs all seeds for one variant."""
    eval_cmds = []
    for s in SEEDS:
        run_dir = run_dir_template.format(s=s)
        csv_path = f"{OUT_DIR}/{name}_s{s}.csv"
        eval_cmds.append(
            f"python -m final_model_training.detailed_eval --run_dir {run_dir} --csv_path {csv_path}"
        )
    
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
        mkdir -p {OUT_DIR}
        
    """) + "\n".join(eval_cmds) + "\n"
    
    path = GENERATED_DIR / f"eval_{name}.sh"
    path.write_text(content)
    os.chmod(path, 0o755)
    return str(path)


def make_condor_sub(name: str, sh_path: str) -> str:
    """Generate Condor submission file."""
    sh_rel = Path(sh_path).relative_to(PROJECT_ROOT)
    condor_name = f"eval_{name}"
    content = dedent(f"""\
        universe                = vanilla
        initialdir              = {PROJECT_ROOT}
        executable              = /bin/bash
        arguments               = "{sh_rel}"
        output                  = {CONDOR_LOGS_DIR}/{condor_name}.$(ClusterId).out
        error                   = {CONDOR_LOGS_DIR}/{condor_name}.$(ClusterId).err
        log                     = {CONDOR_LOGS_DIR}/{condor_name}.$(ClusterId).log
        request_CPUs            = 10
        request_memory          = 80G
        request_GPUs            = 1
        requirements            = (GPUs_GlobalMemoryMb >= 40000) && (TARGET.UidDomain == "coli.uni-saarland.de")
        batch_name              = {condor_name}
        getenv                  = True
        accounting_group        = pausable
        queue 1
    """)
    path = GENERATED_DIR / f"{condor_name}.sub"
    path.write_text(content)
    return str(path)


def main():
    sub_paths = []
    
    for name, run_dir_template in VARIANTS:
        sh_path = make_run_sh(name, run_dir_template)
        sub_path = make_condor_sub(name, sh_path)
        sub_paths.append(sub_path)
        print(f"Created: {name}")
    
    # Master submit script
    submit_all = GENERATED_DIR / "submit_all.sh"
    lines = ["#!/usr/bin/env bash", f"cd {PROJECT_ROOT}", ""]
    lines += [f"condor_submit {Path(p).relative_to(PROJECT_ROOT)}" for p in sub_paths]
    submit_all.write_text("\n".join(lines) + "\n")
    os.chmod(submit_all, 0o755)
    
    print(f"\nGenerated {len(sub_paths)} jobs in {GENERATED_DIR}")
    print(f"Run: bash {submit_all.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
