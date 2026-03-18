#!/usr/bin/env python3
"""
Generate Color Bags training data.

Data splits:
- Training:    plan lengths 11-100, n_per_len scaled by diversity
- Test in-dist: plan lengths 11-100, 200 per length
- Test OOD:    plan lengths 101-200, 200 per length
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from training.constants import COLOR_DATA_BASE, COLOR_TOKENIZER_DIR
from data_generation.color.generator import (
    generate_pairs_for_length,
    get_or_create_tokenizer,
    build_vocab,
)

CONFIG_GROUPS: Dict[Tuple[int, int], Dict] = {
    (3, 2): {'lengths': list(range(11, 28)), 'n_per_len': 8000},
    (3, 3): {'lengths': list(range(28, 36)), 'n_per_len': 10000},
    (4, 3): {'lengths': list(range(36, 52)), 'n_per_len': 10000},
    (4, 4): {'lengths': list(range(52, 64)), 'n_per_len': 10000},
    (5, 4): {'lengths': list(range(64, 84)), 'n_per_len': 10000},
    (5, 5): {'lengths': list(range(84, 100)), 'n_per_len': 10000},
    (6, 5): {'lengths': list(range(100, 101)), 'n_per_len': 10000},
}

TEST_N_PER_LEN = 200
CHUNK_SIZE = 10


def get_n_per_len_for_training(plan_length: int) -> int:
    for _, data in CONFIG_GROUPS.items():
        if plan_length in data['lengths']:
            return data['n_per_len']
    return 10000


def get_output_dir(split: str, well_formed: bool, min_len: int, max_len: int) -> str:
    wf_str = 'wf' if well_formed else 'nwf'
    return os.path.join(COLOR_DATA_BASE, split, f'{wf_str}_{min_len}_{max_len}')


def run_generation_job(split: str, well_formed: bool, min_len: int, max_len: int, n_per_len: int) -> Tuple[bool, str, int]:
    output_dir = get_output_dir(split, well_formed, min_len, max_len)
    output_file = os.path.join(output_dir, 'data.jsonl')
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_file):
        os.remove(output_file)

    vocab = build_vocab()
    total_written = 0

    try:
        with open(output_file, "w") as f:
            for plan_length in range(min_len, max_len + 1):
                for correct_ex, incorrect_ex in generate_pairs_for_length(
                    plan_length=plan_length,
                    n_pairs=n_per_len,
                    well_formed=well_formed,
                    incorrect_type="incomplete",
                    incorrect_position="random",
                    vocab=vocab,
                ):
                    for example in (correct_ex, incorrect_ex):
                        f.write(json.dumps(example) + "\n")
                        total_written += 1

        return True, output_file, total_written
    except Exception as e:
        return False, str(e), 0


def create_training_jobs() -> List[Tuple[str, bool, int, int, int]]:
    jobs = []
    for well_formed in [True, False]:
        for chunk_start in range(11, 101, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE - 1, 100)
            mid_len = (chunk_start + chunk_end) // 2
            n_per_len = get_n_per_len_for_training(mid_len)
            jobs.append(('train', well_formed, chunk_start, chunk_end, n_per_len))
    return jobs


def create_test_jobs() -> List[Tuple[str, bool, int, int, int]]:
    jobs = []

    for well_formed in [True, False]:
        for chunk_start in range(11, 101, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE - 1, 100)
            jobs.append(('test_indist', well_formed, chunk_start, chunk_end, TEST_N_PER_LEN))

    for well_formed in [True, False]:
        for chunk_start in range(101, 201, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE - 1, 200)
            jobs.append(('test_ood', well_formed, chunk_start, chunk_end, TEST_N_PER_LEN))

    return jobs


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Color Bags data')
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='all')
    parser.add_argument('--max_workers', type=int, default=20)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    print(f"Data dir: {COLOR_DATA_BASE}")
    print(f"Tokenizer: {COLOR_TOKENIZER_DIR}")

    jobs = []
    if args.split in ('train', 'all'):
        jobs.extend(create_training_jobs())
    if args.split in ('test', 'all'):
        jobs.extend(create_test_jobs())

    print(f"Total jobs: {len(jobs)}")

    total_pairs = sum((max_len - min_len + 1) * n_per_len for _, _, min_len, max_len, n_per_len in jobs)
    print(f"Total pairs: {total_pairs:,}")
    print(f"Total examples: {total_pairs * 2:,}")

    if args.dry_run:
        print("\nDry run - jobs:")
        for split, wf, min_len, max_len, n_per_len in jobs:
            wf_str = 'WF' if wf else 'NWF'
            print(f"  {split} {wf_str} len={min_len}-{max_len} n_per_len={n_per_len}")
        return

    print(f"\nCreating tokenizer at {COLOR_TOKENIZER_DIR}...")
    get_or_create_tokenizer(tokenizer_dir=COLOR_TOKENIZER_DIR, rebuild=False)
    print("Tokenizer ready.")

    print(f"Running {len(jobs)} jobs with {args.max_workers} workers...")

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_generation_job, split, wf, min_len, max_len, n_per_len):
            (split, wf, min_len, max_len, n_per_len)
            for split, wf, min_len, max_len, n_per_len in jobs
        }

        completed = 0
        failed = 0
        total_examples = 0
        for future in as_completed(futures):
            split, wf, min_len, max_len, n_per_len = futures[future]
            wf_str = 'WF' if wf else 'NWF'
            completed += 1

            try:
                success, msg, count = future.result()
                if success:
                    total_examples += count
                    print(f'[{completed}/{len(jobs)}] OK: {split} {wf_str} {min_len}-{max_len} ({count} examples)')
                else:
                    failed += 1
                    print(f'[{completed}/{len(jobs)}] FAILED: {split} {wf_str} {min_len}-{max_len}: {msg}')
            except Exception as e:
                failed += 1
                print(f'[{completed}/{len(jobs)}] ERROR: {split} {wf_str} {min_len}-{max_len}: {e}')

    print(f"\nCompleted: {completed - failed}/{len(jobs)} succeeded, {failed} failed")
    print(f"Total examples written: {total_examples:,}")


if __name__ == '__main__':
    main()
