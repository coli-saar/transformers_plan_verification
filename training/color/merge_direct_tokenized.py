import argparse
import json
import os
import random
from typing import Dict, List, Optional


def _read_jsonl_files(jsonl_paths: Optional[List[str]]) -> List[Dict]:
    if not jsonl_paths:
        return []
    records = []
    for path in jsonl_paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Merge tokenized Color Bags datasets and build split file"
    )
    parser.add_argument("--train_jsonls", nargs="+", default=None)
    parser.add_argument("--id_jsonls", nargs="+", default=None)
    parser.add_argument("--ood_jsonls", nargs="+", default=None)
    parser.add_argument("--ood_longer_jsonls", nargs="+", default=None)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_split_json", default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    input_jsonl_dict = {
        "train": args.train_jsonls,
        "id": args.id_jsonls,
        "ood": args.ood_jsonls,
        "ood_longer": args.ood_longer_jsonls,
    }

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data_split_dict: Dict[str, List[int]] = {k: [] for k in input_jsonl_dict.keys()}
    example_id = 0

    with open(args.output_jsonl, "w") as fout:
        for split, jsonl_paths in input_jsonl_dict.items():
            records = _read_jsonl_files(jsonl_paths)
            for rec in records:
                rec = dict(rec)
                rec["example_id"] = example_id
                fout.write(json.dumps(rec) + "\n")
                data_split_dict[split].append(example_id)
                example_id += 1

    # Create val/test splits for id and ood, like pretokenize.py
    rng = random.Random(args.seed)
    for split in ["id", "ood"]:
        example_ids = data_split_dict.get(split, [])
        if not example_ids:
            continue
        rng.shuffle(example_ids)
        val_ids = example_ids[: len(example_ids) // 3]
        test_ids = example_ids[len(example_ids) // 3 :]
        del data_split_dict[split]
        data_split_dict[f"val_{split}"] = val_ids
        data_split_dict[f"test_{split}"] = test_ids

    output_split_json = args.output_split_json
    if output_split_json is None:
        output_split_json = args.output_jsonl.replace(".jsonl", ".splits.json")

    with open(output_split_json, "w") as fsplit:
        json.dump(data_split_dict, fsplit, indent=4)

    print(f"Output: {args.output_jsonl}")
    print(f"Output splits: {output_split_json}")
    print(f"Total examples: {example_id}")


if __name__ == "__main__":
    main()
