import argparse
import json
import os
import re
import shutil
import random
from typing import Dict, List

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFKC, Sequence as NormSequence
from tokenizers.pre_tokenizers import Sequence as PreTokSequence, WhitespaceSplit
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from training.constants import VERDICT_CORRECT_TOKEN, VERDICT_INCORRECT_TOKEN, LIGHTS_OUT_REGULAR_TOKENIZER_DIR

DEFAULT_TOKENIZER_DIR = LIGHTS_OUT_REGULAR_TOKENIZER_DIR

# Maximum grid size to support (numbers 0 to MAX_GRID_SIZE-1)
MAX_GRID_SIZE = 5

# Regex patterns
RE_INIT_BLOCK = re.compile(r"\(:init\b([\s\S]*?)\(:goal\b", re.IGNORECASE)
RE_LIGHT_ON = re.compile(r"\(light_on\s+button_(\d+)_(\d+)\)", re.IGNORECASE)
RE_LIGHT_OUT = re.compile(r"\(light_out\s+button_(\d+)_(\d+)\)", re.IGNORECASE)
RE_PRESS_BUTTON = re.compile(r"\(press_button\s+button_(\d+)_(\d+)\)", re.IGNORECASE)


def build_vocab() -> Dict[str, int]:
    """Build fixed vocab for the Lights Out regular domain.
    
    Vocab includes:
    - Numbers 0 to MAX_GRID_SIZE-1 (for row/col indices)
    - Keywords: on, out, press
    - Special tokens: <pad>, <bos>, <eos>, <unk>, <init>, <plan>, <answer>, verdict tokens
    """
    vocab: Dict[str, int] = {}
    idx = 0

    # Add number tokens for grid coordinates
    for n in range(MAX_GRID_SIZE):
        vocab[str(n)] = idx
        idx += 1

    # Add keyword tokens
    for tok in ["on", "out", "press"]:
        vocab[tok] = idx
        idx += 1

    # Add special tokens
    for tok in ["<pad>", "<bos>", "<eos>", "<unk>",
                "<init>", "<plan>", "<answer>",
                VERDICT_CORRECT_TOKEN, VERDICT_INCORRECT_TOKEN]:
        vocab[tok] = idx
        idx += 1

    return vocab


def build_tokenizer() -> PreTrainedTokenizerFast:
    """Build a rule-based WordLevel tokenizer for Lights Out regular domain."""
    vocab = build_vocab()
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.normalizer = NormSequence([NFKC()])
    tok.pre_tokenizer = PreTokSequence([WhitespaceSplit()])

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", pad_token="<pad>",
    )
    hf_tok.add_special_tokens({"additional_special_tokens": [
        "<init>", "<plan>", "<answer>",
        VERDICT_CORRECT_TOKEN, VERDICT_INCORRECT_TOKEN
    ]})
    return hf_tok


def get_or_create_tokenizer(tokenizer_dir: str, rebuild: bool) -> PreTrainedTokenizerFast:
    """Load existing tokenizer or create new one."""
    if rebuild and os.path.exists(tokenizer_dir):
        shutil.rmtree(tokenizer_dir)

    if os.path.exists(tokenizer_dir):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer = build_tokenizer()
        tokenizer.save_pretrained(tokenizer_dir)

    return tokenizer


def tokens_to_ids(tokens: List[str], tokenizer) -> List[int]:
    """Convert tokens to ids. Raises on unknown token."""
    ids = []
    for t in tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid == tokenizer.unk_token_id and t != "<unk>":
            raise ValueError(f"Unknown token: {t!r}")
        ids.append(tid)
    return ids


def parse_problem(problem_pddl: str) -> List[str]:
    """Extract light_on/light_out facts as tokens.
    
    Returns tokens like: ["on", "0", "1", "on", "1", "2", ..., "out", "0", "0", ...]
    """
    m = RE_INIT_BLOCK.search(problem_pddl)
    init_text = m.group(1) if m else problem_pddl

    tokens = []
    # Extract and sort light_on facts
    for r, c in sorted((int(m.group(1)), int(m.group(2))) for m in RE_LIGHT_ON.finditer(init_text)):
        tokens.extend(["on", str(r), str(c)])
    # Extract and sort light_out facts
    for r, c in sorted((int(m.group(1)), int(m.group(2))) for m in RE_LIGHT_OUT.finditer(init_text)):
        tokens.extend(["out", str(r), str(c)])
    return tokens


def parse_plan(plan_str: str) -> List[str]:
    """Parse plan actions as tokens.
    
    Input: "(press_button button_2_2)\n(press_button button_2_4)\n..."
    Returns tokens like: ["press", "2", "2", "press", "2", "4", ...]
    """
    tokens = []
    # Find all press_button actions in order
    for m in RE_PRESS_BUTTON.finditer(plan_str):
        r, c = m.group(1), m.group(2)
        tokens.extend(["press", r, c])
    return tokens


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(description="Tokenize Lights Out Regular JSONL")
    parser.add_argument("--train_jsonls", nargs="+", default=None)
    parser.add_argument("--id_jsonls", nargs="+", default=None)
    parser.add_argument("--ood_jsonls", nargs="+", default=None)
    parser.add_argument("--ood_longer_jsonls", nargs="+", default=None)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_split_json", default=None)

    parser.add_argument("--tokenizer_dir", default=DEFAULT_TOKENIZER_DIR)

    parser.add_argument("--rebuild", action="store_true", help="Rebuild tokenizer from scratch")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=5000)
    args = parser.parse_args()

    tokenizer = get_or_create_tokenizer(args.tokenizer_dir, args.rebuild)

    # Get special token ids
    init_id = tokenizer.convert_tokens_to_ids("<init>")
    plan_id = tokenizer.convert_tokens_to_ids("<plan>")
    answer_id = tokenizer.convert_tokens_to_ids("<answer>")
    correct_id = tokenizer.convert_tokens_to_ids(VERDICT_CORRECT_TOKEN)
    incorrect_id = tokenizer.convert_tokens_to_ids(VERDICT_INCORRECT_TOKEN)

    out_dir = os.path.dirname(args.output_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_total = n_correct = n_incorrect = 0

    input_jsonl_dict = {
        "train": args.train_jsonls,
        "id": args.id_jsonls,
        "ood": args.ood_jsonls,
        "ood_longer": args.ood_longer_jsonls
    }
    data_split_dict = {k: [] for k in input_jsonl_dict.keys()}

    with open(args.output_jsonl, "w") as fout:
        n_total = 0
        for split, input_jsonls in input_jsonl_dict.items():
            if input_jsonls is None:
                print(f"No {split} jsonls provided, skipping")
                continue

            for input_jsonl in input_jsonls:
                with open(input_jsonl) as fin:
                    for line in fin:
                        if args.max_examples and n_total >= args.max_examples:
                            break

                        rec = json.loads(line)
                        is_correct = rec["incorrect_action_index"] is None

                        prob_ids = tokens_to_ids(parse_problem(rec["problem"]), tokenizer)
                        plan_ids = tokens_to_ids(parse_plan(rec["plan"]), tokenizer)
                        verdict_id = correct_id if is_correct else incorrect_id

                        # Build sequence: <init> init <plan> plan <answer> verdict
                        input_ids = (
                            [init_id] + prob_ids +
                            [plan_id] + plan_ids +
                            [answer_id, verdict_id]
                        )

                        out_rec = {
                            "example_id": n_total,
                            "plan_len": rec["plan_length"],
                            "is_correct": is_correct,
                            "input_ids": input_ids,
                        }
                        fout.write(json.dumps(out_rec) + "\n")
                        data_split_dict[split].append(n_total)

                        n_total += 1
                        if is_correct:
                            n_correct += 1
                        else:
                            n_incorrect += 1
                        if n_total % args.progress_every == 0:
                            print(f"Processed {n_total} (correct={n_correct}, incorrect={n_incorrect})")

    # Create train/val/test splits
    splits = list(data_split_dict.keys())
    for split in splits:
        example_ids = data_split_dict[split]
        if split == "train" or split == "ood_longer":
            continue
        random.shuffle(example_ids)
        val_ids, test_ids = example_ids[:len(example_ids)//3], example_ids[len(example_ids)//3:]
        del data_split_dict[split]
        data_split_dict[f"val_{split}"] = val_ids
        data_split_dict[f"test_{split}"] = test_ids

    output_split_json = args.output_split_json if args.output_split_json else args.output_jsonl.replace(".jsonl", ".splits.json")
    with open(output_split_json, "w") as fsplit:
        json.dump(data_split_dict, fsplit, indent=4)

    print(f"Tokenizer: {args.tokenizer_dir}")
    print(f"Output: {args.output_jsonl}")
    print(f"Output splits: {output_split_json}")
    print(f"Total: {n_total} (correct={n_correct}, incorrect={n_incorrect})")

    # Print first example (original, tokenized, decoded)
    first_jsonl = args.train_jsonls[0] if args.train_jsonls else (args.id_jsonls[0] if args.id_jsonls else args.ood_jsonls[0])
    print("\n--- Example ---")
    with open(first_jsonl) as fin_input, open(args.output_jsonl) as fin_output:
        # 1. Original unprocessed input
        input_json = json.loads(next(fin_input).strip())
        print("\n--- Original Input ---")
        print("Problem: ", input_json["problem"])
        print("Plan: ", input_json["plan"])

        # 2. Tokenized output version
        print("\n--- Tokenized Input IDs ---")
        output_line = json.loads(next(fin_output).strip())["input_ids"]
        print("Input IDs: ", output_line)

        # 3. Decoded tokens from input_ids
        print("\n--- Output Input IDs Decoded to Tokens ---")
        print(" ".join(tokenizer.convert_ids_to_tokens(output_line)))


if __name__ == "__main__":
    main()
