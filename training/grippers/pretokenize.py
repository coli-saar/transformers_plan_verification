import argparse
import json
import os
import re
import shutil
import random
from typing import Dict, List, Tuple

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFKC, Sequence as NormSequence
from tokenizers.pre_tokenizers import Sequence as PreTokSequence, WhitespaceSplit
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from training.constants import VERDICT_CORRECT_TOKEN, VERDICT_INCORRECT_TOKEN, GRIPPERS_TOKENIZER_DIR


DEFAULT_TOKENIZER_DIR = GRIPPERS_TOKENIZER_DIR

MAX_ROOM_ID = 300
MAX_BALL_ID = 300
MAX_GRIPPER_ID = 2
MAX_OBJECT_ID = MAX_GRIPPER_ID + MAX_BALL_ID + MAX_ROOM_ID


# Regex helpers - support both ball_10 and ball10 formats
def _obj_pattern() -> str:
    return rf"(?:{object}_?)?(\d+)"

def _obj_pattern_plan(prefix: str) -> str:
    return rf"(?:{prefix}_?)?(\d+)"

# PDDL block patterns
_RE_INIT_BLOCK = re.compile(r"\(:init\b([\s\S]*?)\)\s*\(:goal\b", re.IGNORECASE)
_RE_GOAL_BLOCK = re.compile(r"\(:goal\b([\s\S]*)", re.IGNORECASE)

# Init fact patterns
_RE_AT_ROBBY = re.compile(rf"\(at-robby\s+{_obj_pattern()}\)", re.IGNORECASE)
_RE_AT = re.compile(rf"\(at\s+{_obj_pattern()}\s+{_obj_pattern()}\)", re.IGNORECASE)
_RE_FREE = re.compile(rf"\(free\s+{_obj_pattern()}\)", re.IGNORECASE)
_RE_CARRY = re.compile(rf"\(carry\s+{_obj_pattern()}\s+{_obj_pattern()}\)", re.IGNORECASE)
_RE_HEAVY = re.compile(rf"\(heavy\s+{_obj_pattern()}\)", re.IGNORECASE)
_RE_ROBBY_CHARGED = re.compile(r"\(robby-charged\)", re.IGNORECASE)
_RE_ROOM_DECL = re.compile(rf"\(room\s+{_obj_pattern()}\)", re.IGNORECASE)
_RE_BALL_DECL = re.compile(rf"\(ball\s+{_obj_pattern()}\)", re.IGNORECASE)
_RE_GRIPPER_DECL = re.compile(rf"\(gripper\s+{_obj_pattern()}\)", re.IGNORECASE)


# Plan action patterns
_RE_MOVE = re.compile(rf"\(move\s+{_obj_pattern_plan('object')}\s+{_obj_pattern_plan('object')}\)", re.IGNORECASE)
_RE_PICK = re.compile(rf"\(pick\s+{_obj_pattern_plan('object')}\s+{_obj_pattern_plan('object')}\s+{_obj_pattern_plan('object')}\)", re.IGNORECASE)
_RE_PICK_HEAVY = re.compile(rf"\(pick_heavy\s+{_obj_pattern_plan('object')}\s+{_obj_pattern_plan('object')}\s+{_obj_pattern_plan('object')}\)", re.IGNORECASE)
_RE_DROP = re.compile(rf"\(drop\s+{_obj_pattern_plan('object')}\s+{_obj_pattern_plan('object')}\s+{_obj_pattern_plan('object')}\)", re.IGNORECASE)


def parse_problem(problem_pddl: str, use_declarations: bool = False) -> Tuple[List[str], List[str]]:
    """Extract init facts and goal facts from PDDL, return (init_tokens, goal_tokens)."""
    init_tokens: List[str] = []
    goal_tokens: List[str] = []

    # Extract blocks
    m_init = _RE_INIT_BLOCK.search(problem_pddl)
    init_text = m_init.group(1) if m_init else ""
    init_text = init_text.replace('object_', '')

    m_goal = _RE_GOAL_BLOCK.search(problem_pddl)
    goal_text = m_goal.group(1) if m_goal else ""
    goal_text = goal_text.replace('object_', '')

    # import pdb; pdb.set_trace()
    # Declarations (optional) - type already encoded in token name
    if use_declarations:
        for (rid,) in _extract_sorted(_RE_ROOM_DECL, init_text):
            init_tokens.extend([f"room", f"{rid}"])
        for (bid,) in _extract_sorted(_RE_BALL_DECL, init_text):
            init_tokens.extend([f"ball", f"{bid}"])
        for (gid,) in _extract_sorted(_RE_GRIPPER_DECL, init_text):
            init_tokens.extend([f"gripper", f"{gid}"])

    # Init facts
    for (rid,) in _extract_sorted(_RE_AT_ROBBY, init_text):
        init_tokens.extend(["at-robby", f"{rid}"])
    for bid, rid in _extract_sorted(_RE_AT, init_text):
        init_tokens.extend(["at", f"{bid}", f"{rid}"])
    for (gid,) in _extract_sorted(_RE_FREE, init_text):
        init_tokens.extend(["free", f"{gid}"])
    for bid, gid in _extract_sorted(_RE_CARRY, init_text):
        init_tokens.extend(["carry", f"{bid}", f"{gid}"])
    for (bid,) in _extract_sorted(_RE_HEAVY, init_text):
        init_tokens.extend(["heavy", f"{bid}"])
    if _RE_ROBBY_CHARGED.search(init_text):
        init_tokens.append("robby-charged")

    # Goal facts
    for bid, rid in _extract_sorted(_RE_AT, goal_text):
        goal_tokens.extend(["at", f"{bid}", f"{rid}"])

    return init_tokens, goal_tokens


def parse_plan(plan_str: str) -> List[str]:
    """Parse plan string, return tokens (no semicolons)."""
    # Collect all actions with positions to preserve order
    actions: List[Tuple[int, List[str]]] = []

    for m in _RE_MOVE.finditer(plan_str):
        actions.append((m.start(), ["move", f"{m.group(1)}", f"{m.group(2)}"]))
    for m in _RE_PICK.finditer(plan_str):
        actions.append((m.start(), ["pick", f"{m.group(1)}", f"{m.group(2)}", f"{m.group(3)}"]))
    for m in _RE_PICK_HEAVY.finditer(plan_str):
        actions.append((m.start(), ["pick_heavy", f"{m.group(1)}", f"{m.group(2)}", f"{m.group(3)}"]))
    for m in _RE_DROP.finditer(plan_str):
        actions.append((m.start(), ["drop", f"{m.group(1)}", f"{m.group(2)}", f"{m.group(3)}"]))

    actions.sort(key=lambda x: x[0])
    tokens = [t for _, toks in actions for t in toks]

    return tokens


def build_vocab() -> Dict[str, int]:
    """Fixed vocab for the simplified Grippers representation."""
    vocab: Dict[str, int] = {}
    idx = 0

    # Object tokens
    for i in range(MAX_OBJECT_ID):
        vocab[str(i)] = idx
        idx += 1

    # Domain tokens (no semicolon needed, no type prefixes for declarations)
    for tok in ["at-robby", "at", "free", "carry", "heavy", "robby-charged",
                "move", "pick", "pick_heavy", "drop", "room", "ball", "gripper"]:
        vocab[tok] = idx
        idx += 1

    # Special tokens with clear section markers
    for tok in ["<pad>", "<bos>", "<eos>", "<unk>", 
                "<init>", "<goal>", "<plan>", "<answer>",
                VERDICT_CORRECT_TOKEN, VERDICT_INCORRECT_TOKEN]:
        vocab[tok] = idx
        idx += 1

    return vocab


def build_tokenizer() -> PreTrainedTokenizerFast:
    """Build a rule-based WordLevel tokenizer for Grippers domain."""
    vocab = build_vocab()
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.normalizer = NormSequence([NFKC()])
    tok.pre_tokenizer = PreTokSequence([WhitespaceSplit()])

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", pad_token="<pad>",
    )
    hf_tok.add_special_tokens({"additional_special_tokens": [
        "<init>", "<goal>", "<plan>", "<answer>",
        VERDICT_CORRECT_TOKEN, VERDICT_INCORRECT_TOKEN
    ]})
    return hf_tok


def get_or_create_tokenizer(tokenizer_dir, rebuild):
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


def tokens_to_ids(tokens, tokenizer):
    """Convert tokens to ids. Raises on unknown token."""
    ids = []
    for t in tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid == tokenizer.unk_token_id and t != "<unk>":
            raise ValueError(f"Unknown token: {t!r}")
        ids.append(tid)
    return ids


def _extract_sorted(pattern: re.Pattern, text: str) -> List[Tuple[int, ...]]:
    """Extract all matches, convert groups to ints, return sorted list."""
    return sorted(tuple(int(g) for g in m.groups()) for m in pattern.finditer(text))


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(description="Tokenize Grippers JSONL")
    parser.add_argument("--train_jsonls", nargs="+", default=None)
    parser.add_argument("--id_jsonls", nargs="+", default=None)
    parser.add_argument("--ood_jsonls", nargs="+", default=None)
    parser.add_argument("--ood_longer_jsonls", nargs="+", default=None)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_split_json", default=None)

    parser.add_argument("--tokenizer_dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--use_declarations", action="store_true", help="Include room/ball/gripper declarations")

    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=5000)
    args = parser.parse_args()

    tokenizer = get_or_create_tokenizer(args.tokenizer_dir, args.rebuild)

    # Get special token ids
    init_id = tokenizer.convert_tokens_to_ids("<init>")
    goal_id = tokenizer.convert_tokens_to_ids("<goal>")
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

                        init_tokens, goal_tokens = parse_problem(rec["problem"], args.use_declarations)
                        plan_tokens = parse_plan(rec["plan"])
                        
                        init_ids = tokens_to_ids(init_tokens, tokenizer)
                        goal_ids = tokens_to_ids(goal_tokens, tokenizer)
                        plan_ids = tokens_to_ids(plan_tokens, tokenizer)
                        verdict_id = correct_id if is_correct else incorrect_id

                        # Build sequence: <init> init <plan> plan <goal> goal <answer> verdict
                        input_ids = (
                            [init_id] + init_ids + 
                            [plan_id] + plan_ids + 
                            [goal_id] + goal_ids + 
                            [answer_id, verdict_id]
                        )

                        out_rec = {
                            "example_id": n_total,
                            "plan_len": rec["plan_length"],
                            "is_correct": is_correct,
                            "input_ids": input_ids,
                        }
                        if "n_obj" in rec:
                            out_rec["n_obj"] = rec["n_obj"]
                        fout.write(json.dumps(out_rec) + "\n")
                        data_split_dict[split].append(n_total)

                        n_total += 1
                        if is_correct:
                            n_correct += 1
                        else:
                            n_incorrect += 1
                        if n_total % args.progress_every == 0:
                            print(f"Processed {n_total} (correct={n_correct}, incorrect={n_incorrect})")
            
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


