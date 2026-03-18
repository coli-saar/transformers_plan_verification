"""
Color Bags data generator.

Usage:
    python -m data_generation.color.generator \
        --output_file output.jsonl \
        --min_len 11 --max_len 100 --n_per_len 1000 \
        --incorrect_type incomplete --incorrect_position random
"""

import json
import math
import os
import random
from argparse import ArgumentParser
from typing import Dict, Iterator, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

MAX_TOKEN_ID = 12
VERDICT_CORRECT_TOKEN = " correct"
VERDICT_INCORRECT_TOKEN = " incorrect"


def build_vocab(max_token_id: int = MAX_TOKEN_ID) -> Dict[str, int]:
    """Build vocabulary with separate bag and color tokens."""
    vocab: Dict[str, int] = {}
    idx = 0

    for i in range(max_token_id):
        vocab[f"bag{i}"] = idx
        idx += 1

    for i in range(max_token_id):
        vocab[f"color{i}"] = idx
        idx += 1

    for tok in ["add_color", "remove_all_color", "has-color"]:
        vocab[tok] = idx
        idx += 1

    for tok in ["<pad>", "<bos>", "<eos>", "<unk>",
                "<init>", "<goal>", "<plan>", "<answer>",
                VERDICT_CORRECT_TOKEN, VERDICT_INCORRECT_TOKEN]:
        vocab[tok] = idx
        idx += 1

    return vocab


def build_tokenizer(max_token_id: int = MAX_TOKEN_ID):
    """Build a WordLevel tokenizer."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.normalizers import NFKC, Sequence as NormSequence
    from tokenizers.pre_tokenizers import Sequence as PreTokSequence, WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocab = build_vocab(max_token_id=max_token_id)
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.normalizer = NormSequence([NFKC()])
    tok.pre_tokenizer = PreTokSequence([WhitespaceSplit()])

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    hf_tok.add_special_tokens({
        "additional_special_tokens": [
            "<init>", "<goal>", "<plan>", "<answer>",
            VERDICT_CORRECT_TOKEN, VERDICT_INCORRECT_TOKEN,
        ]
    })
    return hf_tok


def get_or_create_tokenizer(tokenizer_dir: str, rebuild: bool = False, max_token_id: int = MAX_TOKEN_ID):
    """Load existing tokenizer or create new one."""
    from transformers import AutoTokenizer
    import shutil

    if rebuild and os.path.exists(tokenizer_dir):
        shutil.rmtree(tokenizer_dir)

    if os.path.exists(tokenizer_dir):
        return AutoTokenizer.from_pretrained(tokenizer_dir)

    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer = build_tokenizer(max_token_id=max_token_id)
    tokenizer.save_pretrained(tokenizer_dir)
    return tokenizer


def get_config(plan_length: int) -> Tuple[int, int]:
    """Get (n_bags, n_colors) for a plan length. Uses cells ≈ plan_length / 4."""
    target_cells = max(4, plan_length // 4)
    n = int(math.sqrt(target_cells)) + 1
    m = max(2, (target_cells + n - 1) // n)
    return n, m


class ColorBagsGenerator:
    """Generator for Color Bags domain plans."""

    def __init__(self, n_bags: int, n_colors: int, well_formed: bool, max_token_id: int = MAX_TOKEN_ID):
        self.bag_ids = sorted(random.sample(range(max_token_id), n_bags))
        self.color_ids = sorted(random.sample(range(max_token_id), n_colors))
        self.well_formed = well_formed
        self.state: Set[Tuple[int, int]] = set()

    def reset(self):
        self.state = set()

    def get_applicable_actions(self) -> List[Tuple[str, int, int]]:
        """Get all applicable actions in current state."""
        actions = []
        for b in self.bag_ids:
            for c in self.color_ids:
                if self.well_formed:
                    if (b, c) not in self.state:
                        actions.append(("add_color", b, c))
                    else:
                        actions.append(("remove_all_color", b, c))
                else:
                    actions.append(("add_color", b, c))
                    actions.append(("remove_all_color", b, c))
        return actions

    def apply_action(self, action: Tuple[str, int, int]):
        action_type, b, c = action
        if action_type == "add_color":
            self.state.add((b, c))
        else:
            self.state.discard((b, c))

    def _apply_to_state(self, state: Set[Tuple[int, int]], action: Tuple[str, int, int]):
        action_type, b, c = action
        if action_type == "add_color":
            state.add((b, c))
        else:
            state.discard((b, c))

    def _is_applicable_in_state(self, state: Set[Tuple[int, int]], action: Tuple[str, int, int]) -> bool:
        action_type, b, c = action
        has_color = (b, c) in state
        return not has_color if action_type == "add_color" else has_color

    def generate_correct_plan(self, plan_length: int, max_retries: int = 10):
        """Generate a correct plan via random walk."""
        for _ in range(max_retries):
            self.reset()
            plan = []

            for _ in range(plan_length):
                applicable = self.get_applicable_actions()
                action = random.choice(applicable)
                plan.append(action)
                self.apply_action(action)

            if not self.state:
                continue

            return plan, list(self.state)

        return None, None

    def generate_invalid_plan(self, correct_plan: List[Tuple[str, int, int]], goal, position: str = "last"):
        """Generate invalid plan by inserting precondition-violating action (WF only)."""
        if not self.well_formed:
            raise ValueError("Invalid plans only possible for well-formed domain")

        idx = len(correct_plan) - 1 if position == "last" else random.randint(0, len(correct_plan) - 1)

        self.reset()
        for action in correct_plan[:idx]:
            self.apply_action(action)

        applicable = set(self.get_applicable_actions())
        all_actions = (
            [("add_color", b, c) for b in self.bag_ids for c in self.color_ids] +
            [("remove_all_color", b, c) for b in self.bag_ids for c in self.color_ids]
        )
        non_applicable = [a for a in all_actions if a not in applicable]

        if not non_applicable:
            return None, None

        invalid_action = random.choice(non_applicable)
        return correct_plan[:idx] + [invalid_action] + correct_plan[idx + 1:], idx

    def generate_incomplete_plan(self, correct_plan: List[Tuple[str, int, int]], goal: List[Tuple[int, int]], position: str = "last"):
        """Generate incomplete plan by replacing a goal-achieving action."""
        goal_set = set(goal)

        critical_positions = [
            i for i, (t, b, c) in enumerate(correct_plan)
            if t == "add_color" and (b, c) in goal_set
        ]

        if not critical_positions:
            return None, None

        if position == "last":
            positions_to_try = list(reversed(critical_positions))
        else:
            positions_to_try = critical_positions.copy()
            random.shuffle(positions_to_try)

        for idx in positions_to_try:
            result = self._try_incomplete_at_position(correct_plan, goal_set, idx)
            if result is not None:
                return result, idx

        return None, None

    def _try_incomplete_at_position(self, correct_plan, goal_set, idx):
        _, orig_b, orig_c = correct_plan[idx]

        self.reset()
        for action in correct_plan[:idx]:
            self.apply_action(action)
        prefix_state = self.state.copy()

        applicable = self.get_applicable_actions()
        replacements = [
            a for a in applicable
            if not (a[0] == "add_color" and (a[1], a[2]) == (orig_b, orig_c))
        ]
        random.shuffle(replacements)

        for replacement in replacements:
            test_state = prefix_state.copy()
            self._apply_to_state(test_state, replacement)

            suffix_valid = True
            if self.well_formed:
                for action in correct_plan[idx + 1:]:
                    if not self._is_applicable_in_state(test_state, action):
                        suffix_valid = False
                        break
                    self._apply_to_state(test_state, action)
            else:
                for action in correct_plan[idx + 1:]:
                    self._apply_to_state(test_state, action)

            if suffix_valid and not goal_set.issubset(test_state):
                return correct_plan[:idx] + [replacement] + correct_plan[idx + 1:]

        return None

    def to_token_ids(self, plan, goal, is_correct: bool, vocab: Dict[str, int]) -> List[int]:
        """Convert plan and goal to token IDs."""
        ids = [vocab["<plan>"]]
        for action_type, b, c in plan:
            ids.extend([vocab[action_type], vocab[f"bag{b}"], vocab[f"color{c}"]])

        ids.append(vocab["<goal>"])
        for b, c in sorted(goal):
            ids.extend([vocab["has-color"], vocab[f"bag{b}"], vocab[f"color{c}"]])

        ids.append(vocab["<answer>"])
        verdict = VERDICT_CORRECT_TOKEN if is_correct else VERDICT_INCORRECT_TOKEN
        ids.append(vocab[verdict])

        return ids


def generate_pairs_for_length(
    plan_length: int,
    n_pairs: int,
    well_formed: bool,
    incorrect_type: str,
    incorrect_position: str,
    max_token_id: int = MAX_TOKEN_ID,
    max_attempts: Optional[int] = None,
    vocab: Optional[Dict[str, int]] = None,
) -> Iterator[Tuple[Dict, Dict]]:
    """Yield correct/incorrect example pairs for a plan length."""
    if not well_formed and incorrect_type == "invalid":
        raise ValueError("incorrect_type='invalid' only works with well_formed=True")

    if max_attempts is None:
        max_attempts = max(50, n_pairs * 50)

    if vocab is None:
        vocab = build_vocab(max_token_id=max_token_id)

    n_bags, n_colors = get_config(plan_length)

    attempts = 0
    pairs_generated = 0

    while pairs_generated < n_pairs and attempts < max_attempts:
        attempts += 1

        gen = ColorBagsGenerator(n_bags, n_colors, well_formed, max_token_id)

        plan, goal = gen.generate_correct_plan(plan_length)
        if plan is None:
            continue

        if incorrect_type == "invalid":
            inc_plan, inc_idx = gen.generate_invalid_plan(plan, goal, position=incorrect_position)
        else:
            inc_plan, inc_idx = gen.generate_incomplete_plan(plan, goal, position=incorrect_position)

        if inc_plan is None:
            continue

        correct_example = {
            "input_ids": gen.to_token_ids(plan, goal, True, vocab),
            "is_correct": True,
            "plan_len": len(plan),
            "incorrect_action_index": None,
        }

        incorrect_example = {
            "input_ids": gen.to_token_ids(inc_plan, goal, False, vocab),
            "is_correct": False,
            "plan_len": len(inc_plan),
            "incorrect_action_index": inc_idx,
        }

        pairs_generated += 1
        yield correct_example, incorrect_example

    if pairs_generated < n_pairs:
        raise RuntimeError(
            f"Could only generate {pairs_generated}/{n_pairs} pairs for plan length "
            f"{plan_length} after {attempts} attempts."
        )


def main():
    parser = ArgumentParser(description="Generate Color Bags training data")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--min_len", type=int, required=True)
    parser.add_argument("--max_len", type=int, required=True)
    parser.add_argument("--n_per_len", type=int, required=True)
    parser.add_argument("--incorrect_type", choices=["invalid", "incomplete"], default="incomplete")
    parser.add_argument("--incorrect_position", choices=["last", "random"], default="random")
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--handle_existing", choices=["overwrite", "append"], default=None)
    parser.add_argument("--well_formed", dest="well_formed", action="store_true", default=True)
    parser.add_argument("--not_well_formed", dest="well_formed", action="store_false")
    args = parser.parse_args()

    if args.min_len > args.max_len:
        raise ValueError("min_len must be <= max_len")
    if not args.well_formed and args.incorrect_type == "invalid":
        raise ValueError("incorrect_type='invalid' only works with well_formed=True")

    if args.tokenizer_dir:
        get_or_create_tokenizer(args.tokenizer_dir, rebuild=args.rebuild)

    if os.path.exists(args.output_file):
        if args.handle_existing is None:
            raise FileExistsError(f"Output file exists. Set --handle_existing overwrite|append.")
        elif args.handle_existing == "overwrite":
            os.remove(args.output_file)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_pairs = (args.max_len - args.min_len + 1) * args.n_per_len
    print(f"Generating {total_pairs} pairs ({total_pairs * 2} examples)...")
    print(f"  plan lengths: {args.min_len}..{args.max_len}")
    print(f"  well_formed: {args.well_formed}")
    print(f"  incorrect_type: {args.incorrect_type}, position: {args.incorrect_position}")

    if tqdm is None:
        raise ImportError("tqdm required")

    vocab = build_vocab()

    with open(args.output_file, "a") as f:
        for plan_length in range(args.min_len, args.max_len + 1):
            n_bags, n_colors = get_config(plan_length)
            bar = tqdm(total=args.n_per_len, desc=f"len={plan_length} ({n_bags}x{n_colors})")
            for correct_ex, incorrect_ex in generate_pairs_for_length(
                plan_length=plan_length,
                n_pairs=args.n_per_len,
                well_formed=args.well_formed,
                incorrect_type=args.incorrect_type,
                incorrect_position=args.incorrect_position,
                vocab=vocab,
            ):
                for example in (correct_ex, incorrect_ex):
                    f.write(json.dumps(example) + "\n")
                bar.update(1)
            bar.close()

    print(f"Wrote to {args.output_file}")


if __name__ == "__main__":
    main()
