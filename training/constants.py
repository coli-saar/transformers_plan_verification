from typing import List, Tuple
from transformers import PreTrainedTokenizer

PROJECT_ROOT: str = "/nethome/yudo/projects/plv_clean"
CONDOR_LOGS_DIR: str = "scratch_data/logs"

VERDICT_CORRECT_TOKEN: str = ' correct'
VERDICT_INCORRECT_TOKEN: str = ' incorrect'

# Tokenizer directories
GRIPPERS_TOKENIZER_DIR: str = "scratch_data/tokenizers/grippers"
LIGHTS_OUT_EXP_TOKENIZER_DIR: str = "scratch_data/created_tokenizers/lights_out_exp_anonym_loc"
LIGHTS_OUT_REGULAR_TOKENIZER_DIR: str = "scratch_data/created_tokenizers/lights_out_regular"
COLOR_TOKENIZER_DIR: str = "scratch_data/created_tokenizers/color_new_deanonymized"

# Data directories
GRIPPERS_DATA_BASE: str = "/scratch/a7_data/more_comparable_generation/grippers_heavy_datasets_Jan_26"
LIGHTS_OUT_EXP_DATA_DIR: str = "/scratch/a7_data/more_comparable_generation/lights_out_exponential_incomplete_anonym_action_names_lo"
LIGHTS_OUT_REGULAR_DATA_DIR: str = "/scratch/a7_data/more_comparable_generation/lights_out_orig_incomplete"
COLOR_DATA_BASE: str = "/scratch/a7_data/color_new_config_deanon"

HF_HOME: str = "/scratch/yudo/huggingface"

def get_verdict_tokens() -> List[str]:
    """Return the ordered list of verdict tokens."""
    return [VERDICT_CORRECT_TOKEN, VERDICT_INCORRECT_TOKEN]


def format_verdict(is_correct: bool) -> str:
    """Return the verdict token string for a boolean correctness flag."""
    return VERDICT_CORRECT_TOKEN if is_correct else VERDICT_INCORRECT_TOKEN


def add_verdict_special_tokens(tokenizer: PreTrainedTokenizer) -> int:
    """Register verdict tokens as additional special tokens on the tokenizer.

    Returns the number of tokens added (0 if already present).
    """
    tokens = get_verdict_tokens()
    return tokenizer.add_special_tokens({'additional_special_tokens': tokens})


def get_verdict_token_ids(tokenizer: PreTrainedTokenizer) -> Tuple[int, int]:
    """Get token ids for the verdict tokens using the provided tokenizer.

    Returns:
        (correct_token_id, incorrect_token_id)
    """
    correct_id = tokenizer.encode(VERDICT_CORRECT_TOKEN, add_special_tokens=False)[0]
    incorrect_id = tokenizer.encode(VERDICT_INCORRECT_TOKEN, add_special_tokens=False)[0]
    return correct_id, incorrect_id
