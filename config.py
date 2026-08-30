import os as _os
HOME_DIR = _os.path.dirname(_os.path.abspath(__file__))
CACHE_DIR = _os.environ.get("CACHE_DIR", f"{HOME_DIR}/cache")
DATA_DIR = f"{HOME_DIR}/data/interpretability_data"

from instructions import original_direct_instruction as DIRECT_INSTRUCTION
from instructions import original_cot_instruction    as COT_INSTRUCTION

# Legacy alias — some older scripts use INSTRUCTION; kept for backward compatibility.
INSTRUCTION = DIRECT_INSTRUCTION
ANSWER_PREFIX = "#### "
LOGIT_ATTR_RESULT_DIR = f"{HOME_DIR}/results/logit_attribution"
ACTIVATION_PATCHING_RESULT_DIR = f"{HOME_DIR}/results/activation_patching"
NOOP_PATCHING_RESULT_DIR = f"{HOME_DIR}/results/noop_patching"
P1_PADDED_LT2_PATCHING_RESULT_DIR = f"{HOME_DIR}/results/p1_padded_lt2_patching"
P1_PADDED_DELTA0_PATCHING_RESULT_DIR = f"{HOME_DIR}/results/p1_padded_delta0_patching"
GSM_SYM_WITHIN_TEMPLATE_PATCHING_RESULT_DIR = f"{HOME_DIR}/results/gsm_sym_within_template_patching"
LOGIT_LENS_RESULT_DIR = f"{HOME_DIR}/results/logit_lens"
LINEAR_PROBE_RESULT_DIR = f"{HOME_DIR}/results/linear_probe"

PROBLEM_TYPES = [
        "addition", "subtraction", 
        "multiplication", "division"
]
OPERATOR_PAIRS = [
    ("addition", "subtraction", "addition-subtraction"),
    ("multiplication", "division", "multiplication-division"),
]
CROSS_PATCHING_CLEAN_CORR_MAP={
    "addition": [
        'paired_subtraction', 'subtraction',
        'multiplication', 'division'
        ],
    "subtraction": [
        'paired_addition', 'addition',
        'multiplication', 'division'
        ],
    "multiplication": [
        'paired_division', 'division',
        'addition', 'subtraction'
        ],
    "division": [
        'paired_multiplication', 'multiplication',
        'addition', 'subtraction'
        ],
}

SYSTEM_PROMPT = "You are a chatbot who is capable of performing the arithmetic problems."
MODEL_NAME_MAP = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen-7b-instruct",
    "Qwen/Qwen2.5-14B-Instruct": "qwen-14b-instruct",
    "Qwen/Qwen2.5-72B-Instruct": "qwen-72b-instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct": "qwen-math-7b-instruct",
    "google/gemma-2-9b-it": "gemma-2-9b-it",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama-8b-instruct",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b-instruct",
}
