import numpy as np
import torch

def compute_patching_effect(patched_target_logprobs, corrupted_target_logprobs, clean_target_logprobs, diff=True):
    patched_target_logprobs = np.array(patched_target_logprobs, dtype=float)
    patched_clean_ans_logprob = patched_target_logprobs[:, :, :, 0]
    patched_corrupted_ans_logprob = patched_target_logprobs[:, :, :, 1]
    if not (np.isfinite(patched_clean_ans_logprob).all() and np.isfinite(patched_corrupted_ans_logprob).all()):
        print("Non-finite value detected in patched logprobs")
        return None
    corrupted_clean_ans_logprob = corrupted_target_logprobs[0]
    corrupted_corrupted_ans_logprob = corrupted_target_logprobs[1]
    clean_clean_ans_logprob = clean_target_logprobs[0]
    clean_corrupted_ans_logprob = clean_target_logprobs[1]
    if diff:
        numerator = (patched_clean_ans_logprob - patched_corrupted_ans_logprob) - \
                    (corrupted_clean_ans_logprob - corrupted_corrupted_ans_logprob)
        denominator = (clean_clean_ans_logprob - clean_corrupted_ans_logprob) - \
                      (corrupted_clean_ans_logprob - corrupted_corrupted_ans_logprob)
    else:
        numerator = patched_clean_ans_logprob - corrupted_clean_ans_logprob
        denominator = clean_clean_ans_logprob - corrupted_clean_ans_logprob
    effect = np.array(numerator / denominator, dtype=float)
    if not np.all(np.isfinite(effect)):
        print("Non-finite value (NaN or Inf) detected")
        return None
    return numerator / denominator
