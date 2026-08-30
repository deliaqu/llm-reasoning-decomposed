import matplotlib.pyplot as plt
import torch
from config import LOGIT_ATTR_RESULT_DIR, LINEAR_PROBE_RESULT_DIR
import seaborn as sns
import numpy as np
import os
from config import ACTIVATION_PATCHING_RESULT_DIR

#################################### for activation patching ####################################
def plot_cross_patched_logprobs(
    patched_target_logprobs_list, 
    tokenizer, 
    filename,
    patched_predictions=None, 
    num_layers=32,
    title="Avg Log Probabilities Across Layers with Patched Predictions",
    verbose=False,
):
    """
    Plots average log probabilities for clean and corrupted targets across layers.
    Optionally annotates the x-axis with decoded patched predictions.

    Args:
        patched_target_logprobs_list (list): List of runs, each of shape [num_layers][1][1][3] logprobs.
        tokenizer: HuggingFace tokenizer with .decode method.
        patched_predictions (list, optional): List of shape [num_layers][1] with token IDs.
        num_layers (int): Total number of layers (default 32).
        title (str): Title for the plot.
    """
    # Truncate final layers if necessary (e.g., model outputs + final layernorm)
    trimmed = [run[:-2] for run in patched_target_logprobs_list]
    num_effective_layers = len(trimmed[0])
    
    stacked = np.array(trimmed)  # shape: [num_runs, num_layers, 1, 1, 2]
    valid_mask = ~np.isnan(stacked).any(axis=(1, 2, 3, 4))  # shape: [num_runs]
    filtered = stacked[valid_mask]  # shape: [n_valid_runs, num_layers, 1, 1, 2]

    if len(filtered) == 0:
        raise ValueError("All runs contain NaNs; cannot compute mean.")

    avg_logprobs = filtered.squeeze(axis=(2, 3))  # shape: [n_valid_runs, num_layers, 3]
    mean_logprobs = avg_logprobs.mean(axis=0)    # shape: [num_layers, 3]

    logprob_clean = mean_logprobs[:, 0]
    logprob_corrupted = mean_logprobs[:, 1]
    logprob_target = mean_logprobs[:, 2]
    layers = np.arange(num_effective_layers)

    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    if "numerical" in filename:
        ax.plot(layers, logprob_clean, label='Clean Ans', marker='o')
        ax.plot(layers, logprob_corrupted, label='Corrupted Ans', marker='s')
        ax.plot(layers, logprob_target, label='Target Ans (clean abstraction+\ncorrutped operands)', marker='s')
    else: # symbolic
        ax.plot(layers, logprob_target, label='Target Ans (clean abstraction+\ncorrutped operands)', marker='o')
        ax.plot(layers, logprob_corrupted, label='Corrupted Ans', marker='s')

    plt.xlabel('Patched Layer Index', fontsize=18)
    plt.ylabel('Final Layer Log Probability', fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_xticks(layers)  # tick marks at every layer
    tick_step = 5
    labels = [str(i) if i % tick_step == 0 else "" for i in layers]
    ax.set_xticklabels(labels)

    ax.tick_params(axis="both", labelsize=18)

    if patched_predictions is not None:
        y_min, y_max = ax.get_ylim()
        offset = 0.15 * (y_max - y_min)
        for i, layer in enumerate(layers):
            token = tokenizer.decode(patched_predictions[i][0]) if i < len(patched_predictions) else "?"
            ax.text(layer, y_min - offset, token, ha='center', va='top', rotation=45, fontsize=18)
    
    plt.tight_layout()
    if verbose:
        print("Mean logprobs shape:", mean_logprobs.shape)
        print("Any NaNs in clean?", np.isnan(logprob_clean).any())
        print("Any NaNs in corrupted?", np.isnan(logprob_corrupted).any())
        print("Logprob_clean:", logprob_clean)
        print("Logprob_corrupted:", logprob_corrupted)

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

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

#     if np.isnan(effect).any():
#         print("NaN detected")
#         return None
    
    return numerator / denominator


def make_patching_heatmap(data, component, filename=None):
    data = data[:-2]  # Remove FLN and embed_out
    num_layers, width = data.shape[0], data.shape[1]
    data_plot = np.flipud(data.squeeze())

    plt.figure(figsize=(width/3, num_layers/3) if width > 1 else (2, num_layers/3))
    sns.heatmap(data_plot if width > 1 else data_plot.reshape(-1, 1),
                annot=width <= 10,
                cmap="viridis",
                cbar=True,
                yticklabels=np.arange(num_layers)[::-1])
    plt.title(f"Patching Effect ({component})")
    plt.xlabel("Position" if width > 1 else "Value")
    plt.ylabel("Layer (0 at bottom)")
    plt.tight_layout()
    #plt.show()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()
    
def compute_and_plot_effects(results, patching_type, problem_type, patching_scope, model_name, report_logprob_diff=[True], out_dir=ACTIVATION_PATCHING_RESULT_DIR):
    for diff in report_logprob_diff:
        all_effects = []
        for r in results:
            effect = compute_patching_effect(
                r["patched_target_logprobs"],
                r["corrupted_target_logprobs"],
                r["clean_target_logprobs"],
                diff=diff
            )
            if effect is not None:
                all_effects.append(effect)
        if not all_effects:
            continue
        avg_effect = np.mean(np.stack(all_effects), axis=0)
        suffix = "diff" if diff else "logprob"
        os.makedirs(out_dir, exist_ok=True)
        save_path = f"{out_dir}/{model_name}_{patching_type}_{problem_type}_{patching_scope}_{suffix}.png"
        make_patching_heatmap(avg_effect, component=patching_scope, filename=save_path)

def plot_patching_effects_three_lines(
    effects,
    save_path=None,
    title=None,
    tick_every=5,
    figsize=(6, 3.5),   # smaller
    label_font=12,
    tick_font=10,
    legend_font=11,
    lw=2,
):
    color_map = {
        "layer": "#b22222",
        "attn":  "#6a0dad",
        "mlp":   "#808000",
    }

    num_layers = None
    
    for k in ["layer", "attn", "mlp"]:
        if k in effects and effects[k] is not None:
            num_layers = len(np.asarray(effects[k]).reshape(-1))
            break
    if num_layers is None:
        raise ValueError("effects is empty or contains no plottable arrays.")

    x = np.arange(num_layers)

    plt.figure(figsize=figsize)

    for key in ["layer", "attn", "mlp"]:
        if key not in effects:
            continue
        y = np.asarray(effects[key], dtype=float).reshape(-1)
        plt.plot(x, y, label=key, linewidth=lw, color=color_map.get(key, None))

    plt.xlabel("Layer", fontsize=label_font)
    plt.ylabel("Patching Effect", fontsize=label_font)
    if title is not None:
        plt.title(title, fontsize=label_font)

    plt.xticks(x, [str(i) if (i % tick_every == 0) else "" for i in x], fontsize=tick_font)
    plt.yticks(fontsize=tick_font)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(loc="upper left", fontsize=legend_font, frameon=True)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    

def plot_patching_effects_at_once(stacked_results, patching_type, problem_type, model_name, out_dir=ACTIVATION_PATCHING_RESULT_DIR):
    """
        Input: stacked_results = {
                                "layer": layer_results, 
                                "attn": attn_results, 
                                "mlp": mlp_results
                                }
    """
    effects = {} # to store mean effect per patching type by hs
    for (key, hidden_state_result) in stacked_results.items():
        hs_effects = []
        for r in hidden_state_result:
            effect = compute_patching_effect(
                r["patched_target_logprobs"],
                r["corrupted_target_logprobs"],
                r["clean_target_logprobs"],
                diff=True
            )
            if effect is not None:
                hs_effects.append(effect)
        
        if not hs_effects:
            continue
        effects[key] = np.mean(np.stack(hs_effects), axis=0)[:-2]
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/{model_name}_{patching_type}_{problem_type}_all_diff.png"
    plot_patching_effects_three_lines(
        effects,
        save_path=save_path,
    )   
#################################### for logit attribution ####################################
def plot_avg_logits_by_path(role_token_logits, L, problem_type, model_name):
    mapping = {
        "addition": ["c+", 'op (+)'],
        "subtraction": ["c-",'op (-)'],
        "multiplication": ["c*",'op (*)'],
        "division": ["c÷",'op (÷)'],
    }
    paths = ["attn_out", "mlp_out", "resid_final"]
    all_roles = [ "a", "b", "c+","c-","c*","c÷", 'op (+)', 'op (-)', 'op (*)', 'op (÷)']
    #roles = [ 'op (+)', 'op (-)', 'op (*)', 'op (÷)']
    roles = [ "a", "b"] +  mapping[problem_type]
    
    
    for path in paths:
        # Collect all logit values for this path across roles and layers
        all_vals = []
        for role in all_roles:
            L = 0
            layerwise = role_token_logits[(path, role)]
            for layer in layerwise:
                L += 1
                all_vals.extend(layer)
        path_min = min(all_vals)
        path_max = max(all_vals)

        # Now plot with y-limits fixed for this path
        plt.figure(figsize=(8,6))
        for role in roles:
            layerwise = role_token_logits[(path, role)]
            avg = [sum(layer) / len(layer) if layer else 0.0 for layer in layerwise]
            std = [torch.std(torch.tensor(layer)).item() if len(layer) > 1 else 0.0 for layer in layerwise]
            x = list(range(L))
            avg = torch.tensor(avg)
            std = torch.tensor(std)

            plt.plot(x, avg, label=role)
            plt.fill_between(x, avg - std, avg + std, alpha=0.2)

        plt.xticks(ticks=range(L), labels=[f"{i}" for i in range(L)], fontsize=8)
        plt.ylabel("Logit")
        plt.xlabel("Layer")
        plt.title(f"Average Logits ({path}) for Operator (op), Operands (a,b) and Operators (c)")
        plt.ylim(path_min, path_max)  # Per-path fixed scale
        plt.legend(title="Role")
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout(pad=1.5)
        plt.savefig(f'{LOGIT_ATTR_RESULT_DIR}/{problem_type}_{path}_avg_{model_name}.png')
        plt.show()
        
def plot_avg_logit_diff_by_path(logits_set_1, logits_set_2, L, problem_type, model_name):
    paths = ["resid_mid"]
    all_roles = ["a", "b", "c+", "c-", "c*", "c÷", 'op (+)', 'op (-)', 'op (*)', 'op (÷)']
    
    if problem_type == "addition-subtraction":
        roles = ['op (+)', 'op (-)']
    else:
        roles = ['op (*)', 'op (÷)']
    
    for path in paths:
        # Collect all diff values for this path to set global min/max
        all_diffs = []
        for role in all_roles:
            vals1 = logits_set_1[(path, role)]
            vals2 = logits_set_2[(path, role)]
            for layer_vals1, layer_vals2 in zip(vals1, vals2):
                avg1 = sum(layer_vals1) / len(layer_vals1) if layer_vals1 else 0.0
                avg2 = sum(layer_vals2) / len(layer_vals2) if layer_vals2 else 0.0
                all_diffs.append(avg1 - avg2)
        path_min = min(all_diffs)
        path_max = max(all_diffs)

        # Now plot differences for selected roles
        plt.figure(figsize=(8,6))
        for role in roles:
            vals1 = logits_set_1[(path, role)]
            vals2 = logits_set_2[(path, role)]

            avg1 = [sum(layer) / len(layer) if layer else 0.0 for layer in vals1]
            avg2 = [sum(layer) / len(layer) if layer else 0.0 for layer in vals2]
            diff = [a1 - a2 for a1, a2 in zip(avg1, avg2)]

            std1 = [torch.std(torch.tensor(layer)).item() if len(layer) > 1 else 0.0 for layer in vals1]
            std2 = [torch.std(torch.tensor(layer)).item() if len(layer) > 1 else 0.0 for layer in vals2]
            #std_diff = [((s1 ** 2 + s2 ** 2) ** 0.5) for s1, s2 in zip(std1, std2)]  # std of difference
            std_diff = [torch.std(torch.tensor([v1 - v2 for v1, v2 in zip(layer1, layer2)])).item()
            if len(layer1) > 1 and len(layer2) > 1 else 0.0
            for layer1, layer2 in zip(vals1, vals2)]

            x = list(range(L))
            diff = torch.tensor(diff)
            std_diff = torch.tensor(std_diff)

            plt.plot(x, diff, label=role)
            plt.fill_between(x, diff - std_diff, diff + std_diff, alpha=0.2)

        plt.xticks(ticks=range(L), labels=[f"{i}" for i in range(L)], fontsize=8)
        plt.ylabel("Logit Difference")
        plt.xlabel("Layer")
        plt.title(f"Difference in Average Logits ({path}) for Operators")
        plt.ylim(path_min, path_max)
        plt.legend(title="Role")
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout(pad=1.5)
        plt.savefig(f'{LOGIT_ATTR_RESULT_DIR}/{problem_type}_{path}_diff_{model_name}.png')
        plt.show()

#################################### for linear probe ####################################
def plot_linear_probe_accuracy(results, problem_type_label, model_name, distractor=False):
    L = len(list(results.values())[0])
    plt.figure(figsize=(8,6))
    
    for name_repr, accuracies in results.items():
        x = list(range(L))
        plt.plot(x, accuracies, label=name_repr, linewidth=2)

    tick_spacing = max(1, L // 16)
    plt.xticks(ticks=range(0, L, tick_spacing), labels=[f"{i}" for i in range(0, L, tick_spacing)], fontsize=10)
    plt.ylabel("Accuracy")
    plt.xlabel("Layer")
    plt.title(f"Linear Probe Accuracy ({problem_type_label})")
    plt.ylim(0, 1.05)
    plt.legend(title="Representation Type")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout(pad=1.5)
    
    os.makedirs(LINEAR_PROBE_RESULT_DIR, exist_ok=True)
    if distractor:
        plt.savefig(f'{LINEAR_PROBE_RESULT_DIR}/linear_probe_acc_{problem_type_label}_distractor_{model_name}.png')
    else:
        plt.savefig(f'{LINEAR_PROBE_RESULT_DIR}/linear_probe_acc_{problem_type_label}_{model_name}.png')
    plt.close()
