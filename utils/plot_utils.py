import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os

def plot_cross_patched_logprobs(
    patched_target_logprobs_list, 
    tokenizer, 
    filename,
    patched_predictions=None, 
    num_layers=32,
    title="Avg Log Probabilities Across Layers with Patched Predictions",
    verbose=False,
):
    trimmed = [run[:-2] for run in patched_target_logprobs_list]
    num_effective_layers = len(trimmed[0])
    stacked = np.array(trimmed)
    valid_mask = ~np.isnan(stacked).any(axis=(1, 2, 3, 4))
    filtered = stacked[valid_mask]
    if len(filtered) == 0:
        raise ValueError("All runs contain NaNs; cannot compute mean.")
    avg_logprobs = filtered.squeeze(axis=(2, 3))
    mean_logprobs = avg_logprobs.mean(axis=0)
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
    else:
        ax.plot(layers, logprob_target, label='Target Ans (clean abstraction+\ncorrutped operands)', marker='o')
        ax.plot(layers, logprob_corrupted, label='Corrupted Ans', marker='s')
    plt.xlabel('Patched Layer Index', fontsize=18)
    plt.ylabel('Final Layer Log Probability', fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(layers)
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
