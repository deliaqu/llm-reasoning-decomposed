import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_linear_probe_accuracy(results, problem_type_label, model_name, distractor=False, out_dir=None):
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
    if out_dir is not None:
        import os
        os.makedirs(out_dir, exist_ok=True)
        if distractor:
            plt.savefig(f'{out_dir}/linear_probe_acc_{problem_type_label}_distractor_{model_name}.png')
        else:
            plt.savefig(f'{out_dir}/linear_probe_acc_{problem_type_label}_{model_name}.png')
    else:
        plt.show()
    plt.close()
