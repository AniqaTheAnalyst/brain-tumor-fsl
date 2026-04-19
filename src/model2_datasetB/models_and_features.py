# 3_evaluation_and_run.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from IPython.display import display, Image as IPImage
import os
import matplotlib
matplotlib.use('Agg')

# Import from previous files
from 1_data_and_setup import (
    DEVICE, N_WAY, CLASS_NAMES, SHOTS, EPISODES_EVAL, 
    CI_ALPHA, BOOTSTRAP_N, support_pool, query_pool, create_episode
)
from 2_models_and_features import model_configs, support_feats, query_feats

# ========================= HELPER =========================
def save_and_display(fig, filename, dpi=200):
    filepath = f"/kaggle/working/{filename}"
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    display(IPImage(filepath))
    print(f"  ✓ Saved and displayed: {filepath}")
    return filepath

# ========================= CONFIDENCE INTERVAL =========================
def bootstrap_ci(acc_list, n=BOOTSTRAP_N, alpha=CI_ALPHA):
    arr = np.array(acc_list)
    samples = np.random.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    lo = np.percentile(samples, 100 * alpha / 2)
    hi = np.percentile(samples, 100 * (1 - alpha / 2))
    return lo, hi

# ========================= PLOTTING FUNCTIONS =========================
def plot_confusion_matrix(cm, name, k_shot, mean_acc, std_acc, ci_lo, ci_hi):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor='gray', ax=ax)
    ax.set_title(f'{name} – {k_shot}-shot\nAcc: {mean_acc:.3f} ± {std_acc:.3f}   |   95% CI [{ci_lo:.3f}, {ci_hi:.3f}]',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.tick_params(axis='x', rotation=30)
    save_and_display(fig, f"cm_{name}_{k_shot}shot_pub.png")

def plot_summary_chart(results, shots, model_names):
    n_shots = len(shots)
    n_models = len(model_names)
    x = np.arange(n_models)
    width = 0.25
    fig, axes = plt.subplots(1, n_shots, figsize=(6 * n_shots, 5), sharey=True)
    if n_shots == 1:
        axes = [axes]
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for ax, k in zip(axes, shots):
        accs = [results[f"{m}_{k}shot"]['mean_acc'] for m in model_names]
        f1s  = [results[f"{m}_{k}shot"]['macro_f1'] for m in model_names]
        cis  = [(results[f"{m}_{k}shot"]['mean_acc'] - results[f"{m}_{k}shot"]['ci_lo'],
                 results[f"{m}_{k}shot"]['ci_hi'] - results[f"{m}_{k}shot"]['mean_acc'])
                for m in model_names]
        ci_arr = np.array(cis).T

        ax.bar(x - width/2, accs, width, label='Accuracy', color=colors, alpha=0.85,
               yerr=ci_arr, capsize=5, error_kw={'elinewidth':1.5})
        ax.bar(x + width/2, f1s, width, label='Macro-F1', color=colors, alpha=0.45, hatch='//')

        for bar in ax.patches[:len(accs)]:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_title(f'{k}-Shot', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle('Few-Shot Brain Tumor Classification\n(Pure Prototypical | Leak-Free | 95% CI)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_and_display(fig, "summary_accuracy_f1.png", dpi=200)

def plot_perclass_heatmap(results, shots, model_names):
    row_labels = [f"{m}\n({k}-shot)" for k in shots for m in model_names]
    data = np.array([results[f"{m}_{k}shot"]['per_class_f1'] for k in shots for m in model_names])

    fig, ax = plt.subplots(figsize=(8, max(5, len(row_labels) * 0.55 + 1.5)))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=CLASS_NAMES, yticklabels=row_labels,
                vmin=0, vmax=1, linewidths=0.4, linecolor='white', ax=ax)
    ax.set_title('Per-Class F1 Score (all models × shots)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Model / Shot')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    save_and_display(fig, "perclass_f1_heatmap.png", dpi=200)

def plot_shot_progression(results, shots, model_names):
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', '^']
    colors = ['#1565C0', '#2E7D32', '#E65100']

    for i, name in enumerate(model_names):
        accs = [results[f"{name}_{k}shot"]['mean_acc'] for k in shots]
        ci_lo = [results[f"{name}_{k}shot"]['ci_lo'] for k in shots]
        ci_hi = [results[f"{name}_{k}shot"]['ci_hi'] for k in shots]
        ax.plot(shots, accs, marker=markers[i], color=colors[i], linewidth=2,
                markersize=8, label=name)
        ax.fill_between(shots, ci_lo, ci_hi, color=colors[i], alpha=0.12)

    ax.set_xlabel('Number of Shots (k)')
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Accuracy vs. Number of Shots (shaded = 95% CI)', fontsize=13, fontweight='bold')
    ax.set_xticks(shots)
    ax.set_xticklabels([f'{k}-shot' for k in shots])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    save_and_display(fig, "shot_progression.png", dpi=200)

# ========================= RUN FEW-SHOT =========================
def run_fewshot(k_shot, name):
    print(f"\n{'='*90}")
    print(f"  {k_shot}-SHOT | {name}  (Pure Prototypical | Leak-Free)")
    print(f"{'='*90}")

    s_feats = support_feats[name]
    q_feats = query_feats[name]
    all_true, all_pred = [], []
    accs = []

    for ep in tqdm(range(EPISODES_EVAL), desc=f"{k_shot}-shot {name}"):
        supp_paths, qry_paths, sy, qy = create_episode(
            k_shot, support_pool, query_pool, query_per_class=15
        )
        sy, qy = sy.to(DEVICE), qy.to(DEVICE)

        sf = torch.stack([s_feats[p] for p in supp_paths]).to(DEVICE)
        qf = torch.stack([q_feats[p] for p in qry_paths]).to(DEVICE)

        sf = F.normalize(sf, p=2, dim=-1)
        qf = F.normalize(qf, p=2, dim=-1)

        prototypes = torch.stack([sf[sy == i].mean(dim=0) for i in range(N_WAY)])
        dists = 1 - torch.mm(qf, prototypes.t())
        preds = dists.argmin(dim=1)

        ep_true = qy.cpu().numpy()
        ep_pred = preds.cpu().numpy()

        all_true.extend(ep_true)
        all_pred.extend(ep_pred)
        accs.append(accuracy_score(ep_true, ep_pred))

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    ci_lo, ci_hi = bootstrap_ci(accs)

    precision, recall, f1, sup = precision_recall_fscore_support(
        all_true, all_pred, average=None, zero_division=0, labels=list(range(N_WAY))
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average='macro', zero_division=0
    )
    cm = confusion_matrix(all_true, all_pred)

    print(f"\n  Accuracy  : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  95% CI    : [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Macro F1  : {macro_f1:.4f}")

    print(f"\n  Per-class metrics:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"    {cls:12} | P: {precision[i]:.3f} | R: {recall[i]:.3f} | F1: {f1[i]:.3f} | N: {sup[i]}")

    plot_confusion_matrix(cm, name, k_shot, mean_acc, std_acc, ci_lo, ci_hi)

    return {
        'mean_acc': mean_acc, 'std_acc': std_acc,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'macro_f1': macro_f1,
        'per_class_f1': f1,
    }

# ========================= MAIN EXECUTION =========================
if __name__ == "__main__":
    SAVE_DIR = "/kaggle/working/visualization_figure"
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n{'#'*90}")
    print(" PUBLICATION-READY FEW-SHOT EXPERIMENTS (Leak-Free | 600 Episodes | 95% CI)")
    print(f"{'#'*90}\n")

    results = {}
    for k in SHOTS:
        for name in model_configs.keys():
            key = f"{name}_{k}shot"
            results[key] = run_fewshot(k_shot=k, name=name)

    print(f"\n{'#'*90}")
    print(" STATISTICAL SIGNIFICANCE (McNemar's Test)")
    print(f"{'#'*90}")
    # Add your McNemar code here if needed

    print(f"\n{'='*90}")
    print(" FINAL RESULTS SUMMARY")
    print(f"{'='*90}")

    # Generate final visualizations
    print("\nGenerating summary visualizations...")
    plot_summary_chart(results, SHOTS, list(model_configs.keys()))
    plot_perclass_heatmap(results, SHOTS, list(model_configs.keys()))
    plot_shot_progression(results, SHOTS, list(model_configs.keys()))

    print(f"\n✅ All experiments and visualizations complete.")
    print("All plots are saved and displayed inline in Kaggle.")