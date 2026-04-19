# 3_evaluation_and_run.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from IPython.display import Image, display
import os

# Import from other files
from 1_data_and_setup import *      # support_pool, query_pool, create_episode, etc.
from 2_models_and_features import * # model_configs, support_feats, query_feats

def save_and_display(fig, filename, dpi=200):
    filepath = f"/kaggle/working/{filename}"
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    display(Image(filename=filepath, width=800))
    print(f" ✓ Saved: {filepath}")
    return filepath

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
    ax.set_title(f'{name} – {k_shot}-shot\nAcc: {mean_acc:.3f} ± {std_acc:.3f} | 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.tick_params(axis='x', rotation=30)
    save_and_display(fig, f"cm_{name}_{k_shot}shot_pub.png")

def plot_summary_chart(results, shots, model_names):
    # (paste your original plot_summary_chart function here)
    pass   # ← Replace with your full function from original code

def plot_perclass_heatmap(results, shots, model_names):
    # (paste your original function)
    pass

def plot_shot_progression(results, shots, model_names):
    # (paste your original function)
    pass

# ========================= MAIN RUN FUNCTION =========================
def run_fewshot(k_shot, name):
    print(f"\n{'='*90}")
    print(f" {k_shot}-SHOT | {name} (Pure Prototypical | Leak-Free)")
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

    print(f"\n Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f" 95% CI : [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f" Macro F1 : {macro_f1:.4f}")

    print(f"\n Per-class metrics:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f" {cls:12} | P: {precision[i]:.3f} | R: {recall[i]:.3f} | F1: {f1[i]:.3f} | N: {sup[i]}")

    plot_confusion_matrix(cm, name, k_shot, mean_acc, std_acc, ci_lo, ci_hi)

    return {
        'mean_acc': mean_acc, 'std_acc': std_acc,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'macro_f1': macro_f1,
        'per_class_f1': f1,
    }

# ========================= MAIN EXECUTION =========================
if __name__ == "__main__":
    print(f"📁 Saving all outputs to: {SAVE_DIR}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    shots = [1, 5, 10]
    results = {}

    for k in shots:
        for name in model_configs.keys():
            key = f"{name}_{k}shot"
            results[key] = run_fewshot(k, name)

    print("\n🎉 All experiments completed!")
    print(" STATISTICAL SIGNIFICANCE (McNemar's Test) - To be implemented")
    
    # You can call plot_summary_chart(results, shots, list(model_configs.keys())) here