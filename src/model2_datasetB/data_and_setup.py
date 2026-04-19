# 1_data_and_setup.py
import torch
import random
import numpy as np
import os
import hashlib
from PIL import Image

# ========================= CONFIG =========================
IMAGE_SIZE = 224
N_WAY = 4
SHOTS = [1, 5, 10]
EPISODES_EVAL = 600

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

CI_ALPHA = 0.05
BOOTSTRAP_N = 1000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

print(f"\nFew-Shot Setup | Device: {DEVICE}")
print(f"Episodes: {EPISODES_EVAL} | CI: {int((1-CI_ALPHA)*100)}%")

# ========================= DATASET =========================
DATASET_PATH = "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, "Training")
TEST_DIR = os.path.join(DATASET_PATH, "Testing")

print("\n📦 Using SINGLE dataset only (Dataset 1)")
print("Train:", TRAIN_DIR)
print("Test :", TEST_DIR)

# ========================= LOAD DATA =========================


def load_classwise_split(root):
    classwise = {i: [] for i in range(N_WAY)}
    for i, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(root, cls)
        if os.path.exists(cls_dir):
            classwise[i] = [
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
            ]
    return classwise


support_pool = load_classwise_split(TRAIN_DIR)
query_pool = load_classwise_split(TEST_DIR)

print("\nDataset Distribution:")
print("Support:", [len(support_pool[i]) for i in range(N_WAY)])
print("Query  :", [len(query_pool[i]) for i in range(N_WAY)])

# ========================= DUPLICATE REMOVAL =========================


def hash_image(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def remove_duplicates(pool):
    cleaned = {}
    removed = 0
    for cls, paths in pool.items():
        seen = set()
        new_list = []
        for p in paths:
            h = hash_image(p)
            if h not in seen:
                seen.add(h)
                new_list.append(p)
            else:
                removed += 1
        cleaned[cls] = new_list
    return cleaned, removed


print("\n🔧 Removing duplicates...")
support_pool, r1 = remove_duplicates(support_pool)
query_pool, r2 = remove_duplicates(query_pool)

print(f"Removed TRAIN duplicates: {r1}")
print(f"Removed TEST duplicates: {r2}")

# ========================= LEAKAGE CHECK =========================
print("\n================ LEAKAGE CHECK ================")
train_paths = set(sum(support_pool.values(), []))
test_paths = set(sum(query_pool.values(), []))

overlap = train_paths.intersection(test_paths)
train_hashes = {hash_image(p) for p in train_paths}
test_hashes = {hash_image(p) for p in test_paths}
dup = train_hashes.intersection(test_hashes)

print("Train-Test overlap:", len(overlap))
print("Exact duplicate images:", len(dup))

assert len(overlap) == 0, "Leakage detected!"
assert len(dup) == 0, "Duplicate leakage detected!"

print("✅ PASS: No leakage detected")

# ========================= SUMMARY =========================
print("\n================ FINAL DATASET STATS ================")
print("Support samples:", sum(len(v) for v in support_pool.values()))
print("Query samples  :", sum(len(v) for v in query_pool.values()))
print("Classes        :", CLASS_NAMES)

# ========================= EPISODE CREATION =========================


def create_episode(k_shot, support_classwise, query_classwise, query_per_class=15):
    support_paths, query_paths = [], []
    support_labels, query_labels = [], []
    for cls_idx in range(N_WAY):
        s_imgs = support_classwise[cls_idx]
        s_sel = random.choices(s_imgs, k=k_shot) if len(
            s_imgs) < k_shot else random.sample(s_imgs, k_shot)

        q_imgs = query_classwise[cls_idx]
        q_sel = random.choices(q_imgs, k=query_per_class) if len(
            q_imgs) < query_per_class else random.sample(q_imgs, query_per_class)

        support_paths.extend(s_sel)
        query_paths.extend(q_sel)
        support_labels.extend([cls_idx] * k_shot)
        query_labels.extend([cls_idx] * query_per_class)

    return (
        support_paths,
        query_paths,
        torch.tensor(support_labels, dtype=torch.long),
        torch.tensor(query_labels, dtype=torch.long)
    )


print("\n✅ Data setup completed. Ready for models & features.")
