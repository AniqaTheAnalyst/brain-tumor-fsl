# 1_data_and_setup.py
import torch
import random
import os
import hashlib
from PIL import Image

# ========================= CONSTANTS =========================
N_WAY = 4
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES_EVAL = 600
BOOTSTRAP_N = 1000
CI_ALPHA = 0.05
QUERY_PER_CLASS = 15

DATASET_PATH = "/kaggle/input/datasets/sartajbhuvaji/brain-tumor-classification-mri"
TRAIN_DIR = os.path.join(DATASET_PATH, "Training")
TEST_DIR = os.path.join(DATASET_PATH, "Testing")

print("Using Dataset:", DATASET_PATH)


def hash_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return hash(image_path)


def load_classwise_split(root):
    classwise = {i: [] for i in range(N_WAY)}
    print(f"\n🔍 Loading split from: {root}")

    for i, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(root, cls)
        if os.path.exists(cls_dir):
            paths = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            classwise[i] = paths
            print(f" ✅ {cls:18} → {len(paths)} images")
        else:
            print(f" ❌ Missing folder: {cls_dir}")
    return classwise


def create_episode(k_shot, support_classwise, query_classwise, query_per_class=15):
    support_paths, query_paths = [], []
    support_labels, query_labels = [], []
    for cls_idx in range(N_WAY):
        s_imgs = support_classwise[cls_idx]
        if len(s_imgs) < k_shot:
            s_sel = random.choices(s_imgs, k=k_shot)
        else:
            s_sel = random.sample(s_imgs, k_shot)

        q_imgs = query_classwise[cls_idx]
        if len(q_imgs) < query_per_class:
            q_sel = random.choices(q_imgs, k=query_per_class)
        else:
            q_sel = random.sample(q_imgs, query_per_class)

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


# ========================= LOAD DATA =========================
support_pool = load_classwise_split(TRAIN_DIR)
query_pool = load_classwise_split(TEST_DIR)

print("\nFinal Dataset Distribution:")
print("Support:", [len(support_pool[i]) for i in range(N_WAY)])
print("Query :", [len(query_pool[i]) for i in range(N_WAY)])

if sum(len(v) for v in support_pool.values()) == 0:
    raise ValueError("No images loaded! Check dataset path or folder names.")

print("\n================ LEAKAGE CHECK & REMOVAL ================")
train_paths = set(sum(support_pool.values(), []))
test_paths = set(sum(query_pool.values(), []))

# Hash-based duplicate detection
train_hashes_dict = {}
for cls, paths in support_pool.items():
    for p in paths:
        h = hash_image(p)
        train_hashes_dict.setdefault(h, []).append(p)

test_hashes_dict = {}
for cls, paths in query_pool.items():
    for p in paths:
        h = hash_image(p)
        test_hashes_dict.setdefault(h, []).append(p)

train_hashes_set = set(train_hashes_dict.keys())
test_hashes_set = set(test_hashes_dict.keys())
dup = train_hashes_set.intersection(test_hashes_set)

print(f"Train-Test path overlap: {len(train_paths & test_paths)}")
print(f"Exact duplicate images (by MD5): {len(dup)}")

if len(dup) > 0:
    print(f"⚠️ Found {len(dup)} leaking images. Removing from test set...")
    removed_from_test = 0
    new_query_pool = {}
    for cls, paths in query_pool.items():
        new_list = []
        for p in paths:
            if hash_image(p) in dup:
                removed_from_test += 1
                continue
            new_list.append(p)
        if new_list:
            new_query_pool[cls] = new_list
    query_pool = new_query_pool
    print(f"✅ Removed {removed_from_test} leaking images from TEST set.")

print("🎉 PASS: No data leakage between support and query pools.\n")

print("\n================ FINAL DATASET STATS ================")
print("Support samples:", sum(len(v) for v in support_pool.values()))
print("Query samples :", sum(len(v) for v in query_pool.values()))
print("Classes :", CLASS_NAMES)
