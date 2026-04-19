Model 1 → Dataset 1: Sartaj Brain Tumor Classification MRI

Source: https://github.com/sartajbhuvaji/brain-tumor-classification-dataset
Total Images: ~3,264
Classes (4 classes):
Glioma tumor
Meningioma tumor
Pituitary tumor
No tumor

Structure: Already split into Training and Testing folders
Notes:
Contains natural class imbalance
Some duplicate images and minor data leakage present

Usage: Used for Model 1 experiments (Baseline dataset)

Model 2 → Dataset 2: Brain Tumor MRI Dataset

Source: Masoud Nickparvar Brain Tumor MRI Dataset
Total Images: ~7,200
Classes (4 classes):
Glioma
Meningioma
Pituitary tumor
No tumor

Structure: Balanced dataset (~1400 train + 400 test images per class)
Improvements:
Duplicates removed
Data leakage significantly reduced
Better class balance
Improved label consistency

Usage: Used for Model 2 experiments (Improved / Curated dataset)
