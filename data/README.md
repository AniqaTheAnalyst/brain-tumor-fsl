\##Datasets Information



This project uses two different publicly available brain MRI datasets to evaluate model robustness under different data conditions.



\---



\### Model 1-> Dataset 1: SARTAJ Brain Tumor Dataset

\- Source: https://github.com/sartajbhuvaji/brain-tumor-classification-dataset

\- Total Images: \~3264

\- Classes:

&#x20; - Glioma tumor

&#x20; - Meningioma tumor

&#x20; - Pituitary tumor

&#x20; - No tumor

\- Structure:

&#x20; - Already split into Training and Testing folders

\- Notes:

&#x20; - Contains class imbalance in some categories

&#x20; - Used for Model 1 experiments (baseline dataset)



\---



\### Model 2->Dataset 2: Brain Tumor MRI Dataset (Curated Version)

\- Total Images: 7200

\- Classes:

&#x20; - Glioma

&#x20; - Meningioma

&#x20; - Pituitary tumor

&#x20; - No tumor

\- Structure:

&#x20; - Balanced dataset (1400 train + 400 test per class)

\- Improvements:

&#x20; - Removed duplicate images

&#x20; - Reduced data leakage

&#x20; - Balanced class distribution

&#x20; - Improved label consistency

\- Used for Model 2 experiments (improved dataset)



