# Group11_FORS-EMG2026-_CSE475
---

## FORS-EMG Hand Gesture Recognition — Group 11

*Course:* CSE475: Machine Learning, Summer 2026, East West University
*Instructor:* Dr. Raihan Ul Islam, Associate Professor, Dept. of CSE

*Group Members:*
- M Sadman Hossain Bhuiyan (ID: 2023-3-60-393)
- Sheikh Tanvir Mohiuddin (ID: 2022-1-60-351)
- Shanjida Sultana Akhi (ID: 2022-3-60-250)
- Sadman Jahan Mojumder (ID: 2022-1-60-324)

---

### Dataset

FORS-EMG (Rumman et al., 2024) — 19 subjects, 12 hand gestures, 3 forearm orientations, 8 EMG channels at 985 Hz. 3,420 recordings in .mat format (8000 samples × 8 channels each). Available on [Kaggle](https://www.kaggle.com/datasets/ummerummanchaity/fors-emg-a-novel-semg-dataset).

---

### Requirements

pip install torch scipy scikit-learn xgboost lightgbm numpy pandas matplotlib seaborn tqdm shap lime

---

### How to Run

1. Download the dataset from Kaggle and update DATA_DIR in each notebook.
2. Run notebooks in order: *Task 1* (EDA) → *Task 2* (Baselines) → *Task 3* (GNN + Ablation).

---

### Repository Structure
```text
├── Group11_FORS-EMG_task1_eda.ipynb
│   └── EDA
│
├── Group11_Fors_EMG_Task2_Baseline.ipynb
│   └── 10 baseline classifiers
│
├── Group11_FORS_EMG_Task3_GNN_expolit.ipynb
│   └── GNN + ablation + explainability (SHAP/LIME)
│
├── Group11_FORS_EMG_Improved_Baseline.ipynb
│   └── TSD features + improved 1D-CNN
│
├── Group11_FORS_EMG_GNN_Proposal.docx
│   └── Task 2 GNN proposal
│
├── Group11_Task3_Report.docx
│   └── Task 3 final report
│
└── models/
    └── gnn_final.pt
       └── Saved GNN weights + configuration
```

### Model & Preprocessing

*Preprocessing:* Butterworth bandpass 20–450 Hz + notch 50 Hz, 492-sample windows at 50% overlap (~31 windows per trial).

*Task 2 features:* 104 statistical features (13 per channel × 8 channels).

*Task 3 GNN:* 8 nodes (one per EMG channel), 7-dim TSD node features (M₀, M₂, M₄, Sparseness, IRF, COV, TKEO), correlation-based dynamic adjacency matrix. 3 message-passing layers → global mean+max pooling → FC head → 12 gesture classes. 26,380 parameters.

---

### Results

All results use subject-wise split: 15 train / 2 val / 2 test subjects.

*Task 2 — Best baselines:*

| Model | Val Accuracy |
|---|---|
| MLP (best) | 55.02% |
| LightGBM | 51.93% |
| 1D-CNN | 52.41% |

*Task 3 — GNN:*

| Split | Accuracy | Macro F1 |
|---|---|---|
| Validation | 30.55% | 0.279 |
| Test | 40.76% | 0.404 |
| 5-Fold CV (mean ± std) | 36.56% ± 5.30% | — |

Wilcoxon signed-rank test vs. MLP: W = 0, p = 0.0625 — not significant at α = 0.05.

Literature reference: Çelik & Can (2026) achieves 74.0% subject-wise with a hybrid TCN+BiLSTM+Transformer model.

---

### Key Experimental Findings

- *Graph construction is the most impactful factor.* Correlation-only edges with threshold τ = 0.5 give the best result (val acc 0.373), outperforming spatial-only (0.294) and combined (0.291) adjacency.
- *Batch normalisation is essential* — removing it drops accuracy by ~10 points.
- *The GNN does not surpass the MLP baseline* in its current form. The gap is explained by sparse node features (only 7-dim TSD), over-smoothing on the small 8-node graph, and subject-specific correlation patterns that are harder to generalise than flat statistics.
- Best single ablation config: correlation-only adjacency, τ = 0.5, dropout = 0.2, cosine LR schedule, Adam, 3 layers, BatchNorm ON.

---

### Explainability

- *SHAP:* M₀ (RMS energy) is the globally dominant feature. Hand Close is driven by anterior flexors (CH1, CH2, CH5, CH6); Wrist Extension by posterior extensors (CH3, CH4, CH7, CH8) — physiologically consistent.
- *LIME:* Correct predictions show high-confidence, biologically sensible feature contributions. Misclassified samples (e.g., Index → Right Angle) show low confidence and diffuse contributions, reflecting genuine gesture similarity rather than model failure.

---

### Technologies

Python · PyTorch · scikit-learn · XGBoost · LightGBM · SciPy · SHAP · LIME · NumPy · Pandas · Matplotlib
www.kaggle.com
