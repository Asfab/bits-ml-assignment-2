# Dry Bean Variety Classification — ML Assignment 2

M.Tech (AIML), BITS Pilani WILP · Machine Learning · Assignment 2

## a. Problem statement

Dry beans are the most produced pulse crop in the world, and seed quality depends on selling each
variety separately. Sorting beans by hand is slow and error-prone. The goal of this project is to
predict the **variety of a dry bean (7 classes)** from 16 shape measurements that a computer-vision
system takes from one photo of the bean.

This is a **multi-class classification** problem. Five classifiers are trained on the same data,
compared with six evaluation metrics, and served through an interactive Streamlit web app that
evaluates the saved models on uploaded test data without retraining them.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Dry Bean Dataset |
| Source | UCI Machine Learning Repository, dataset 602 — https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |
| Authors | Koklu, M. & Ozkan, I. A. (2020), *Computers and Electronics in Agriculture*, 174, 105507 |
| Instances | 13,611 (13,543 after removing 68 exact duplicate rows) |
| Features | 16, all numeric (continuous) |
| Target column | `Class` — 7 bean varieties |
| Missing values | None |
| Train / test split | 80% / 20%, stratified by class, `random_state=42` → 10,834 training rows, 2,709 test rows |

**Features** (all computed from the bean's outline in a high-resolution image):

| Group | Features |
|---|---|
| Size | Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, EquivDiameter |
| Shape ratios | AspectRation, Eccentricity, Extent, Solidity, roundness, Compactness |
| Shape factors | ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4 |

**Class distribution** (after removing duplicates — the classes are **imbalanced**):

| Variety | Rows | Share |
|---|---:|---:|
| DERMASON | 3,546 | 26.2% |
| SIRA | 2,636 | 19.5% |
| SEKER | 2,027 | 15.0% |
| HOROZ | 1,860 | 13.7% |
| CALI | 1,630 | 12.0% |
| BARBUNYA | 1,322 | 9.8% |
| BOMBAY | 522 | 3.9% |

**Preprocessing** (stored inside every saved model pipeline, so the app never fits anything):
- Exact duplicate rows removed, so the same bean cannot appear in both the training and the test set.
- All features standardised (mean 0, standard deviation 1). The features have very different
  ranges — `Area` is around 20,000–250,000, while `ShapeFactor2` is around 0.0006–0.004 — so without
  scaling, the distance-based models (KNN, SVM) and Logistic Regression would be dominated by the
  size features.
- A median imputer is included for safety, although this dataset has no missing values.

## c. GitHub repository

Repository: **https://github.com/Asfab/bits-ml-assignment-2**

Live Streamlit app: 

```
ml-assignment-2/
├── app.py                    # Streamlit app: loads saved models, never retrains
├── train_models.py           # trains and evaluates the 5 models, saves them
├── model_evaluation.py       # shared metric code used by training and by the app
├── requirements.txt          # pinned library versions (Python 3.10–3.13)
├── README.md
├── test_data.csv             # held-out 20% test split (16 features + Class)
├── data/
│   └── dry_bean_dataset.csv  # original UCI dataset
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k_nearest_neighbors.joblib
    ├── gaussian_naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── training_metadata.json   # features, classes, metrics, library versions
    └── metrics_table.md         # comparison table produced by training
```

How to reproduce:

```bash
pip install -r requirements.txt
python train_models.py      # trains all 5 models, writes model/ and test_data.csv
streamlit run app.py        # opens the app at http://localhost:8501
```

## d. Models used

| Model | Settings used |
|---|---|
| Logistic Regression | multinomial, `max_iter=3000` |
| Decision Tree | `max_depth=10`, `min_samples_leaf=5` (limits overfitting) |
| K-Nearest Neighbors | `k=9`, distance-weighted votes |
| Naive Bayes | Gaussian (all features are continuous) |
| Support Vector Machine | RBF kernel, `C=10`, `probability=True` (needed for AUC) |

### Comparison table (test set, 2,709 beans)

Because the problem has 7 classes, Precision, Recall and F1 are **weighted averages** over the
classes (each class weighted by its number of test rows), and AUC is the **one-vs-rest** AUC,
weighted the same way.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9195 | 0.9935 | 0.9201 | 0.9195 | 0.9197 | 0.9028 |
| Decision Tree | 0.9070 | 0.9729 | 0.9075 | 0.9070 | 0.9071 | 0.8875 |
| KNN | 0.9133 | 0.9851 | 0.9142 | 0.9133 | 0.9134 | 0.8951 |
| Naive Bayes | 0.8970 | 0.9899 | 0.8997 | 0.8970 | 0.8972 | 0.8762 |
| Support Vector Machine | 0.9236 | 0.9931 | 0.9238 | 0.9236 | 0.9236 | 0.9076 |

Supporting numbers used in the observations below:

| Model | Training accuracy | Test accuracy | Gap | Recall on SIRA | Recall on BARBUNYA |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9262 | 0.9195 | +0.0067 | 0.886 | 0.891 |
| Decision Tree | 0.9518 | 0.9070 | +0.0448 | 0.848 | 0.902 |
| KNN | 1.0000 | 0.9133 | +0.0867 | 0.871 | 0.868 |
| Naive Bayes | 0.8972 | 0.8970 | +0.0002 | 0.896 | 0.796 |
| Support Vector Machine | 0.9396 | 0.9236 | +0.0160 | 0.871 | 0.913 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second best overall (accuracy 0.9195, MCC 0.9028) and the **highest AUC (0.9935)**, so its predicted probabilities rank the classes very well. The training–test gap is very small (0.0067), so it generalises well. Its strong result shows that, after scaling, most varieties are almost linearly separable by size and shape. Its main errors are DERMASON ↔ SIRA (94 of its errors). It is also the fastest model to train (under 0.2 s) and easy to interpret. |
| Decision Tree | Lowest AUC (0.9729) and second-lowest MCC (0.8875). A tree gives "step-like" probabilities (the class shares in one leaf), so its ranking of classes is coarser than that of the other models, which lowers AUC. Even with `max_depth=10` and `min_samples_leaf=5`, training accuracy (0.9518) is clearly higher than test accuracy (0.9070), which shows some overfitting. It has the weakest recall on SIRA (0.848), because axis-aligned splits on single features cannot model the curved border between SIRA and DERMASON well. |
| KNN | Mid-ranked (accuracy 0.9133, MCC 0.8951). Training accuracy of 1.0 is expected with distance weighting (each training bean is its own nearest neighbour), so the test score is the honest measure. Scaling was essential: without it, the distance would be decided almost only by `Area` and `ConvexArea`. Many features are strongly correlated (for example Area vs ConvexArea ≈ 1.0), so the distance counts the "size" information several times, and that explains why KNN does a little worse than SVM. Its recall on BARBUNYA (0.868) is the second lowest, because it mixes BARBUNYA up with CALI. |
| Naive Bayes | Lowest accuracy (0.8970) and MCC (0.8762), but a high AUC (0.9899). Gaussian NB assumes that the features are independent within each class. Here this assumption is strongly violated (11 of the 16 features have a correlation above 0.9 with another feature), so the model counts the same evidence several times and makes over-confident, sometimes wrong, decisions. Its probability *ranking* is still good, which is why AUC stays high while accuracy drops. It struggles most with BARBUNYA (recall 0.796). It shows no overfitting (training and test accuracy are both 0.897). |
| Support Vector Machine | **Best on Accuracy (0.9236), Precision, Recall, F1 and MCC (0.9076)**, and its AUC (0.9931) is almost equal to the best. The RBF kernel draws curved class borders, which helps exactly where linear models struggle (best DERMASON recall 0.924 and best BARBUNYA recall 0.913). The training–test gap is small (0.016), so `C=10` does not overfit. Its cost is the slowest training (about 2 s, including the extra cross-validation needed for `probability=True`), which is still very small for this dataset. |
| **Overall winner for this dataset** | **Support Vector Machine.** It has the highest MCC (0.9076), the best single metric for an imbalanced 7-class problem, because it uses all cells of the confusion matrix. It is also best on Accuracy, Precision, Recall and F1. Logistic Regression is a very close second (MCC 0.9028, best AUC) and would be a good choice if a simpler, more explainable model is preferred. For all models, the hardest pair is **DERMASON vs SIRA**, two small white varieties with very similar shapes, and the easiest class is **BOMBAY** (100% recall for every model), because those beans are much larger than all the others. |

## Streamlit app features

- **CSV upload** of labelled test data (or one click to use the bundled `test_data.csv`, which can also be downloaded from the sidebar)
- **Model selection dropdown** for the five classifiers
- **All six evaluation metrics** for the selected model
- **Confusion matrix** heat map and **classification report** per variety
- **Compare all models** tab: metrics table with the best value in each column highlighted, and a bar chart for any metric
- **About** tab: variety descriptions, and training vs test accuracy for each model
- Download of predictions as CSV
