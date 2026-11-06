# Dry Bean Variety Classification

## a. Problem statement

A grading line has to assign each dry bean to one of seven registered varieties from a camera image. The varieties overlap in size, so the smaller ones are easy to mix up by eye. This project trains five classifiers on 16 geometric measurements of each bean and compares them on the same held-out test set. The Streamlit app loads those saved models and scores a test CSV. It does not train again.

## b. Dataset description

The data is the Dry Bean Dataset from the UCI Machine Learning Repository.

Koklu, M. and Ozkan, I.A. (2020). Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques. *Computers and Electronics in Agriculture*, 174, 105507.

The file has **13,611 beans** and **16 numeric features**, with no missing values. That is above the assignment minimum of 12 features and 500 rows. Each feature is a size or shape measurement from a segmented image. The original column name `AspectRation` is kept as published.

Features: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.

The target column is `Class`.

| Variety | Beans | Share |
| --- | ---: | ---: |
| DERMASON | 3546 | 26.05% |
| SIRA | 2636 | 19.37% |
| SEKER | 2027 | 14.89% |
| HOROZ | 1928 | 14.17% |
| CALI | 1630 | 11.98% |
| BARBUNYA | 1322 | 9.71% |
| BOMBAY | 522 | 3.84% |

DERMASON is the largest class. A rule that always predicts DERMASON scores **0.2604** accuracy on the test split. The models below are judged against that baseline.

Many columns describe the same bean. Area and ConvexArea have correlation **0.9999**. Compactness and ShapeFactor3 have correlation **0.9987**. Twenty-one feature pairs have absolute correlation of at least 0.90.

The split is stratified 80/20 with `random_state=2026`: **10,888** training rows and **2,723** test rows. `StandardScaler` sits inside each saved pipeline and is fit on the training rows only. `test_data.csv` is that raw test split, including `Class`.

## c. Github Repository Link

Replace this before submission:

**https://github.com/shrejad/ml-assignment-2-drybean**

Live app, replace this too:

**https://github.com/shrejad/ml-assignment-2-drybean**

The repository contains `app.py`, `train_models.py`, `evaluate.py`, `inference.py`, `requirements.txt`, `runtime.txt`, `README.md`, `test_data.csv`, `data/dry_bean.csv`, `lab_execution.ipynb`, and `model/` (five pipelines plus `label_encoder.joblib`).

## d. Models used

All five models use the same split and the same scaled features.

| Model | Settings |
| --- | --- |
| Logistic Regression | `lbfgs`, `max_iter=2000` |
| Decision Tree | scikit-learn defaults, `random_state=2026` |
| KNN | `k=5`, uniform weights |
| Naive Bayes | `GaussianNB` |
| Support Vector Machine | RBF kernel, `C=1`, `gamma=scale`, probability estimates turned on so AUC is defined |

GaussianNB is used instead of MultinomialNB because the features are continuous measurements. After scaling, some values are negative, which a multinomial likelihood cannot take.

Precision, Recall, and F1 are **weighted** by the number of test beans in each variety. AUC is **weighted one-vs-rest** from `predict_proba`. MCC is the multiclass Matthews correlation. Values are shown to 4 decimal places. Weighted recall equals accuracy when each bean has one true variety, so those two columns match on purpose.

### Comparison table (2,723 held-out test beans)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9251 | 0.9937 | 0.9251 | 0.9251 | 0.9251 | 0.9095 |
| Decision Tree | 0.8898 | 0.9322 | 0.8902 | 0.8898 | 0.8899 | 0.8669 |
| KNN | 0.9210 | 0.9827 | 0.9211 | 0.9210 | 0.9211 | 0.9046 |
| Naive Bayes | 0.9008 | 0.9907 | 0.9015 | 0.9008 | 0.9008 | 0.8805 |
| Support Vector Machine | 0.9321 | 0.9942 | 0.9324 | 0.9321 | 0.9322 | 0.9179 |

Logistic regression precision is 0.925149 before rounding, and its accuracy is 0.925083. Both display as 0.9251.

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Test F1 is 0.9251, AUC is 0.9937, and MCC is 0.9095. Train accuracy is 0.9248 and test accuracy is 0.9251, so the linear boundary does not overfit. It finishes in under a second and is 0.0071 F1 behind the SVM. The main mistakes are symmetric: 46 DERMASON beans called SIRA, and 46 SIRA beans called DERMASON. |
| Decision Tree | This is the weakest test model: F1 0.8899, AUC 0.9322, MCC 0.8669. Train accuracy is 1.0000 and test accuracy is 0.8898, a gap of 0.1102. The tree grew to depth 28 with 724 leaves, so it memorised training beans. The largest test mix-ups are DERMASON/SIRA (68 and 59) and CALI predicted as BARBUNYA (20). The first splits use MajorAxisLength, ShapeFactor1, and Perimeter. |
| KNN | With k=5, test F1 is 0.9211, AUC is 0.9827, and MCC is 0.9046. Train accuracy is 0.9435 against test accuracy 0.9210. Neighbours only work after scaling. The same k on the raw numbers scores 0.7165 test accuracy, because Area and ConvexArea are in the thousands and the shape factors are near zero. SIRA is the weak class, with F1 0.8593. BOMBAY, the large variety, has F1 1.0000. |
| Naive Bayes | GaussianNB reaches test F1 0.9008 and MCC 0.8805, while AUC stays high at 0.9907. The ranking of varieties is good, and the chosen label is weaker. The model treats the 16 columns as independent, and they are not: Area and ConvexArea correlate at 0.9999. Train accuracy is 0.8951 and test accuracy is 0.9008, so this is bias rather than memorisation. It swaps DERMASON and SIRA often (67 and 40) and calls 29 BARBUNYA beans CALI. |
| Support Vector Machine | The RBF SVM has the best test F1 (0.9322), the best AUC (0.9942), and the best MCC (0.9179). Train accuracy is 0.9325 and test accuracy is 0.9321. It keeps 2,314 support vectors out of 10,888 training rows, about 21.3%, so the boundary is not a copy of the training set. BOMBAY has F1 1.0000 on all 104 test beans of that variety. SIRA is still hardest, at F1 0.8805. |
| Overall, Winner for your dataset? | Support Vector Machine. The rule is highest test weighted F1, then MCC, then AUC, then accuracy, and this model leads all three of F1, MCC, and AUC. Every model beats the 0.2604 majority baseline by a wide margin. Logistic regression is the closest alternative and is the one to prefer if fit time matters more than the 0.0071 F1 gap. |

## Reproduce and deploy

Pinned versions are in `requirements.txt` (scikit-learn 1.9.1, Python 3.12). The saved pipelines were fit with those versions.

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

On BITS Virtual Lab, run `lab_execution.ipynb` or `python train_models.py` from this folder and screenshot the comparison table.

Streamlit Community Cloud: branch `main`, entrypoint `app.py`. The app opens on `test_data.csv`, and a CSV upload replaces that file. The sidebar selects one of the five saved models. The page then shows the six metrics, a confusion matrix, a classification report, and a live comparison of all five models. Uploaded rows are scored with `predict` and `predict_proba` only.
