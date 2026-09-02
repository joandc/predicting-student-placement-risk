# Student Placement Risk Prediction Using Machine Learning

This repository contains a capstone machine learning project that predicts whether a student may be at risk of not securing a placement. The project is designed as a decision-support prototype for a Career Development Cell (CDC), helping staff identify students who may benefit from earlier support before placement or recruitment activities begin.

Primary notebook:

- `capstone_student_placement_risk_prediction.ipynb`

## Project Objective

The goal of this project is to build and evaluate a classification model that predicts student placement risk using academic performance, technical skills, internship/project experience, and assessment-related features.

The model output is intended to support CDC staff by:

- identifying students with higher predicted placement risk;
- grouping students into low, medium, and high-risk bands;
- explaining major prediction drivers using SHAP;
- exporting prediction results for Power BI dashboard reporting;
- supporting a prototype student-facing Streamlit app.

## Dataset

The project uses a synthetic Kaggle student placement dataset with 9,000 student records. The original target variable is `placed`, where:

- `placed = 1` means the student received a job offer;
- `placed = 0` means the student did not receive a job offer.

For modeling, a derived target variable was created:

```text
at_risk = 1 - placed
```

This means:

- `at_risk = 1` represents students who were not placed;
- `at_risk = 0` represents students who were placed.

The dataset is imbalanced, with approximately:

- 86% placed students;
- 14% not placed / at-risk students.

Because of this imbalance, the project prioritizes recall for the minority `at_risk` class.

## Features Used

The model uses predictors from four main categories:

| Category | Example Features |
|---|---|
| Academic performance | `cgpa`, `college_tier`, `backlogs` |
| Technical skills | `python_skill`, `dsa_skill`, `ml_skill`, `web_dev_skill` |
| Experience | `internships`, `projects` |
| Assessment scores | `coding_score`, `communication_score`, `aptitude_score`, `resume_score` |

The following fields were excluded to avoid data leakage or duplication:

- `student_id`
- `placed`
- `at_risk`
- `salary_lpa`
- `company_type`
- `job_role`
- `skill_score`

`salary_lpa`, `company_type`, and `job_role` were excluded because they are post-outcome variables that would only be known after placement. `skill_score` was excluded because it is a direct sum of the individual skill columns.

## Modeling Workflow

The notebook follows this workflow:

0. Import libraries
1. Load the dataset
2. Perform exploratory data analysis
3. Create the `at_risk` target
4. Remove leakage columns
5. Define feature groups
6. Create train-validation-test split
7. Build preprocessing pipelines
8. Create baseline models
9. Define evaluation helper function
10. Train and evaluate baseline models
11. Tune hyperparameters
12. Run learning curve analysis
13. Compare tuned models on the validation set
14. Tune the classification threshold
15. Evaluate the selected model on the test set
16. Apply SHAP explainability
15. Export results for Power BI

The data was split into training, validation, and test sets so that:

- the training set is used for model fitting and hyperparameter tuning;
- the validation set is used for model comparison and threshold tuning;
- the test set is used only once for final evaluation.

## Models Compared

Three classification models were compared:

- Logistic Regression
- Random Forest
- XGBoost

Class imbalance was handled using:

- `class_weight="balanced"` for Logistic Regression and Random Forest;
- `scale_pos_weight` for XGBoost.

## Selected Model

The selected final model is:

```text
Logistic Regression with tuned threshold = 0.55
```

Logistic Regression was selected because it provided the best practical balance between:

- high recall for at-risk students;
- acceptable precision;
- strong PR-AUC and ROC-AUC;
- lower overfitting risk;
- easier interpretability for CDC staff.

## Final Test Performance

Final test results for the selected Logistic Regression model:

| Metric | Value |
|---|---:|
| Selected threshold | 0.55 |
| At-risk precision | 0.532 |
| At-risk recall | 0.908 |
| At-risk F1-score | 0.670 |
| Balanced accuracy | 0.886 |
| ROC-AUC | 0.958 |
| PR-AUC | 0.809 |

Final confusion matrix:

| Actual / Predicted | Not At Risk | At Risk |
|---|---:|---:|
| Not At Risk | 1,332 | 208 |
| At Risk | 24 | 236 |

The model correctly identified 236 out of 260 actually at-risk students in the test set.

## Why Recall Was Prioritized

The project focuses on identifying students who may need support. In this context, missing an actually at-risk student is more costly than flagging a student who may ultimately be placed.

Therefore, recall for the minority `at_risk` class was prioritized over accuracy alone.

## SHAP Explainability

SHAP analysis was used to interpret the final Logistic Regression model. The global SHAP bar plot was used to identify which features had the strongest overall influence on predicted placement risk.

The most influential features included:

- `resume_score`
- `backlogs`
- `coding_score`
- `college_tier`
- `internships`

SHAP explanations were also translated into plain-language reasons for dashboard use, such as:

- resume needs improvement;
- backlogs present;
- low coding score;
- no internship experience;
- low communication score.

Important note: SHAP explains how the trained model made predictions. It does not prove that these features caused placement or non-placement outcomes.

## Power BI Dashboard Output

The project exports prediction results for a Power BI dashboard. The dashboard is designed to help CDC staff review risk levels and prioritize student support.

Example dashboard pages:

1. **Overview**
   - total prediction records;
   - risk-band distribution;
   - overall placement-risk summary.

2. **High & Medium Risk Student Support**
   - filtered student list;
   - risk probability;
   - key SHAP-based risk drivers;
   - primary support category;
   - recommended interventions.

Power BI output files include:
- `powerbi_student_risk_predictions_shap.csv`


## Sample Screenshots - Power BI Dashboard Prototype
![Correlation Heatmap](screenshot-1.png)

![Correlation Heatmap](screenshot-2.png)

## Limitations

This project should be interpreted as a prototype, not a production-ready system.

Key limitations:

- The dataset is synthetic and sourced from Kaggle.
- The exact placement outcome deadline is not documented.
- Real institutional data would be needed before deployment.
- Model predictions should not be used to restrict student opportunities.
- SHAP explanations are predictive explanations, not causal conclusions.
- Future use would require monitoring, retraining, and validation across placement cycles.

## Recommended Future Improvements

Future work could include:

- testing the workflow on real institutional student data;
- defining a clear placement outcome timeframe;
- collecting consistent career-readiness scores;
- adding program-specific features where appropriate;
- monitoring model performance by program or cohort;
- evaluating whether CDC interventions improve student outcomes;
- expanding the Streamlit app and Power BI dashboard based on CDC feedback.

## How to Run the Notebook

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap ipykernel
```

Then open and run:

```text
capstone_student_placement_risk_prediction.ipynb
```

Recommended order:

1. Run all EDA and preprocessing cells.
2. Train baseline models.
3. Run hyperparameter tuning.
4. Evaluate tuned models on the validation set.
5. Tune the classification threshold.
6. Run final test evaluation once.
7. Run SHAP analysis.
8. Export Power BI files.

## Project Status

Status: completed capstone prototype.

Final selected model:

```text
Logistic Regression, threshold = 0.55
```

Primary intended user:

```text
Career Development Cell / career support staff
```
