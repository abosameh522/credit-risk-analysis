# Credit Risk Analysis

An analysis of historical credit records that looks at which recorded factors are associated with bad credit outcomes and compares three classifiers as illustrative benchmarks.

## Problem

A lender records applicant and loan details and later observes whether each credit was repaid. Two practical questions follow:

- Which recorded factors are associated with a bad outcome?
- Can a simple model separate bad from good outcomes well enough to be useful as a screening signal?

This project works on a small historical sample. It is an analysis and a model comparison, not a lending decision system.

## Dataset

Source: [UCI South German Credit](https://archive.ics.uci.edu/dataset/573/south+german+credit+update), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The raw `.asc` file and the code table (`codetable.txt`) are included under `data/raw/`.

| Item | Value |
| --- | --- |
| Records | 1,000 |
| Columns | 21 (20 predictors + outcome) |
| Measured numeric fields | Duration (months), credit amount (DM), age (years) |
| Coded categorical fields | 17 |
| Outcome | `0 = bad` (300), `1 = good` (700) |
| Period | 1973–1975 |

Important fields include `checking_status`, `duration_months`, `credit_history`, `credit_amount_dm`, `savings`, and `age_years`. The outcome is the positive class for bad credit (`bad_credit = 1`).

Two source limitations matter: bad credits were deliberately oversampled, so the 30% bad rate is not a population rate, and credit amounts were transformed in the original data with an unspecified method. There is no income variable.

## Analysis Workflow

```mermaid
flowchart LR
    A[South German Credit] --> B[Cleaning and code mapping]
    B --> C[Exploratory analysis]
    C --> D[Preprocessing]
    D --> E[Model training]
    E --> F[Evaluation]
    F --> G[Interpretation]
```

1. Inspect dimensions, types, missing values, duplicates, category codes, and target balance.
2. Rename fields and map coded categories using the original code table. No rows are removed.
3. Explore risk across credit amount, duration, checking status, savings, housing, and age.
4. Split 80/20 with stratification and fit imputation, one-hot encoding, and scaling inside a pipeline.
5. Compare Logistic Regression, Decision Tree, and Random Forest on the held-out test set.
6. Interpret the logistic regression coefficients as directional associations.

## Key Questions

- Which factors are associated with higher bad-credit rates?
- How does the checking-account status relate to outcomes?
- Does a longer duration or larger amount relate to higher risk?
- Which of the three models ranks bad outcomes best?
- Which features push predicted risk up or down?

## Exploratory Analysis

![Historical credit outcomes](images/credit_outcomes.png)

300 of the 1,000 records ended in a bad outcome. The classes are imbalanced, so accuracy alone is not enough to judge a model.

![Bad credit rate by checking account status](images/checking_risk.png)

Checking-account status shows the clearest split: records with no account had a 49.3% bad rate, and accounts at 200+ DM had the lowest at 11.7%.

![Bad credit rate by credit amount quartile](images/amount_risk.png)

The highest credit-amount quartile had a 42.0% bad rate, higher than the other three quartiles. These are descriptive associations, not causal estimates.

## Modeling

Three classifiers were trained with simple settings and evaluated on one held-out test set of 200 records (60 bad outcomes) at a 0.5 threshold.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.770 | 0.769 | 0.333 | 0.465 | 0.755 |
| Logistic Regression | 0.740 | 0.591 | 0.433 | 0.500 | 0.742 |
| Decision Tree | 0.735 | 0.640 | 0.267 | 0.376 | 0.689 |

Random Forest has the highest ROC-AUC. At the 0.5 threshold it catches 20 of 60 bad outcomes, so its recall is limited. Logistic Regression catches more bad outcomes (highest recall and F1) but with lower precision. Precision matters because false alarms create reviews; recall matters because missed bad outcomes can be costly.

![Held-out model comparison](images/model_comparison.png)

![Confusion matrix](images/confusion_matrix.png)

## Explainability

The logistic regression coefficients show the direction of each feature's association with predicted risk, after encoding and scaling. Positive coefficients push predicted bad-credit risk up; negative coefficients push it down.

![Selected logistic regression coefficients](images/feature_coefficients.png)

The strongest associations are checking status (no account and below 0 DM raise risk, 200+ DM lowers it), longer credit duration, and certain credit-history categories. Because the features are correlated, each coefficient is conditional on the others, and these are associations rather than causes. No SHAP or other model-agnostic method is used here.

## Key Findings

- Checking accounts below 0 DM had a 39.0% bad-outcome rate, versus 11.7% for accounts at 200+ DM.
- Credits longer than 24 months had a 44.3% bad-outcome rate, versus 25.7% for shorter credits.
- Above-median credit amounts had a 32.2% bad-outcome rate, versus 27.8% for the lower half.
- The highest amount quartile reached a 42.0% bad rate.
- The sample's 30% bad rate reflects deliberate oversampling and should not be read as a current rate.

## Project Structure

```text
credit-risk-analysis/
├── notebooks/
│   └── credit_risk_analysis.ipynb   # Full analysis and modeling
├── data/raw/
│   ├── SouthGermanCredit.asc        # Source data
│   └── codetable.txt                # Category code definitions
├── images/                          # Charts saved by the notebook
├── portfolio/                       # Three summary figures
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone https://github.com/abosameh522/credit-risk-analysis.git
cd credit-risk-analysis
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

cd notebooks
jupyter nbconvert --to notebook --execute --inplace credit_risk_analysis.ipynb
```

The notebook reads `../data/raw/SouthGermanCredit.asc` and writes charts to `../images/` and summary figures to `../portfolio/`. Run it from the `notebooks/` directory, as the paths are relative to it.

## Limitations

- The data are small, old (1973–1975), and deliberately oversampled.
- There is no income variable, and credit amounts were transformed in an unspecified way.
- The reported metrics come from a single held-out split; the winner is chosen from that same test set, with no cross-validation.
- Category rates and coefficients are associations, not causal effects.
- A score from this data should support human review, not decide lending outcomes.

## Future Improvements

- Use cross-validation and a separate validation set for model selection.
- Evaluate thresholds on validation data to balance recall and precision for bad outcomes.
- Add calibration and a fairness review before any real use.
- Try a boosting model as an additional comparison.

## Author

[Ahmed Sameh](https://github.com/abosameh522)
