"""Stratified k-fold cross-validation for the three credit-risk models.

This reproduces the preprocessing and model settings from
notebooks/credit_risk_analysis.ipynb, then reports the mean and standard
deviation of each metric across folds. Results are written to reports/.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/raw/SouthGermanCredit.asc"
REPORTS_DIR = PROJECT_ROOT / "reports"

COLUMN_NAMES = {
    "laufkont": "checking_status",
    "laufzeit": "duration_months",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "credit_amount_dm",
    "sparkont": "savings",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "residence_duration",
    "verm": "property",
    "alter": "age_years",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "number_credits",
    "beruf": "job",
    "pers": "people_liable",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}

NUMERIC_FEATURES = ["duration_months", "credit_amount_dm", "age_years"]

CATEGORY_LABELS = {
    "checking_status": {
        1: "No account",
        2: "Below 0 DM",
        3: "0-200 DM",
        4: "200+ DM",
    },
    "credit_history": {
        0: "Past delay",
        1: "Critical/other credit",
        2: "No/prior paid",
        3: "Current paid",
        4: "All paid",
    },
    "purpose": {
        0: "Other",
        1: "New car",
        2: "Used car",
        3: "Furniture",
        4: "Radio/TV",
        5: "Appliances",
        6: "Repairs",
        7: "Education",
        8: "Vacation",
        9: "Retraining",
        10: "Business",
    },
    "savings": {
        1: "Unknown/no account",
        2: "Below 100 DM",
        3: "100-500 DM",
        4: "500-1000 DM",
        5: "1000+ DM",
    },
    "employment_duration": {
        1: "Unemployed",
        2: "Below 1 year",
        3: "1-4 years",
        4: "4-7 years",
        5: "7+ years",
    },
    "installment_rate": {1: "35%+", 2: "25-35%", 3: "20-25%", 4: "Below 20%"},
    "personal_status_sex": {
        1: "Male separated",
        2: "Female non-single/male single",
        3: "Male married/widowed",
        4: "Female single",
    },
    "other_debtors": {1: "None", 2: "Co-applicant", 3: "Guarantor"},
    "residence_duration": {
        1: "Below 1 year",
        2: "1-4 years",
        3: "4-7 years",
        4: "7+ years",
    },
    "property": {
        1: "Unknown/none",
        2: "Car/other",
        3: "Savings/insurance",
        4: "Real estate",
    },
    "other_installment_plans": {1: "Bank", 2: "Stores", 3: "None"},
    "housing": {1: "Free", 2: "Rent", 3: "Own"},
    "number_credits": {1: "1", 2: "2-3", 3: "4-5", 4: "6+"},
    "job": {
        1: "Unemployed/nonresident",
        2: "Unskilled",
        3: "Skilled",
        4: "Manager/self-employed",
    },
    "people_liable": {1: "3+", 2: "0-2"},
    "telephone": {1: "No", 2: "Yes"},
    "foreign_worker": {1: "Yes", 2: "No"},
}

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=15, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, random_state=42
    ),
}

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
N_SPLITS = 5
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load and map the South German Credit data exactly as the notebook does."""
    df = pd.read_csv(DATA_PATH, sep=r"\s+").rename(columns=COLUMN_NAMES)

    for column, labels in CATEGORY_LABELS.items():
        unexpected = set(df[column].dropna().unique()) - set(labels)
        if unexpected:
            raise ValueError(f"Unexpected codes in {column}: {unexpected}")
        df[column] = df[column].map(labels).astype("category")

    if set(df["credit_risk"].unique()) != {0, 1}:
        raise ValueError("Unexpected target values")

    df["bad_credit"] = (df["credit_risk"] == 0).astype(int)
    return df.drop(columns="credit_risk")


def build_preprocess(categorical_features: list[str]) -> ColumnTransformer:
    """Build the preprocessing step used inside every model pipeline."""
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_pipelines(categorical_features: list[str]) -> dict[str, Pipeline]:
    """Create the same preprocessing and model pipelines used in the notebook."""
    preprocess = build_preprocess(categorical_features)
    return {
        name: Pipeline([("preprocess", preprocess), ("model", estimator)])
        for name, estimator in MODELS.items()
    }


def run_cross_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run stratified k-fold CV and return per-fold and summary results."""
    df = load_data()
    features = df.drop(columns="bad_credit")
    target = df["bad_credit"]
    categorical_features = [c for c in features.columns if c not in NUMERIC_FEATURES]

    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    pipelines = build_pipelines(categorical_features)

    fold_rows = []
    for name, pipeline in pipelines.items():
        scores = cross_validate(pipeline, features, target, cv=cv, scoring=METRICS)
        for fold in range(N_SPLITS):
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold + 1,
                    **{metric: scores[f"test_{metric}"][fold] for metric in METRICS},
                }
            )

    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.melt(
            id_vars=["model", "fold"],
            value_vars=METRICS,
            var_name="metric",
            value_name="value",
        )
        .groupby(["model", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    folds.to_csv(REPORTS_DIR / "cross_validation_folds.csv", index=False)
    summary.to_csv(REPORTS_DIR / "cross_validation_summary.csv", index=False)
    return folds, summary


def main() -> None:
    df = load_data()
    folds, summary = run_cross_validation()

    print(f"Rows: {len(df)}; bad-credit rate: {df['bad_credit'].mean():.1%}")
    print(f"Folds: {N_SPLITS} (stratified, shuffled, random_state={RANDOM_STATE})")
    print()

    table = summary.assign(
        result=summary["mean"].map("{:.3f}".format)
        + " +/- "
        + summary["std"].map("{:.3f}".format)
    ).pivot(index="model", columns="metric", values="result")[METRICS]
    table = table.loc[list(MODELS)]
    print(table.to_string())
    print()
    print("Per-fold scores written to reports/cross_validation_folds.csv")
    print("Mean/std written to reports/cross_validation_summary.csv")
    print(f"Fold rows: {len(folds)}")


if __name__ == "__main__":
    main()
