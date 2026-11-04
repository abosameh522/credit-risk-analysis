"""Nested cross-validation with cost-based thresholds and feature importance.

The original 200-row held-out test set is not used for hyperparameter tuning
or threshold selection. Nested CV runs on the 800 training rows. The test set
is evaluated once at the end for the selected model family.

Cost matrix: a bad applicant predicted good costs 5; a good applicant
predicted bad costs 1. Thresholds are chosen on inner out-of-fold predictions
to minimise that cost.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from cross_validation import NUMERIC_FEATURES, build_preprocess, load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGES_DIR = PROJECT_ROOT / "images"

RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
SEARCH_ITER = 20
COST_FALSE_NEGATIVE = 5  # bad applicant predicted good
COST_FALSE_POSITIVE = 1  # good applicant predicted bad
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)

METRICS = [
    "roc_auc",
    "accuracy_05",
    "precision_05",
    "recall_05",
    "f1_05",
    "accuracy_tuned",
    "precision_tuned",
    "recall_tuned",
    "f1_tuned",
    "threshold",
    "cost_05",
    "cost_tuned",
    "cost_per_applicant_05",
    "cost_per_applicant_tuned",
]


def build_specs() -> list[dict]:
    """Define the model families, including balanced class-weight variants."""
    return [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "space": {"model__C": loguniform(1e-3, 1e2)},
        },
        {
            "name": "Logistic Regression (balanced)",
            "estimator": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "space": {"model__C": loguniform(1e-3, 1e2)},
        },
        {
            "name": "Decision Tree",
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "space": {
                "model__max_depth": [3, 4, 5, 6, 8, None],
                "model__min_samples_leaf": [5, 10, 15, 20, 30],
                "model__criterion": ["gini", "entropy"],
            },
        },
        {
            "name": "Decision Tree (balanced)",
            "estimator": DecisionTreeClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            "space": {
                "model__max_depth": [3, 4, 5, 6, 8, None],
                "model__min_samples_leaf": [5, 10, 15, 20, 30],
                "model__criterion": ["gini", "entropy"],
            },
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=1
            ),
            "space": {
                "model__n_estimators": [200, 300],
                "model__max_depth": [None, 4, 6, 8, 10],
                "model__min_samples_leaf": [1, 3, 5, 10],
                "model__max_features": ["sqrt", "log2", None],
            },
        },
        {
            "name": "Random Forest (balanced)",
            "estimator": RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "space": {
                "model__n_estimators": [200, 300],
                "model__max_depth": [None, 4, 6, 8, 10],
                "model__min_samples_leaf": [1, 3, 5, 10],
                "model__max_features": ["sqrt", "log2", None],
            },
        },
        {
            "name": "HistGradientBoosting",
            "estimator": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "space": {
                "model__learning_rate": loguniform(0.03, 0.3),
                "model__max_iter": [100, 200, 300],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [10, 20, 30],
                "model__l2_regularization": [0.0, 0.1, 1.0],
            },
        },
        {
            "name": "HistGradientBoosting (balanced)",
            "estimator": HistGradientBoostingClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            "space": {
                "model__learning_rate": loguniform(0.03, 0.3),
                "model__max_iter": [100, 200, 300],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [10, 20, 30],
                "model__l2_regularization": [0.0, 0.1, 1.0],
            },
        },
    ]


def make_pipeline(estimator, categorical_features: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocess(categorical_features)),
            ("model", estimator),
        ]
    )


def cost_at_threshold(y_true, probabilities, threshold: float) -> int:
    """German Credit cost: 5 for a missed bad applicant, 1 for a false alarm."""
    y_true = np.asarray(y_true)
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    false_negatives = int(((y_true == 1) & (predicted == 0)).sum())
    false_positives = int(((y_true == 0) & (predicted == 1)).sum())
    return COST_FALSE_NEGATIVE * false_negatives + COST_FALSE_POSITIVE * false_positives


def select_threshold(y_true, probabilities) -> float:
    costs = [cost_at_threshold(y_true, probabilities, t) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmin(costs))])


def metrics_at(y_true, probabilities, threshold: float) -> dict[str, float]:
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted),
        "f1": f1_score(y_true, predicted),
    }


def run_nested_cv(
    features: pd.DataFrame, target: pd.Series, categorical_features: list[str]
) -> tuple[pd.DataFrame, dict[str, list[tuple]]]:
    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    records = []
    fitted = {}
    for spec in build_specs():
        name = spec["name"]
        fitted[name] = []
        for fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(features, target), start=1
        ):
            x_train = features.iloc[train_idx]
            x_test = features.iloc[test_idx]
            y_train = target.iloc[train_idx]
            y_test = target.iloc[test_idx]

            search = RandomizedSearchCV(
                make_pipeline(spec["estimator"], categorical_features),
                spec["space"],
                n_iter=SEARCH_ITER,
                scoring="roc_auc",
                cv=inner_cv,
                random_state=RANDOM_STATE,
                n_jobs=1,
                refit=True,
            )
            search.fit(x_train, y_train)
            best = search.best_estimator_

            inner_probabilities = cross_val_predict(
                best, x_train, y_train, cv=inner_cv, method="predict_proba"
            )[:, 1]
            threshold = select_threshold(y_train, inner_probabilities)

            probabilities = best.predict_proba(x_test)[:, 1]
            default_metrics = metrics_at(y_test, probabilities, 0.5)
            tuned_metrics = metrics_at(y_test, probabilities, threshold)
            default_cost = cost_at_threshold(y_test, probabilities, 0.5)
            tuned_cost = cost_at_threshold(y_test, probabilities, threshold)

            records.append(
                {
                    "model": name,
                    "fold": fold,
                    "roc_auc": roc_auc_score(y_test, probabilities),
                    **{f"{key}_05": value for key, value in default_metrics.items()},
                    **{f"{key}_tuned": value for key, value in tuned_metrics.items()},
                    "threshold": threshold,
                    "cost_05": default_cost,
                    "cost_tuned": tuned_cost,
                    "cost_per_applicant_05": default_cost / len(y_test),
                    "cost_per_applicant_tuned": tuned_cost / len(y_test),
                    "test_rows": len(y_test),
                    "best_params": str(search.best_params_),
                }
            )
            fitted[name].append((best, x_test, y_test))
            print(
                f"{name:36s} fold {fold}: auc={roc_auc_score(y_test, probabilities):.3f} "
                f"recall@0.5={default_metrics['recall']:.3f} "
                f"recall@{threshold:.2f}={tuned_metrics['recall']:.3f} "
                f"cost={tuned_cost}"
            )

    return pd.DataFrame(records), fitted


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in folds.groupby("model"):
        for metric in METRICS:
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": group[metric].mean(),
                    "std": group[metric].std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def class_weight_effect(folds: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("Logistic Regression", "Logistic Regression (balanced)"),
        ("Decision Tree", "Decision Tree (balanced)"),
        ("Random Forest", "Random Forest (balanced)"),
        ("HistGradientBoosting", "HistGradientBoosting (balanced)"),
    ]
    rows = []
    for base, balanced in pairs:
        base_folds = folds[folds["model"] == base].set_index("fold")
        balanced_folds = folds[folds["model"] == balanced].set_index("fold")
        for metric in ["recall_05", "recall_tuned", "cost_tuned", "precision_05"]:
            delta = balanced_folds[metric] - base_folds[metric]
            rows.append(
                {
                    "family": base,
                    "metric": metric,
                    "mean_delta": delta.mean(),
                    "std_delta": delta.std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def final_holdout_check(
    spec: dict,
    features: pd.DataFrame,
    target: pd.Series,
    categorical_features: list[str],
    inner_cv: StratifiedKFold,
) -> tuple[dict, Pipeline]:
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=RANDOM_STATE
    )
    search = RandomizedSearchCV(
        make_pipeline(spec["estimator"], categorical_features),
        spec["space"],
        n_iter=SEARCH_ITER,
        scoring="roc_auc",
        cv=inner_cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_
    inner_probabilities = cross_val_predict(
        model, x_train, y_train, cv=inner_cv, method="predict_proba"
    )[:, 1]
    threshold = select_threshold(y_train, inner_probabilities)
    probabilities = model.predict_proba(x_test)[:, 1]
    tuned_metrics = metrics_at(y_test, probabilities, threshold)
    default_metrics = metrics_at(y_test, probabilities, 0.5)
    result = {
        "model": spec["name"],
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "recall_05": default_metrics["recall"],
        "precision_05": default_metrics["precision"],
        "f1_05": default_metrics["f1"],
        "recall_tuned": tuned_metrics["recall"],
        "precision_tuned": tuned_metrics["precision"],
        "f1_tuned": tuned_metrics["f1"],
        "cost_05": cost_at_threshold(y_test, probabilities, 0.5),
        "cost_tuned": cost_at_threshold(y_test, probabilities, threshold),
        "test_rows": len(y_test),
        "best_params": str(search.best_params_),
    }
    return result, model


def plot_feature_importance(
    fitted: dict[str, list[tuple]], model_name: str
) -> pd.DataFrame:
    rows = []
    for fold, (model, x_test, y_test) in enumerate(fitted[model_name], start=1):
        result = permutation_importance(
            model,
            x_test,
            y_test,
            scoring="roc_auc",
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        for feature, importance in zip(
            x_test.columns, result.importances_mean, strict=True
        ):
            rows.append(
                {"fold": fold, "feature": feature, "importance": importance}
            )

    importances = pd.DataFrame(rows)
    summary = (
        importances.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    summary.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    top = summary.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        top["feature"],
        top["mean"],
        xerr=top["std"],
        color="#4C78A8",
        alpha=0.85,
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean decrease in ROC-AUC when shuffled (outer folds)")
    ax.set_title(f"Permutation importance: {model_name}")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "feature_importance.png", dpi=170)
    plt.close(fig)
    return summary


def print_summary(summary: pd.DataFrame) -> None:
    table = summary.assign(
        result=summary["mean"].map("{:.3f}".format)
        + " +/- "
        + summary["std"].map("{:.3f}".format)
    ).pivot(index="model", columns="metric", values="result")
    order = (
        summary[summary["metric"] == "cost_tuned"]
        .sort_values("mean")["model"]
        .tolist()
    )
    columns = ["roc_auc", "recall_05", "recall_tuned", "cost_tuned", "threshold"]
    print(table.loc[order, columns].to_string())
    print()


def main() -> None:
    df = load_data()
    features = df.drop(columns="bad_credit")
    target = df["bad_credit"]
    categorical_features = [c for c in features.columns if c not in NUMERIC_FEATURES]

    print(
        f"Rows: {len(df)}; bad-credit rate: {target.mean():.1%}; "
        f"outer folds: {OUTER_FOLDS}; inner folds: {INNER_FOLDS}"
    )
    print(
        f"Costs: missing a bad applicant = {COST_FALSE_NEGATIVE}, "
        f"false alarm = {COST_FALSE_POSITIVE}"
    )
    print()

    folds, fitted = run_nested_cv(features, target, categorical_features)
    summary = summarize(folds)
    effects = class_weight_effect(folds)

    REPORTS_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    folds.to_csv(REPORTS_DIR / "nested_cv_folds.csv", index=False)
    summary.to_csv(REPORTS_DIR / "nested_cv_summary.csv", index=False)
    effects.to_csv(REPORTS_DIR / "class_weight_effect.csv", index=False)

    print("\nNested CV (mean +/- std across outer folds)")
    print_summary(summary)

    best_name = (
        summary[summary["metric"] == "cost_tuned"]
        .sort_values("mean")
        .iloc[0]["model"]
    )
    print(f"Selected by lowest mean tuned cost: {best_name}\n")

    print("Class-weight effect (balanced minus base, mean +/- std)")
    effect_table = effects.assign(
        result=effects["mean_delta"].map("{:+.3f}".format)
        + " +/- "
        + effects["std_delta"].map("{:.3f}".format)
    ).pivot(index="family", columns="metric", values="result")
    print(effect_table.to_string())
    print()

    spec = {item["name"]: item for item in build_specs()}[best_name]
    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    holdout, _ = final_holdout_check(
        spec, features, target, categorical_features, inner_cv
    )
    pd.DataFrame([holdout]).to_csv(
        REPORTS_DIR / "final_holdout.csv", index=False
    )
    print("Final check on the untouched 200-row test set")
    print(
        f"{holdout['model']}: auc={holdout['roc_auc']:.3f} "
        f"recall@0.5={holdout['recall_05']:.3f} "
        f"recall@{holdout['threshold']:.2f}={holdout['recall_tuned']:.3f} "
        f"cost@0.5={holdout['cost_05']} cost@threshold={holdout['cost_tuned']}"
    )
    print()

    importances = plot_feature_importance(fitted, best_name)
    print("Top permutation importances (mean across outer folds)")
    print(importances.head(8).to_string(index=False))
    print()
    print("Wrote reports/nested_cv_folds.csv, nested_cv_summary.csv,")
    print("class_weight_effect.csv, final_holdout.csv, feature_importance.csv")
    print("Wrote images/feature_importance.png")


if __name__ == "__main__":
    main()
