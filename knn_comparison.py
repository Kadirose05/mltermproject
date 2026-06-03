from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "openfoodfacts_nutrition_grade_clean.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
TARGET = "nutrition_grade_fr"
RANDOM_STATE = 42


def evaluate_model(name: str, model, x_test: pd.DataFrame, y_test: pd.Series, labels: list[str]) -> dict:
    y_pred = model.predict(x_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }

    print(name)
    print("-" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    print()

    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=labels))

    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix:")
    print(
        pd.DataFrame(
            matrix,
            index=[f"actual_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )
    )
    print()

    save_confusion_matrix_plot(name, matrix, labels)

    return metrics


def save_confusion_matrix_plot(name: str, matrix, labels: list[str]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()

    file_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    output_file = OUTPUT_DIR / f"{file_name}_confusion_matrix.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"Confusion matrix plot saved: {output_file}")
    print()


def main() -> None:
    df = pd.read_csv(DATA_FILE)

    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    labels = sorted(y.unique())

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("KNN Comparison")
    print("=" * 50)
    print(f"Train records: {len(x_train):,}")
    print(f"Test records: {len(x_test):,}")
    print()

    knn_without_cv = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]
    )
    knn_without_cv.fit(x_train, y_train)

    metrics_without_cv = evaluate_model(
        "KNN Without CV",
        knn_without_cv,
        x_test,
        y_test,
        labels,
    )

    knn_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier()),
        ]
    )

    param_grid = {
        "classifier__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"],
    }

    grid_search = GridSearchCV(
        estimator=knn_pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)

    print("Best KNN CV Parameters")
    print("-" * 50)
    print(grid_search.best_params_)
    print(f"Best CV macro F1: {grid_search.best_score_:.4f}")
    print()

    metrics_with_cv = evaluate_model(
        "KNN With CV",
        grid_search.best_estimator_,
        x_test,
        y_test,
        labels,
    )

    comparison = pd.DataFrame([metrics_without_cv, metrics_with_cv])
    print("Final Comparison")
    print("-" * 50)
    print(comparison.round(4))

    OUTPUT_DIR.mkdir(exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "knn_comparison_metrics.csv", index=False)
    print()
    print(f"Comparison saved: {OUTPUT_DIR / 'knn_comparison_metrics.csv'}")


if __name__ == "__main__":
    main()
