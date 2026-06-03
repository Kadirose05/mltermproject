from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "openfoodfacts_nutrition_grade_clean.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
TARGET = "nutrition_grade_fr"
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(DATA_FILE)

    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("Logistic Regression Results")
    print(f"Train records: {len(x_train):,}")
    print(f"Test records: {len(x_test):,}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print()

    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=labels))

    print("Confusion matrix:")
    print(
        pd.DataFrame(
            matrix,
            index=[f"actual_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )
    )

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
    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()

    output_file = OUTPUT_DIR / "logistic_regression_confusion_matrix.png"
    plt.savefig(output_file, dpi=200)
    plt.close()

    print()
    print(f"Confusion matrix plot saved: {output_file}")


if __name__ == "__main__":
    main()
