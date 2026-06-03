from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "openfoodfacts_nutrition_grade_clean.csv"
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

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("Decision Tree Results")
    print(f"Train records: {len(x_train):,}")
    print(f"Test records: {len(x_test):,}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print()

    labels = sorted(y.unique())
    print("Classification report:")
    print(classification_report(y_test, y_pred, labels=labels))

    print("Confusion matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels),
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ))

    print()
    print("Top feature importances:")
    importances = pd.Series(model.feature_importances_, index=x.columns)
    print(importances.sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
