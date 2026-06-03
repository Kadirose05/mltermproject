from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "openfoodfacts_nutrition_grade_clean.csv"
OUTPUT_DIR = BASE_DIR / "eda_outputs"
TARGET = "nutrition_grade_fr"


def print_dataset_overview(df: pd.DataFrame, features: list[str]) -> None:
    print("DATASET OVERVIEW")
    print("-" * 50)
    print(f"Number of records: {len(df):,}")
    print(f"Number of features: {len(features)}")
    print(f"Target column: {TARGET}")
    print(f"Target class count: {df[TARGET].nunique()}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate records: {int(df.duplicated().sum())}")
    print()


def print_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    target_distribution = pd.DataFrame(
        {
            "count": df[TARGET].value_counts().sort_index(),
            "percentage": (df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2),
        }
    )

    print("TARGET DISTRIBUTION")
    print("-" * 50)
    print(target_distribution)
    print()

    return target_distribution


def print_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_values = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percentage": (df.isna().mean() * 100).round(2),
        }
    )

    print("MISSING VALUE ANALYSIS")
    print("-" * 50)
    print(missing_values)
    print()

    return missing_values


def print_zero_value_analysis(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    zero_values = pd.DataFrame(
        {
            "zero_count": (df[features] == 0).sum(),
            "zero_percentage": ((df[features] == 0).mean() * 100).round(2),
        }
    ).sort_values("zero_percentage", ascending=False)

    print("ZERO VALUE ANALYSIS")
    print("-" * 50)
    print(zero_values)
    print()

    return zero_values


def print_feature_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    summary = df[features].describe().T
    summary = summary[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(4)

    print("FEATURE SUMMARY STATISTICS")
    print("-" * 50)
    print(summary)
    print()

    return summary


def print_correlation_analysis(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    correlation_matrix = df[features].corr(method="pearson")

    pairs = []
    for index, first_feature in enumerate(features):
        for second_feature in features[index + 1 :]:
            correlation = correlation_matrix.loc[first_feature, second_feature]
            pairs.append(
                {
                    "feature_1": first_feature,
                    "feature_2": second_feature,
                    "correlation": round(correlation, 4),
                    "absolute_correlation": round(abs(correlation), 4),
                }
            )

    correlation_pairs = pd.DataFrame(pairs).sort_values(
        "absolute_correlation",
        ascending=False,
    )

    print("TOP FEATURE CORRELATIONS")
    print("-" * 50)
    print(correlation_pairs.head(15)[["feature_1", "feature_2", "correlation"]])
    print()

    print("HIGH CORRELATIONS (absolute correlation >= 0.80)")
    print("-" * 50)
    high_correlations = correlation_pairs[correlation_pairs["absolute_correlation"] >= 0.80]
    if high_correlations.empty:
        print("No feature pairs with absolute correlation >= 0.80")
    else:
        print(high_correlations[["feature_1", "feature_2", "correlation"]])
    print()

    return correlation_matrix


def print_target_correlation(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    grade_mapping = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
    target_numeric = df[TARGET].map(grade_mapping)
    target_correlation = (
        df[features]
        .corrwith(target_numeric)
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .round(4)
        .to_frame("correlation_with_target")
    )

    print("FEATURE CORRELATION WITH TARGET")
    print("-" * 50)
    print(target_correlation)
    print()

    return target_correlation


def save_tables(
    target_distribution: pd.DataFrame,
    missing_values: pd.DataFrame,
    zero_values: pd.DataFrame,
    feature_summary: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    target_correlation: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    target_distribution.to_csv(OUTPUT_DIR / "target_distribution.csv")
    missing_values.to_csv(OUTPUT_DIR / "missing_values.csv")
    zero_values.to_csv(OUTPUT_DIR / "zero_value_analysis.csv")
    feature_summary.to_csv(OUTPUT_DIR / "feature_summary.csv")
    correlation_matrix.round(4).to_csv(OUTPUT_DIR / "feature_correlation_matrix.csv")
    target_correlation.to_csv(OUTPUT_DIR / "target_correlation.csv")


def save_plots_if_available(
    df: pd.DataFrame,
    features: list[str],
    zero_values: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError:
        print("PLOT GENERATION SKIPPED")
        print("-" * 50)
        print("matplotlib/seaborn is not installed.")
        print("Install them with: pip install matplotlib seaborn")
        print()
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET, order=sorted(df[TARGET].unique()))
    plt.title("Target Class Distribution")
    plt.xlabel("Nutrition Grade")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=zero_values.reset_index().rename(columns={"index": "feature"}),
        x="zero_percentage",
        y="feature",
    )
    plt.title("Zero Value Percentage by Feature")
    plt.xlabel("Zero Percentage")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "zero_value_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_correlation_heatmap.png", dpi=200)
    plt.close()

    selected_features = [
        "energy_100g",
        "fat_100g",
        "saturated-fat_100g",
        "sugars_100g",
        "fiber_100g",
        "proteins_100g",
        "sodium_100g",
        "ingredients_count",
    ]

    for feature in selected_features:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=feature, bins=40, kde=True)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{feature}_distribution.png", dpi=200)
        plt.close()

    print("PLOTS SAVED")
    print("-" * 50)
    print(f"Output folder: {OUTPUT_DIR}")
    print()


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    features = [column for column in df.columns if column != TARGET]

    print_dataset_overview(df, features)
    target_distribution = print_target_distribution(df)
    missing_values = print_missing_values(df)
    zero_values = print_zero_value_analysis(df, features)
    feature_summary = print_feature_summary(df, features)
    correlation_matrix = print_correlation_analysis(df, features)
    target_correlation = print_target_correlation(df, features)

    save_tables(
        target_distribution,
        missing_values,
        zero_values,
        feature_summary,
        correlation_matrix,
        target_correlation,
    )
    save_plots_if_available(df, features, zero_values, correlation_matrix)

    print("EDA tables saved successfully.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
