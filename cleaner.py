from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "en.openfoodfacts.org.products.tsv"
OUTPUT_FILE = BASE_DIR / "openfoodfacts_nutrition_grade_clean.csv"

TARGET = "nutrition_grade_fr"

RAW_COLUMNS = [
    TARGET,
    "ingredients_text",
    "additives_n",
    "ingredients_from_palm_oil_n",
    "ingredients_that_may_be_from_palm_oil_n",
    "energy_100g",
    "fat_100g",
    "saturated-fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",
    "trans-fat_100g",
    "cholesterol_100g",
    "calcium_100g",
    "iron_100g",
    "potassium_100g",
    "vitamin-c_100g",
    "vitamin-a_100g",
]

FEATURES = [
    "energy_100g",
    "fat_100g",
    "saturated-fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",
    "trans-fat_100g",
    "cholesterol_100g",
    "calcium_100g",
    "iron_100g",
    "potassium_100g",
    "vitamin-c_100g",
    "vitamin-a_100g",
    "additives_n",
    "ingredients_from_palm_oil_n",
    "ingredients_that_may_be_from_palm_oil_n",
    "ingredients_count",
]

NUMERIC_FEATURES = [feature for feature in FEATURES if feature != "ingredients_count"]

PLAUSIBLE_RANGES = {
    "energy_100g": (0, 4000),
    "fat_100g": (0, 100),
    "saturated-fat_100g": (0, 100),
    "carbohydrates_100g": (0, 100),
    "sugars_100g": (0, 100),
    "fiber_100g": (0, 100),
    "proteins_100g": (0, 100),
    "salt_100g": (0, 100),
    "sodium_100g": (0, 40),
    "trans-fat_100g": (0, 100),
    "cholesterol_100g": (0, 10),
    "calcium_100g": (0, 10),
    "iron_100g": (0, 10),
    "potassium_100g": (0, 10),
    "vitamin-c_100g": (0, 10),
    "vitamin-a_100g": (0, 10),
    "additives_n": (0, 100),
    "ingredients_from_palm_oil_n": (0, 100),
    "ingredients_that_may_be_from_palm_oil_n": (0, 100),
    "ingredients_count": (1, 200),
}

CORE_NUTRIENTS = [
    "energy_100g",
    "fat_100g",
    "saturated-fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",
]

MACRO_NUTRIENTS = [
    "fat_100g",
    "carbohydrates_100g",
    "proteins_100g",
    "fiber_100g",
]


def count_ingredients(value: object) -> int:
    if pd.isna(value):
        return 0

    return len([part for part in str(value).split(",") if part.strip()])


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk[chunk[TARGET].isin(list("abcde"))].copy()

    for column in NUMERIC_FEATURES:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

    chunk["ingredients_count"] = chunk["ingredients_text"].map(count_ingredients)
    chunk = chunk.dropna(subset=FEATURES + [TARGET])

    mask = pd.Series(True, index=chunk.index)
    for column, (lower, upper) in PLAUSIBLE_RANGES.items():
        mask &= chunk[column].between(lower, upper)

    # Drop products whose nutrition table is effectively empty but encoded as 0.
    mask &= chunk["energy_100g"] > 0
    mask &= (chunk[MACRO_NUTRIENTS] != 0).any(axis=1)
    mask &= (chunk[CORE_NUTRIENTS] != 0).sum(axis=1) >= 4

    return chunk.loc[mask, FEATURES + [TARGET]]


def main() -> None:
    cleaned_parts = []

    for chunk in pd.read_csv(
        INPUT_FILE,
        sep="\t",
        usecols=RAW_COLUMNS,
        chunksize=50_000,
        low_memory=False,
    ):
        cleaned = clean_chunk(chunk)
        if not cleaned.empty:
            cleaned_parts.append(cleaned)

    if not cleaned_parts:
        raise RuntimeError("No usable records were found.")

    final_df = pd.concat(cleaned_parts, ignore_index=True)
    final_df = final_df.drop_duplicates().reset_index(drop=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Records: {len(final_df):,}")
    print(f"Features: {len(FEATURES)}")
    print("Target distribution:")
    print(final_df[TARGET].value_counts().sort_index())


if __name__ == "__main__":
    main()
