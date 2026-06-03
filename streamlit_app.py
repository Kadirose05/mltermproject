from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgb_model_full.joblib"
DATA_PATH = BASE_DIR / "openfoodfacts_nutrition_grade_clean.csv"
TARGET = "nutrition_grade_fr"

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

GRADE_LABELS = {
    4: "A",
    3: "B",
    2: "C",
    1: "D",
    0: "E",
}

GRADE_COLORS = {
    "A": "#16803c",
    "B": "#7aa328",
    "C": "#d9a300",
    "D": "#d96b1c",
    "E": "#b42318",
}

RANGES = {
    "energy_100g": (0.0, 4000.0, 1.0),
    "fat_100g": (0.0, 100.0, 0.1),
    "saturated-fat_100g": (0.0, 100.0, 0.1),
    "carbohydrates_100g": (0.0, 100.0, 0.1),
    "sugars_100g": (0.0, 100.0, 0.1),
    "fiber_100g": (0.0, 100.0, 0.1),
    "proteins_100g": (0.0, 100.0, 0.1),
    "salt_100g": (0.0, 100.0, 0.01),
    "sodium_100g": (0.0, 40.0, 0.001),
    "trans-fat_100g": (0.0, 100.0, 0.1),
    "cholesterol_100g": (0.0, 10.0, 0.001),
    "calcium_100g": (0.0, 10.0, 0.001),
    "iron_100g": (0.0, 10.0, 0.0001),
    "potassium_100g": (0.0, 10.0, 0.001),
    "vitamin-c_100g": (0.0, 10.0, 0.0001),
    "vitamin-a_100g": (0.0, 1.0, 0.00001),
    "additives_n": (0.0, 100.0, 1.0),
    "ingredients_from_palm_oil_n": (0.0, 10.0, 1.0),
    "ingredients_that_may_be_from_palm_oil_n": (0.0, 10.0, 1.0),
    "ingredients_count": (1.0, 200.0, 1.0),
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_defaults():
    if not DATA_PATH.exists():
        return {feature: 0.0 for feature in FEATURES}
    df = pd.read_csv(DATA_PATH, usecols=FEATURES)
    return df.median(numeric_only=True).to_dict()


def energy_input(defaults):
    energy_unit = st.radio(
        "Energy unit",
        ["kJ / 100g", "kcal / 100g"],
        horizontal=True,
    )

    default_kj = float(defaults.get("energy_100g", 0.0))
    if energy_unit == "kcal / 100g":
        default_value = default_kj / 4.184
        maximum = 1000.0
        help_text = "Converted to kJ before prediction because the model was trained with energy_100g in kJ."
    else:
        default_value = default_kj
        maximum = 4000.0
        help_text = "Open Food Facts energy_100g is stored as kJ per 100g."

    entered_value = st.number_input(
        f"energy_100g ({energy_unit})",
        min_value=0.0,
        max_value=maximum,
        value=min(max(default_value, 0.0), maximum),
        step=1.0,
        format="%.3f",
        help=help_text,
    )

    if energy_unit == "kcal / 100g":
        return entered_value * 4.184
    return entered_value


def number_input(feature, defaults):
    minimum, maximum, step = RANGES[feature]
    value = float(defaults.get(feature, minimum))
    value = min(max(value, minimum), maximum)
    return st.number_input(
        feature,
        min_value=minimum,
        max_value=maximum,
        value=value,
        step=step,
        format="%.6f" if step < 0.001 else "%.3f",
    )


def render_grade(grade):
    color = GRADE_COLORS[grade]
    st.markdown(
        f"""
        <div style="
            border: 1px solid {color};
            border-left: 10px solid {color};
            border-radius: 8px;
            padding: 18px 20px;
            margin-bottom: 16px;
            background: #ffffff;">
            <div style="font-size: 14px; color: #555;">Predicted nutrition grade</div>
            <div style="font-size: 56px; font-weight: 700; color: {color}; line-height: 1;">{grade}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Nutrition Grade Prediction",
        page_icon="",
        layout="wide",
    )

    st.title("Nutrition Grade Prediction")

    model = load_model()
    defaults = load_defaults()

    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        st.subheader("Nutritional values per 100g")
        values = {}

        groups = [
            (
                "Macronutrients",
                [
                    "energy_100g",
                    "fat_100g",
                    "saturated-fat_100g",
                    "carbohydrates_100g",
                    "sugars_100g",
                    "fiber_100g",
                    "proteins_100g",
                    "salt_100g",
                    "sodium_100g",
                ],
            ),
            (
                "Additional nutrients",
                [
                    "trans-fat_100g",
                    "cholesterol_100g",
                    "calcium_100g",
                    "iron_100g",
                    "potassium_100g",
                    "vitamin-c_100g",
                    "vitamin-a_100g",
                ],
            ),
            (
                "Ingredient indicators",
                [
                    "additives_n",
                    "ingredients_from_palm_oil_n",
                    "ingredients_that_may_be_from_palm_oil_n",
                    "ingredients_count",
                ],
            ),
        ]

        for group_name, group_features in groups:
            with st.expander(group_name, expanded=group_name == "Macronutrients"):
                cols = st.columns(2)
                for index, feature in enumerate(group_features):
                    with cols[index % 2]:
                        if feature == "energy_100g":
                            values[feature] = energy_input(defaults)
                        else:
                            values[feature] = number_input(feature, defaults)

        predict = st.button("Predict", type="primary", use_container_width=True)

    input_df = pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)

    with right:
        st.subheader("Prediction")
        if predict:
            encoded_prediction = int(model.predict(input_df)[0])
            grade = GRADE_LABELS[encoded_prediction]
            render_grade(grade)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                probability_df = pd.DataFrame(
                    {
                        "Grade": [GRADE_LABELS[int(cls)] for cls in model.classes_],
                        "Probability": probabilities,
                    }
                ).sort_values("Grade")

                st.dataframe(
                    probability_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Probability": st.column_config.ProgressColumn(
                            "Probability",
                            min_value=0,
                            max_value=1,
                            format="%.3f",
                        )
                    },
                )
        else:
            st.info("Enter nutritional values and run prediction.")

        st.subheader("Input preview")
        st.dataframe(input_df, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
