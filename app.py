
import os
import pickle
import streamlit as st
import pandas as pd
import sklearn

st.set_page_config(
    page_title="Starbucks Calorie Predictor",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Starbucks Calorie Predictor")
st.write("Predict the estimated calories in a Starbucks beverage.")

# Load the model from the same folder as app.py
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model.pkl")


@st.cache_resource
def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


if not os.path.isfile(MODEL_PATH):
    st.error("model.pkl not found in the app folder.")
    st.write("Expected path:", MODEL_PATH)
    st.stop()

try:
    bundle = load_model(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise ValueError(
            "This is not the expected model bundle. "
            "Run your Kaggle training code again."
        )

    model = bundle["model"]
    features = bundle["features"]
    options = bundle["options"]

except Exception as e:
    st.error("Could not load your model.")
    st.code(f"{type(e).__name__}: {e}")
    st.info(
        "Check your model file and use the same scikit-learn "
        "version as the one used during training."
    )
    st.stop()

st.caption(f"Model file found: {os.path.basename(MODEL_PATH)}")

# Collect user inputs
st.subheader("Select beverage details")
user_inputs = {}

for feature in features:
    values = options.get(feature, [])

    if not values:
        st.error(f"No input options found for {feature}.")
        st.stop()

    user_inputs[feature] = st.selectbox(
        feature.replace("_", " ").title(),
        options=values
    )

# Predict
if st.button("Predict Calories", type="primary"):
    input_df = pd.DataFrame([user_inputs], columns=features)

    try:
        prediction = float(model.predict(input_df)[0])

        st.success("Prediction completed!")
        st.metric(
            "Estimated Calories",
            f"{max(0, prediction):.0f} kcal"
        )

        st.caption(
            "This is a machine-learning estimate, not an official "
            "Starbucks nutrition value."
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.code(f"{type(e).__name__}: {e}")

# Model evaluation information
metrics = bundle.get("metrics", {})

with st.expander("About the model"):
    st.write("Algorithm: Decision Tree Regressor")
    st.write("Target:", bundle.get("target", "Calories"))

    if metrics:
        st.write("MAE:", metrics.get("mae", "Not available"))
        st.write("R²:", metrics.get("r2", "Not available"))
