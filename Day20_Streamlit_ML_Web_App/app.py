from pathlib import Path
import joblib 
import streamlit as st

st.set_page_config(
    page_title='Iris Flower Predictor',
    page_icon='🌸',
    layout='centered'
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR /"model"/"iris_model.joblib"

if not MODEL_PATH.exists():
    st.error("Model not found")
    st.stop()

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
target_names = model_package["target_names"]

st.title("🌸 Iris Flower Predictor")

st.write("Enter the flower measurements below "
    "to predict its species.")

sepal_length = st.slider(
    "Sepal length (cm)",
    min_value=4.0,
    max_value=8.0,
    value=5.1,
    step=0.1
)

sepal_width = st.slider(
    "Sepal width (cm)",
    min_value=2.0,
    max_value=4.5,
    value=3.5,
    step=0.1
)

petal_length = st.slider(
    "Petal length (cm)",
    min_value=1.0,
    max_value=7.0,
    value=1.4,
    step=0.1
)

petal_width = st.slider(
    "Petal width (cm)",
    min_value=0.1,
    max_value=2.5,
    value=0.2,
    step=0.1
)

if st.button("Predict FLower Speices: ",type="primary"):
    measuremnets = [[sepal_length,
                     sepal_width,petal_length,petal_width]]

    prediced_class = int(model.predict(measuremnets)[0])

    prediced_speices = target_names[prediced_class]

    probabilites = model.predict_proba(measuremnets)[0]

    st.success(
        f"Predicted species: {prediced_speices.title()}"
    )

    st.subheader("Class probabilites")

    for name, probability in zip(target_names,probabilites):
        st.write(f"**{name.title()}**:{probability:.1%}")
        st.progress(float(probability))

    st.caption("Probabilities are model estimates, not guarantees "
        "that a prediction is correct.")