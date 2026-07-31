import streamlit as st
import joblib
import numpy as np

# Page Settings
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

# Dark Theme
st.markdown("""
<style>
.stApp{
    background-color:#0E1117;
}
h1{
    color:white;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# Load Model
model = joblib.load("best_model.pkl")

st.title("🌸 Iris Flower Prediction")

sl = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.10,
    step=0.1
)

sw = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.50,
    step=0.1
)

pl = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.40,
    step=0.1
)

pw = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.20,
    step=0.1
)

if st.button("Predict"):

    sample = np.array([[sl, sw, pl, pw]])

    prediction = model.predict(sample)[0]

    st.success(f"Predicted Flower: {prediction}")