import streamlit as st
import pandas as pd
import joblib

# load model
model = joblib.load("models/iris_model.pkl")

# title
st.title("Iris Species Prediction for 513")

st.write("Enter flower measurements")

# inputs
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# prediction
if st.button("Predict"):

    sample_data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ]
    )

    prediction = model.predict(sample_data)

    st.success(
        f"Predicted Species: {prediction[0]}"
    )