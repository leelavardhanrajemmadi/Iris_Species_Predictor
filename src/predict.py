import joblib
import pandas as pd

# load trained model
model = joblib.load("../models/iris_model.pkl")

# sample input
sample_data = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
)

# prediction
prediction = model.predict(sample_data)

print("Predicted Species:")
print(prediction[0])