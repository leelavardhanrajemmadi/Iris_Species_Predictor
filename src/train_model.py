import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# load dataset
df = pd.read_csv("../data/Iris.csv")

# features and target
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# create model
model = RandomForestClassifier()

# train model
model.fit(X_train, y_train)

# predictions
predictions = model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:")
print(accuracy)

# save model
joblib.dump(model, "../models/iris_model.pkl")

print("\nModel Saved Successfully")