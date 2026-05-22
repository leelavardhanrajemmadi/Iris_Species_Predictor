import pandas as pd
from sklearn.model_selection import train_test_split

# load dataset
df = pd.read_csv("../data/Iris.csv")

# features
X = df.drop("Species", axis=1)

# target
y = df["Species"]

# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)