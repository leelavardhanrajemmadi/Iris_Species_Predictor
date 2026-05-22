import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("../data/titanic.csv")

# basic info
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# visualization
sns.countplot(x=df["Survived"])

plt.title("Survival Count")

plt.show()