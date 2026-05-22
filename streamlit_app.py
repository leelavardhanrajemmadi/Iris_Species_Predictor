import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# page title
st.title("Titanic Analytics Dashboard")

# load dataset
df = pd.read_csv("data/titanic.csv")

# sidebar
st.sidebar.header("Filters")

gender = st.sidebar.selectbox(
    "Select Gender",
    ["All"] + list(df["Sex"].unique())
)

pclass = st.sidebar.selectbox(
    "Select Passenger Class",
    ["All"] + sorted(df["Pclass"].unique())
)

# filtering
filtered_df = df.copy()

if gender != "All":
    filtered_df = filtered_df[
        filtered_df["Sex"] == gender
    ]

if pclass != "All":
    filtered_df = filtered_df[
        filtered_df["Pclass"] == pclass
    ]

# dataset preview
st.subheader("Dataset Preview")

st.dataframe(filtered_df.head())

# dataset shape
st.subheader("Dataset Shape")

st.write(filtered_df.shape)

# survival count
st.subheader("Survival Count")

fig1, ax1 = plt.subplots()

a=sns.countplot(
    x=filtered_df["Survived"],
    ax=ax1
)
for i in a.containers:
    a.bar_label(i)
st.pyplot(fig1)

# gender survival comparison
st.subheader("Gender vs Survival")

fig2, ax2 = plt.subplots()

a=sns.countplot(
    x=filtered_df["Sex"],
    hue=filtered_df["Survived"],
    ax=ax2
)
for i in a.containers:
    a.bar_label(i)
    

st.pyplot(fig2)

# age distribution
st.subheader("Age Distribution")

fig3, ax3 = plt.subplots()

sns.histplot(
    filtered_df["Age"].dropna(),
    kde=True,
    ax=ax3
)

st.pyplot(fig3)

# passenger class distribution
st.subheader("Passenger Class Distribution")

fig4, ax4 = plt.subplots()

sns.countplot(
    x=filtered_df["Pclass"],
    ax=ax4
)

st.pyplot(fig4)