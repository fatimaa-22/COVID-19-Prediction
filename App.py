import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")
st.title("🌍 COVID-19 Country-Wise Dashboard & Prediction")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("country_wise_latest.csv")

data = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("📌 Options")
selected_country = st.sidebar.selectbox(
    "Select Country",
    data["Country/Region"].unique()
)

country_data = data[data["Country/Region"] == selected_country]

# ---------------- METRICS ----------------
st.subheader(f"📊 COVID-19 Statistics for {selected_country}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Confirmed Cases", int(country_data["Confirmed"].values[0]))
col2.metric("Total Deaths", int(country_data["Deaths"].values[0]))
col3.metric("Total Recovered", int(country_data["Recovered"].values[0]))
col4.metric("Active Cases", int(country_data["Active"].values[0]))

# ---------------- VISUALIZATION ----------------
st.subheader("📈 Confirmed vs Deaths")

plt.figure()
sns.scatterplot(x="Confirmed", y="Deaths", data=data)
st.pyplot(plt)

# ---------------- MACHINE LEARNING + PICKLE ----------------
model_file = "covid_model.pkl"

X = data[['Confirmed', 'Recovered', 'Active', 'New cases', 'New deaths']]
y = data['Deaths']

# Train model only once
if not os.path.exists(model_file):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Save model using pickle
    with open(model_file, "wb") as file:
        pickle.dump(model, file)

else:
    # Load saved model
    with open(model_file, "rb") as file:
        model = pickle.load(file)

# ---------------- PREDICTION SECTION ----------------
st.subheader("🤖 Predict Deaths")

confirmed = st.number_input("Confirmed Cases", min_value=0)
recovered = st.number_input("Recovered Cases", min_value=0)
active = st.number_input("Active Cases", min_value=0)
new_cases = st.number_input("New Cases", min_value=0)
new_deaths = st.number_input("New Deaths", min_value=0)

if st.button("Predict"):
    input_data = [[confirmed, recovered, active, new_cases, new_deaths]]
    prediction = model.predict(input_data)
    st.success(f"🧮 Predicted Deaths: {int(prediction[0])}")
