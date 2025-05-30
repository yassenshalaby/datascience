import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import requests
from io import BytesIO

# === Load Data and Model from GitHub ===
data_url = "https://raw.githubusercontent.com/yassenshalaby/datascience/780dfd0b3059aef540f54b8b3351f7ad7710297e/climate_change_dataset.csv"
model_url = "https://raw.githubusercontent.com/yassenshalaby/datascience/780dfd0b3059aef540f54b8b3351f7ad7710297e/co2_predictor_model.joblib"

@st.cache_data
def load_data():
    return pd.read_csv(data_url).dropna()

data = load_data()
model = load(BytesIO(requests.get(model_url).content))

# === Feature Engineering for data ===
data = data.sort_values(["Year"])

# Create new features
data["CO2 per Million People"] = data["CO2 Emissions (Tons/Capita)"] * 1e6 / data["Population"]
data["Temp Change YoY"] = data.sort_values(["Country", "Year"]).groupby("Country")["Avg Temperature (°C)"].diff()
data["Forest per Person"] = data["Forest Area (%)"] / data["Population"]
data["Log Population"] = np.log1p(data["Population"])
data["Temp^2"] = data["Avg Temperature (°C)"] ** 2
data["Rainfall^2"] = data["Rainfall (mm)"] ** 2

# Min-Max Scaling
columns_to_scale = [
    "CO2 per Million People", "Temp Change YoY",
    "Forest per Person", "Log Population",
    "Temp^2", "Rainfall^2"
]

for col in columns_to_scale:
    min_val = data[col].min()
    max_val = data[col].max()
    data[col + "_scaled"] = (data[col] - min_val) / (max_val - min_val)

# === Streamlit Interface ===
st.title("Climate Change Analysis and CO2 Prediction")
st.markdown("### Research Questions Insights")

# === Q1: Avg Temp Over Time ===
st.subheader("Q1: Average Global Temperature Over Time")
q1 = data.groupby("Year")["Avg Temperature (°C)"].mean().reset_index()
fig1, ax1 = plt.subplots()
sns.lineplot(data=q1, x="Year", y="Avg Temperature (°C)", ax=ax1)
ax1.grid(True)
st.pyplot(fig1)

# === Q2: Top CO2 Emitters ===
st.subheader("Q2: Top 10 Countries by Average CO2 Emissions")
q2 = data.groupby("Country")["CO2 Emissions (Tons/Capita)"].mean().sort_values(ascending=False).head(10).reset_index()
fig2, ax2 = plt.subplots()
sns.barplot(data=q2, x="CO2 Emissions (Tons/Capita)", y="Country", ax=ax2)
ax2.grid(True)
st.pyplot(fig2)

# === Q3: Renewable Energy vs CO2 ===
st.subheader("Q3: Renewable Energy vs CO2 Emissions")
fig3, ax3 = plt.subplots()
sns.scatterplot(data=data, x="Renewable Energy (%)", y="CO2 Emissions (Tons/Capita)", ax=ax3)
ax3.grid(True)
st.pyplot(fig3)

# === Q4: Forest Area Trend ===
st.subheader("Q4: Forest Area (%) Over Time")
q4 = data.groupby("Year")["Forest Area (%)"].mean().reset_index()
fig4, ax4 = plt.subplots()
sns.lineplot(data=q4, x="Year", y="Forest Area (%)", ax=ax4)
ax4.grid(True)
st.pyplot(fig4)

# === Q5: Extreme Weather & Sea Level ===
st.subheader("Q5: Extreme Weather Events vs Sea Level Rise")
q5 = data.groupby("Year")[["Extreme Weather Events", "Sea Level Rise (mm)"]].sum().reset_index()
fig5, ax5 = plt.subplots()
sns.lineplot(data=q5, x="Year", y="Extreme Weather Events", label="Extreme Weather", ax=ax5)
sns.lineplot(data=q5, x="Year", y="Sea Level Rise (mm)", label="Sea Level Rise", ax=ax5)
ax5.grid(True)
ax5.legend()
st.pyplot(fig5)

# === Prediction Section ===
st.markdown("### Predict CO2 Emissions")
user_input = {}
fields = ["Avg Temperature (°C)", "Sea Level Rise (mm)", "Rainfall (mm)", "Population",
          "Renewable Energy (%)", "Extreme Weather Events", "Forest Area (%)"]
for field in fields:
    user_input[field] = st.number_input(field, value=0.0)

if st.button("Predict CO2 Emissions"):
    input_df = pd.DataFrame([user_input])

    # Recreate derived features for prediction
    input_df["CO2 per Million People"] = 0
    input_df["Temp Change YoY"] = 0
    input_df["Forest per Person"] = input_df["Forest Area (%)"] / input_df["Population"]
    input_df["Log Population"] = np.log1p(input_df["Population"])
    input_df["Temp^2"] = input_df["Avg Temperature (°C)"] ** 2
    input_df["Rainfall^2"] = input_df["Rainfall (mm)"] ** 2

    # Apply scaling
    def minmax(col, ref):
        return (col - ref.min()) / (ref.max() - ref.min())

    input_df["CO2 per Million People_scaled"] = 0
    input_df["Temp Change YoY_scaled"] = 0
    input_df["Forest per Person_scaled"] = minmax(input_df["Forest per Person"], data["Forest per Person"])
    input_df["Log Population_scaled"] = minmax(input_df["Log Population"], data["Log Population"])
    input_df["Temp^2_scaled"] = minmax(input_df["Temp^2"], data["Temp^2"])
    input_df["Rainfall^2_scaled"] = minmax(input_df["Rainfall^2"], data["Rainfall^2"])

    # Select final features
    final_features = [
        "Avg Temperature (°C)", "Sea Level Rise (mm)", "Rainfall (mm)", "Population",
        "Renewable Energy (%)", "Extreme Weather Events", "Forest Area (%)",
        "CO2 per Million People_scaled", "Temp Change YoY_scaled",
        "Forest per Person_scaled", "Log Population_scaled",
        "Temp^2_scaled", "Rainfall^2_scaled"
    ]
    input_df = input_df[final_features]

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted CO2 Emissions: {prediction:.2f} Tons/Capita")
