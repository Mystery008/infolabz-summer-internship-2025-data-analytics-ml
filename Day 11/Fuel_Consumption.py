import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

st.set_page_config(page_title="Fuel Consumption Predictor", layout="centered")
st.title("🚗 Fuel Consumption Predictor")
st.markdown("""
This app predicts:
- **Fuel Consumption (City)**
- **Fuel Consumption (Highway)**
- **Combined Fuel Consumption**

based on vehicle specifications.
""")

# Load data
data = pd.read_csv("clean_fuel.csv")

# Encode categorical features
class_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()

data['VEHICLE CLASS'] = class_encoder.fit_transform(data['VEHICLE CLASS'])
data['TRANSMISSION'] = transmission_encoder.fit_transform(data['TRANSMISSION'])
data['FUEL'] = fuel_encoder.fit_transform(data['FUEL'])

# Features and targets
features = ["VEHICLE CLASS", "ENGINE SIZE", "CYLINDERS", "TRANSMISSION", "FUEL", "EMISSIONS"]
target = ["FUEL CONSUMPTION", "HWY (L/100 km)", "COMB (L/100 km)"]
X = data[features]
y = data[target]

# Train model only once if not already trained
@st.cache_resource
def train_model():
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Streamlit inputs
with st.form("prediction_form"):
    vehicle_class = st.selectbox("Select Vehicle Class", class_encoder.classes_)
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, step=0.1)
    cylinders = st.number_input("Cylinders", min_value=2, max_value=16, step=1)
    transmission = st.selectbox("Select Transmission", transmission_encoder.classes_)
    fuel = st.selectbox("Select Fuel Type", fuel_encoder.classes_)
    emissions = st.number_input("Enter Emissions (g/km)", min_value=50, max_value=700)
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_data = pd.DataFrame({
            "VEHICLE CLASS": [class_encoder.transform([vehicle_class])[0]],
            "ENGINE SIZE": [engine_size],
            "CYLINDERS": [cylinders],
            "TRANSMISSION": [transmission_encoder.transform([transmission])[0]],
            "FUEL": [fuel_encoder.transform([fuel])[0]],
            "EMISSIONS": [emissions]
        })

        predictions = model.predict(input_data)[0]

        st.success("Predicted Fuel Metrics:")
        st.write(f"**Fuel Consumption (City)**: {predictions[0]:.2f} L/100km")
        st.write(f"**Fuel Consumption (Highway)**: {predictions[1]:.2f} L/100km")
        st.write(f"**Combined Fuel Consumption**: {predictions[2]:.2f} L/100km")

    except ValueError as e:
        st.error(f"Encoding error: {e}")
