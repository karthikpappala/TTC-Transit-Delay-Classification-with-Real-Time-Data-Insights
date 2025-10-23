import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and the column list
model = joblib.load('bus_delay_rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="TTC Bus Delay Predictor", layout="wide")
st.title('🚌 TTC Bus Delay Predictor')
st.write("Enter the details of a bus trip to predict whether it will be on time or delayed.")

# --- Create columns for layout ---
col1, col2 = st.columns(2)

# --- Input Fields ---
with col1:
    st.subheader("Trip Details")
    hour = st.slider('Hour of Day (24-hour format)', 0, 23, 17)
    month = st.selectbox('Month', options=list(range(1, 13)), index=9)
    day_of_week = st.selectbox('Day of the Week', options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    incident_type = st.selectbox('Incident Type', options=['Operations - Operator', 'Mechanical', 'General Delay', 'Collision - TTC', 'Cleaning - Unsanitary'])

with col2:
    st.subheader("Route & Location Context")
    route = st.text_input('Bus Route (e.g., 501, 32, 95)', '501')
    direction = st.selectbox('Direction', options=['E', 'W', 'N', 'S', 'Unknown'])
    min_gap = st.slider('Minimum Gap (minutes between buses)', 0, 200, 20)


# --- Data Processing and Prediction ---
if st.button('Predict Delay Status'):
    # Determine TimeOfDay and IsWeekend from inputs
    if 5 <= hour < 12:
        time_of_day = 'Morning'
    elif 12 <= hour < 17:
        time_of_day = 'Afternoon'
    elif 17 <= hour < 21:
        time_of_day = 'Evening'
    else:
        time_of_day = 'Night'

    is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0

    # Create a DataFrame from the inputs
    new_data_raw = pd.DataFrame({
        'Min Gap': [min_gap],
        'Hour': [hour],
        'Month': [month],
        'IsWeekend': [is_weekend],
        'AvgRouteDelay': [25.5], # Using placeholder averages, can be refined
        'AvgDirectionDelay': [22.1], # Using placeholder averages, can be refined
        'Route': [route],
        'Incident': [incident_type],
        'Direction': [direction],
        'DayOfWeek': [day_of_week],
        'TimeOfDay': [time_of_day]
    })

    # One-hot encode the new data
    new_data_encoded = pd.get_dummies(new_data_raw)

    # Align columns with the training data
    new_data_aligned = new_data_encoded.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    prediction_proba = model.predict_proba(new_data_aligned)
    prediction = model.predict(new_data_aligned)

    # --- Display Results ---
    st.subheader('Prediction Result')
    probability_delayed = prediction_proba[0][1] * 100

    if prediction[0] == 1:
        st.error(f'Prediction: **Delayed** (Probability: {probability_delayed:.2f}%)')
    else:
        st.success(f'Prediction: **On Time** (Probability: {100 - probability_delayed:.2f}%)')

    st.write("Probabilities:", {'On Time': f"{prediction_proba[0][0]:.2%}", 'Delayed': f"{prediction_proba[0][1]:.2%}"})
