# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress the InconsistentVersionWarning that appears when unpickling
warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator.*")

# --- Configuration for Streamlit App ---
st.set_page_config(page_title="Forest Fire Area Prediction App")
st.title("Forest Fire Area Prediction")
st.markdown("Use the sidebar to input parameters and predict the burned area.")

# --- Load required models and scaler ---
# It is assumed that 'rf_model.pkl' and 'scaler.pkl' are in the same directory.
try:
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Error: Model files 'rf_model.pkl' or 'scaler.pkl' not found.")
    st.info("Please run 'forestfire.py' first to train and save the model.")
    st.stop()

# --- Define mappings for categorical variables ---
# These mappings should be consistent with the LabelEncoder used in forestfire.py
month_map = {
    'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
    'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11
}
day_map = {
    'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6
}

# --- Sidebar inputs for user interaction ---
st.sidebar.header("Input Parameters")

def get_user_input():
    """
    Collects user input from the sidebar and returns a DataFrame.
    """
    # These spatial features were part of the original training data and must be
    # included for the scaler and model to work correctly.
    X = st.sidebar.slider('X (Spatial coordinate)', 1, 9, 5)
    Y = st.sidebar.slider('Y (Spatial coordinate)', 2, 9, 5)

    month = st.sidebar.selectbox('Month', options=list(month_map.keys()))
    day = st.sidebar.selectbox('Day', options=list(day_map.keys()))
    ffmc = st.sidebar.slider('FFMC (Fine Fuel Moisture Code)', 18.7, 96.2, 90.0)
    dmc = st.sidebar.slider('DMC (Duff Moisture Code)', 1.1, 291.3, 100.0)
    dc = st.sidebar.slider('DC (Drought Code)', 7.9, 860.6, 500.0)
    isi = st.sidebar.slider('ISI (Initial Spread Index)', 0.0, 56.1, 10.0)
    temp = st.sidebar.slider('Temperature ($$^\circ C$$)', 2.2, 33.3, 20.0)
    rh = st.sidebar.slider('Relative Humidity (%)', 15.0, 100.0, 50.0)
    wind = st.sidebar.slider('Wind Speed (km/h)', 0.4, 9.4, 4.0)
    rain = st.sidebar.slider('Rain (mm)', 0.0, 6.4, 0.0)

    # Convert month and day to their numeric values
    month_encoded = month_map[month]
    day_encoded = day_map[day]

    # Create a DataFrame with all the features in the same order as the training data
    data = {
        'X': X, 'Y': Y, 'month': month_encoded, 'day': day_encoded,
        'FFMC': ffmc, 'DMC': dmc, 'DC': dc, 'ISI': isi, 'temp': temp, 'RH': rh,
        'wind': wind, 'rain': rain
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = get_user_input()

# Display user input
st.subheader("User Input Parameters")
st.write(input_df)

# Make prediction and display result
if st.sidebar.button("Predict Forest Fire Area"):
    # The scaler was fitted on a DataFrame that included 'X' and 'Y'.
    # We must ensure the input DataFrame has the same columns.
    
    # Identify numerical columns to scale, including 'X' and 'Y'
    numerical_cols = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
    
    # Scale the numerical input features
    scaled_input_df = input_df.copy()
    scaled_input_df[numerical_cols] = scaler.transform(scaled_input_df[numerical_cols])

    prediction_log = model.predict(scaled_input_df)

    # Inverse transform the prediction to get the original scale
    prediction_area = np.expm1(prediction_log)

    st.subheader("Prediction Result")
    st.success(f"**Predicted Burned Area:** {prediction_area[0]:.2f} ha")
    st.info("Note: The prediction is an estimate based on the provided model.")
