import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Title and Description
st.write("""
# HDB Resale Price Prediction App
This app predicts the **HDB resale price** based on user inputs!
""")

# Sidebar for User Input
st.sidebar.header('User Input Parameters')

# Function to capture user input
def user_input_features():
    flat_type = st.sidebar.selectbox(
        'Flat Type',
        ['1-room', '2-room', '3-room', '4-room', '5-room', 'Executive', 'Multi-Generational']
    )
    floor_area = st.sidebar.slider('Floor Area (sqm)', 30, 200, 90)
    lease_remaining = st.sidebar.slider('Remaining Lease (years)', 0, 99, 75)
    town = st.sidebar.selectbox(
        'Town',
        [
            'Ang Mo Kio', 'Bedok', 'Clementi', 'Jurong West', 'Tampines', 'Woodlands',
            'Bukit Batok', 'Hougang', 'Pasir Ris', 'Queenstown', 'Sembawang', 
            'Sengkang', 'Toa Payoh', 'Yishun'
        ]
    )
    # Add any other columns for user input
    storey_range = st.sidebar.selectbox(
        'Storey Range',
        ['1-3', '4-6', '7-9', '10-12', '13-15', '16-18', '19-21', '22-24', '25-27', '28-30', '31 and above']
    )
    distance_to_mrt = st.sidebar.slider('Distance to Nearest MRT (km)', 0.1, 5.0, 1.0)

    # DataFrame for inputs
    data = {
        'flat_type': flat_type,
        'floor_area': floor_area,
        'lease_remaining': lease_remaining,
        'town': town,
        'storey_range': storey_range,
        'distance_to_mrt': distance_to_mrt
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Collect user inputs
df = user_input_features()

# Display User Inputs
st.subheader('User Input Parameters')
st.write(df)

# Load Pre-trained Model and Encoders
try:
    clf = joblib.load('hdb_price_predictor.pkl')  # Load your trained model
    le_flat_type = joblib.load('le_flat_type.pkl')  # LabelEncoder for flat_type
    le_town = joblib.load('le_town.pkl')  # LabelEncoder for town
    le_storey_range = joblib.load('le_storey_range.pkl')  # LabelEncoder for storey_range
except FileNotFoundError as e:
    st.error("Required files not found. Please ensure the model and encoders are available.")
    st.stop()

# Encode user inputs
try:
    df['flat_type'] = le_flat_type.transform(df['flat_type'])
    df['town'] = le_town.transform(df['town'])
    df['storey_range'] = le_storey_range.transform(df['storey_range'])
except ValueError as e:
    st.error("Error encoding user inputs. Please ensure values match the trained dataset.")
    st.stop()

# Make predictions
prediction = clf.predict(df)

# Display results
st.subheader('Prediction')
st.write(f"Estimated Resale Price: **${prediction[0]:,.2f}**")