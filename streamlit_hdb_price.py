import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
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
    floor_area_sqm = st.sidebar.slider('Floor Area (sqm)', 30, 400, 90)
    remaining_lease_year = st.sidebar.slider('Remaining Lease (years)', 0, 99, 75)
    town = st.sidebar.selectbox(
        'Town',
        [
            'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'YISHUN', 'WOODLANDS'
        ]
    )
    flat_type = st.sidebar.selectbox(
        'Flat Type',
        ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATIONAL']
    )
    storey_range = st.sidebar.selectbox(
        'Storey Range',
        ['1 TO 3', '4 TO 6', '7 TO 9', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']
    )
    flat_model = st.sidebar.selectbox(
        'Flat Model',
        ['2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved', 'Improved-Maisonette', 'Maisonette', 'Model A', 'Model A-Maisonette', 'Model A2', 'Multi Generation', 'New Generation', 'Premium Apartment', 'Premium Apartment Loft', 'Premium Maisonette', 'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2']
    )
        

    # DataFrame for inputs
    data = {
        'floor_area_sqm': floor_area_sqm,
        'remaining_lease_year': remaining_lease_year,
        'town': town,
        'flat_type': flat_type,
        'storey_range': storey_range,
        'flat_model': flat_model
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
    clf = joblib.load('hdb price predictor.pkl')  # Load your trained model
except FileNotFoundError as e:
    st.error("Required files not found. Please ensure the model and encoders are available.")
    st.stop()

# Make predictions
prediction = clf.predict(df)

# Display results
st.subheader('Prediction')
st.write(f"Estimated Resale Price: **${prediction[0]:,.2f}**")