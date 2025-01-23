import streamlit as st
import pandas as pd
import joblib

# App title with styling and HDB image
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display image of HDB flats
st.image("hdb_picture.jpg", caption="HDB Flats in Singapore", use_container_width=True)

# Title and subtitle
st.markdown(
    """
    <div class="title">HDB Resale Price Prediction</div>
    <div class="subtitle">Enter the flat details below to predict the resale price</div>
    """,
    unsafe_allow_html=True
)

# User Input section
def user_inputs():
    st.sidebar.header("Input Details")
    floor_area_sqm = st.sidebar.slider("Floor Area (sqm)", 0, 500, 90)
    remaining_lease_year = st.sidebar.slider("Remaining Lease (years)", 0, 99, 75)
    town = st.sidebar.selectbox(
        "Town",
        ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 
         'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 
         'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 
         'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN']
    )
    flat_type = st.sidebar.selectbox(
        "Flat Type",
        ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
    )
    storey_range = st.sidebar.selectbox(
        "Storey Range",
        ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', 
         '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']
    )
    flat_model = st.sidebar.selectbox(
        "Flat Model",
        ['2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved', 'Improved-Maisonette', 'Maisonette', 
         'Model A', 'Model A-Maisonette', 'Model A2', 'Multi Generation', 'New Generation', 'Premium Apartment', 
         'Premium Apartment Loft', 'Premium Maisonette', 'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2']
    )
    return floor_area_sqm, remaining_lease_year, town, flat_type, storey_range, flat_model

# Collect user inputs
floor_area_sqm, remaining_lease_year, town, flat_type, storey_range, flat_model = user_inputs()

# Display input data summary
st.markdown("### Your Flat Specifications")
st.table({
    "Specification": ["Floor Area (sqm)", "Remaining Lease (years)", "Town", "Flat Type", "Storey Range", "Flat Model"],
    "Value": [floor_area_sqm, remaining_lease_year, town, flat_type, storey_range, flat_model]
})

# Define all possible one-hot encoded columns
all_columns = (
    ['floor_area_sqm', 'remaining_lease_year'] +
    [f'town_{t}' for t in ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH', 
                           'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 
                           'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 
                           'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN']] +
    [f'flat_type_{ft}' for ft in ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']] +
    [f'storey_range_{sr}' for sr in ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', 
                                     '22 TO 24', '25 TO 27', '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', 
                                     '43 TO 45', '46 TO 48', '49 TO 51']] +
    [f'flat_model_{fm}' for fm in ['2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved', 'Improved-Maisonette', 
                                   'Maisonette', 'Model A', 'Model A-Maisonette', 'Model A2', 'Multi Generation', 
                                   'New Generation', 'Premium Apartment', 'Premium Apartment Loft', 'Premium Maisonette', 
                                   'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2']]
)

# Prepare input data for prediction
data = {
    'floor_area_sqm': [floor_area_sqm],
    'remaining_lease_year': [remaining_lease_year],
    f'town_{town}': [1],
    f'flat_type_{flat_type}': [1],
    f'storey_range_{storey_range}': [1],
    f'flat_model_{flat_model}': [1]
}
df = pd.DataFrame(data)

# Ensure all columns exist
for col in all_columns:
    if col not in df:
        df[col] = 0  # Add missing columns with default 0

# Load Pre-Trained Model
try:
    clf = joblib.load('hdb price predictor.pkl')  # Load your trained model
except FileNotFoundError as e:
    st.error("Model file not found. Please ensure 'hdb price predictor.pkl' is available.")
    st.stop()

# Prediction section
if st.button("🏠 Predict Resale Price 🏠"):
    with st.spinner("Calculating price..."):
        try:
            prediction = clf.predict(df)
            st.success(f"The predicted resale price for the HDB flat is: **${prediction[0]:,.2f}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")