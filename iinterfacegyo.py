import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained pipeline
with open('capstone3_gyo.sav', 'rb') as file:
    pipeline = pickle.load(file)

st.title('''Hotel Booking Cancellation Prediction App
            By Dhiyafath Anargyo Orion JCDSOL - 013 - 026''')

st.markdown("""
## Predicting Hotel Booking Cancellation
Use this app to predict whether a hotel booking will be canceled based on various features.
""")

# User inputs for the features
country = st.selectbox('Country', ['IRL', 'FRA', 'PRT', 'NLD', 'ESP', 'UMI', 'CN', 'LUX', 'BRA', 'BEL', 'JPN', 'DEU', 'ITA', 'CHE', 'GBR', 'AGO', 'SRB', 'COL', 'CHN', 'SWE', 'AUT', 'CIV', 'CZE', 'POL', 'USA', 'SGP', 'RUS', 'ROU', 'DNK', 'IND', 'MAR', 'PHL', 'ARG', 'ISL', 'ZAF', 'LBN', 'MOZ', 'TUR', 'BGD', 'MEX', 'CAF', 'NOR', 'FIN', 'UKR', 'EGY', 'ISR', 'nan', 'KOR', 'AZE', 'HUN', 'AUS', 'EST', 'CHL', 'SVN', 'PRY', 'ABW', 'ALB', 'LTU', 'ARE', 'HRV', 'SAU', 'NZL', 'LVA', 'ATA', 'KAZ', 'DZA', 'TWN', 'CRI', 'BIH', 'BGR', 'IRQ', 'OMN', 'VEN', 'IDN', 'GEO', 'MLT', 'IRN', 'BLR', 'URY', 'LBY', 'TUN', 'BEN', 'MYS', 'MWI', 'GRC', 'CYP', 'CPV', 'HKG', 'PRI', 'MKD', 'MUS', 'IMN', 'PAN', 'NGA', 'GLP', 'KHM', 'PER', 'QAT', 'SEN', 'MAC', '...'])
market_segment = st.selectbox('Market Segment', ['Offline TA/TO', 'Online TA', 'Direct', 'Groups', 'Corporate', 'Complementary', 'Aviation', 'Undefined'])
previous_cancellations = st.number_input('Previous Cancellations', min_value=0, step=1, value=0)
booking_changes = st.number_input('Booking Changes', min_value=0, step=1, value=0)
deposit_type = st.selectbox('Deposit Type', ['No Deposit', 'Non Refund', 'Refundable'])
days_in_waiting_list = st.number_input('Days in Waiting List', min_value=0, step=1, value=0)
customer_type = st.selectbox('Customer Type', ['Transient-Party', 'Transient', 'Contract', 'Group'])
reserved_room_type = st.selectbox('Reserved Room Type', ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'L', 'P'])
required_car_parking_spaces = st.number_input('Required Car Parking Spaces', min_value=0, step=1, value=0)
total_of_special_requests = st.number_input('Total of Special Requests', min_value=0, step=1, value=0)

# Create a DataFrame with the inputs
data = {
    'country': [country],
    'market_segment': [market_segment],
    'previous_cancellations': [previous_cancellations],
    'booking_changes': [booking_changes],
    'deposit_type': [deposit_type],
    'days_in_waiting_list': [days_in_waiting_list],
    'customer_type': [customer_type],
    'reserved_room_type': [reserved_room_type],
    'required_car_parking_spaces': [required_car_parking_spaces],
    'total_of_special_requests': [total_of_special_requests]
}

input_df = pd.DataFrame(data)

# Predict the target
if st.button('Predict Cancellation'):
    prediction = pipeline.predict(input_df)
    if prediction[0] == 1:
        st.write('The booking is likely to be canceled.')
    else:
        st.write('The booking is likely to be kept.')

# Tombol untuk reset/flush
if st.button('Reset'):
    st.experimental_rerun()