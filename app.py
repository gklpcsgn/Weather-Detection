import pandas as pd
import streamlit as st
import numpy as np
import pickle

# load the model from disk
scaler = pickle.load(open('Preprocessing/all_scaler.pkl', 'rb'))
model_gbc_rain = pickle.load(open('Models/best_rain.pkl', 'rb'))
model_gbc_fog = pickle.load(open('Models/best_fog.pkl', 'rb'))
logmodel_thunderstorm = pickle.load(open('Models/best_thunderstorm.pkl', 'rb'))

columns = ['TempHighF', 'TempAvgF', 'TempLowF', 'DewPointHighF', 'DewPointAvgF',
       'DewPointLowF', 'HumidityHighPercent', 'HumidityAvgPercent',
       'HumidityLowPercent', 'SeaLevelPressureHighInches',
       'SeaLevelPressureAvgInches', 'SeaLevelPressureLowInches',
       'VisibilityHighMiles', 'VisibilityAvgMiles', 'VisibilityLowMiles',
       'WindHighMPH', 'WindAvgMPH', 'WindGustMPH', 'PrecipitationSumInches']

def user_input_features():
    TempHighF = st.sidebar.slider('TempHighF', 0, 100, 50)
    TempAvgF = st.sidebar.slider('TempAvgF', 0, 100, 50)
    TempLowF = st.sidebar.slider('TempLowF', 0, 100, 50)
    DewPointHighF = st.sidebar.slider('DewPointHighF', 0, 100, 50)
    DewPointAvgF = st.sidebar.slider('DewPointAvgF', 0, 100, 50)
    DewPointLowF = st.sidebar.slider('DewPointLowF', 0, 100, 50)
    HumidityHighPercent = st.sidebar.slider('HumidityHighPercent', 0, 100, 50)
    HumidityAvgPercent = st.sidebar.slider('HumidityAvgPercent', 0, 100, 50)
    HumidityLowPercent = st.sidebar.slider('HumidityLowPercent', 0, 100, 50)
    SeaLevelPressureHighInches = st.sidebar.slider('SeaLevelPressureHighInches', 0, 100, 50)
    SeaLevelPressureAvgInches = st.sidebar.slider('SeaLevelPressureAvgInches', 0, 100, 50)
    SeaLevelPressureLowInches = st.sidebar.slider('SeaLevelPressureLowInches', 0, 100, 50)
    VisibilityHighMiles = st.sidebar.slider('VisibilityHighMiles', 0, 100, 50)
    VisibilityAvgMiles = st.sidebar.slider('VisibilityAvgMiles', 0, 100, 50)
    VisibilityLowMiles = st.sidebar.slider('VisibilityLowMiles', 0, 100, 50)
    WindHighMPH = st.sidebar.slider('WindHighMPH', 0, 100, 50)
    WindAvgMPH = st.sidebar.slider('WindAvgMPH', 0, 100, 50)
    WindGustMPH = st.sidebar.slider('WindGustMPH', 0, 100, 50)
    PrecipitationSumInches = st.sidebar.slider('PrecipitationSumInches', 0, 100, 50)
    data = {'TempHighF': TempHighF,
            'TempAvgF': TempAvgF,
            'TempLowF': TempLowF,
            'DewPointHighF': DewPointHighF,
            'DewPointAvgF': DewPointAvgF,
            'DewPointLowF': DewPointLowF,
            'HumidityHighPercent': HumidityHighPercent,
            'HumidityAvgPercent': HumidityAvgPercent,
            'HumidityLowPercent': HumidityLowPercent,
            'SeaLevelPressureHighInches': SeaLevelPressureHighInches,
            'SeaLevelPressureAvgInches': SeaLevelPressureAvgInches,
            'SeaLevelPressureLowInches': SeaLevelPressureLowInches,
            'VisibilityHighMiles': VisibilityHighMiles,
            'VisibilityAvgMiles': VisibilityAvgMiles,
            'VisibilityLowMiles': VisibilityLowMiles,
            'WindHighMPH': WindHighMPH,
            'WindAvgMPH': WindAvgMPH,
            'WindGustMPH': WindGustMPH,
            'PrecipitationSumInches': PrecipitationSumInches}
    features = pd.DataFrame(data, index=[0])
    return features

# set a title and a subheader
st.title('Weather Prediction App')
st.subheader('Please enter the following parameters')

# store the user input into a variable
input_df = user_input_features()

# transform the user input
input_df = scaler.transform(input_df)

# set a subheader and display the user input
st.subheader('User Input parameters')
st.write(input_df)

# create and prettify the output
st.subheader('Prediction')
st.write('Rain: ', model_gbc_rain.predict(input_df))
st.write('Fog: ', model_gbc_fog.predict(input_df))
st.write('Thunderstorm: ', logmodel_thunderstorm.predict(input_df))

# create and prettify the output
st.subheader('Prediction Probability')
st.write('Rain: ', model_gbc_rain.predict_proba(input_df))
st.write('Fog: ', model_gbc_fog.predict_proba(input_df))
st.write('Thunderstorm: ', logmodel_thunderstorm.predict_proba(input_df))

# hwo to run the app
# streamlit run myapp.py
