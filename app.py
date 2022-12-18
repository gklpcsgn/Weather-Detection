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
    TempHighF = st.sidebar.slider('Highest Temprature (Fahrenheit) ', 20.0, 110.0, 80.0,step=0.01)
    TempAvgF = st.sidebar.slider('Average Temprature (Fahrenheit)', 20.0,110.0 , 70.0,step=0.01)
    TempLowF = st.sidebar.slider('Lowest Temprature (Fahrenheit)', 20.0, 110.0, 60.0,step=0.01)
    DewPointHighF = st.sidebar.slider('Highest Dew Point (Fahrenheit)', 0.0, 80.0, 60.0,step=0.01)
    DewPointAvgF = st.sidebar.slider('Average Dew Point (Fahrenheit)', 0.0, 80.0, 55.0,step=0.01)
    DewPointLowF = st.sidebar.slider('Lowest Dew Point (Fahrenheit)', 0.0, 80.0, 50.0,step=0.01)
    HumidityHighPercent = st.sidebar.slider('Highest Humidity (Percent)', 0.0, 100.0, 85.0,step=0.01)
    HumidityAvgPercent = st.sidebar.slider('Average Humidity (Percent)', 0.0, 100.0, 65.0,step=0.01)
    HumidityLowPercent = st.sidebar.slider('Lowest Humidity (Percent)', 0.0, 100.0, 45.0,step=0.01)
    SeaLevelPressureHighInches = st.sidebar.slider('Highest Sea Level Pressure (Inches)', 29.5, 31.0, 30.0,step=0.01)
    SeaLevelPressureAvgInches = st.sidebar.slider('Average Sea Level Pressure (Inches)', 29.5, 31.0, 30.0,step=0.01)
    SeaLevelPressureLowInches = st.sidebar.slider('Lowest Sea Level Pressure (Inches)', 29.5, 31.0, 30.0,step=0.01)
    VisibilityHighMiles = st.sidebar.slider('Highest Visibility (Miles)', 0.0, 10.0, 9.0,step=0.01)
    VisibilityAvgMiles = st.sidebar.slider('Average Visibility (Miles)', 0.0, 10.0, 9.0,step=0.01)
    VisibilityLowMiles = st.sidebar.slider('Lowest Visibility (Miles)', 0.0, 10.0, 7.0,step=0.01)
    WindHighMPH = st.sidebar.slider('Highest Wind (MPH)', 0.0, 30.0, 13.0,step=0.01)
    WindAvgMPH = st.sidebar.slider('Average Wind (MPH)', 0.0, 30.0, 5.0,step=0.01)
    WindGustMPH = st.sidebar.slider('Wind Gust (MPH)', 0.0, 60.0, 20.0,step=0.01)
    PrecipitationSumInches = st.sidebar.slider('Precipitation Sum (Inches)', 0.0, 10.0, 1.0,step=0.01)
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

# create and prettify the output

rain = model_gbc_rain.predict(input_df)
fog = model_gbc_fog.predict(input_df)
thunderstorm = logmodel_thunderstorm.predict(input_df)

# get probabilities
rain_prob = model_gbc_rain.predict_proba(input_df)
fog_prob = model_gbc_fog.predict_proba(input_df)
thunderstorm_prob = logmodel_thunderstorm.predict_proba(input_df)

# create a data frame for both the prediction and the probabilities
output_df = pd.DataFrame({'Rain': rain,
                            'Fog': fog,
                            'Thunderstorm': thunderstorm})


output_df = output_df.replace({0: 'No', 1: 'Yes'})

# format the probabilities to 2 decimal places
rain_prob = np.round(rain_prob, 2)
fog_prob = np.round(fog_prob, 2)
thunderstorm_prob = np.round(thunderstorm_prob, 2)

output_df_prob = pd.DataFrame({'Rain': rain_prob[:,1],
                            'Fog': fog_prob[:,1],
                            'Thunderstorm': thunderstorm_prob[:,1]})

# concatenate the both data frames
output_df = pd.concat([output_df, output_df_prob], axis=0)

output_df.set_index([pd.Index(['Prediction', 'Probability'])], inplace=True)

# prettify the output


# display the output
st.write('The weather will be:')
st.write(output_df)
