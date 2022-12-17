import pandas as pd
import streamlit as st
import numpy as np

def preprocess_for_selection(df):
    df["PrecipitationSumInches"] = df["PrecipitationSumInches"].replace("T", 0.00)
    df.replace('-', np.nan, inplace=True)
    cols = df.columns.drop(['Date', 'Events'])
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    
    nan_cols = df.columns.drop(['Date', 'Events', 'Day', 'Month', 'Year','TempHighF', 'TempAvgF', 'TempLowF','PrecipitationSumInches'])
    
    for col in nan_cols:
        df[col] = df.groupby('Month')[col].transform(lambda x: x.fillna(x.mean()))
    df.drop(['Day','Month', 'Year'], axis=1, inplace=True)
    
    df["Rain"] = df["Events"].apply(lambda x: 1 if "Rain" in str(x) else 0)
    df["Fog"] = df["Events"].apply(lambda x: 1 if "Fog" in str(x) else 0)
    df["Thunderstorm"] = df["Events"].apply(lambda x: 1 if "Thunderstorm" in str(x) else 0)

    df.drop("Events", axis=1, inplace=True)
    df.drop("Date", axis=1, inplace=True)

    # Train test split
    from sklearn.model_selection import train_test_split
    
    X = df.drop(["Rain","Fog","Thunderstorm"], axis=1)
    y = df[["Rain","Fog","Thunderstorm"]]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=101)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=101)
    

    from sklearn.preprocessing import MinMaxScaler 
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    X_train = pd.DataFrame(X_train, columns=df.columns.drop(["Rain","Fog","Thunderstorm"]))
    X_val = pd.DataFrame(X_val, columns=df.columns.drop(["Rain","Fog","Thunderstorm"]))
    y_train = pd.DataFrame(y_train, columns=["Rain","Fog","Thunderstorm"])
    y_val = pd.DataFrame(y_val, columns=["Rain","Fog","Thunderstorm"])

    return  X_train, X_test, y_train, y_test, X_val, y_val

from sklearn.linear_model   import   LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv('austin_weather.csv')

X_train, X_test, y_train, y_test, X_val, y_val = preprocess_for_selection(data)

X = X_train
y = y_train

# we will make a multilabel feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# select the best 5 features
bestfeatures = SelectKBest(score_func=chi2, k=6)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns


model_gbc_rain = GradientBoostingClassifier(n_estimators=150, random_state=101)
model_gbc_fog = GradientBoostingClassifier(n_estimators=150, random_state=101)
logmodel_thunderstorm = LogisticRegression()

# fit the model with the selected features
model_gbc_rain.fit(X_train[featureScores.nlargest(6,'Score')['Specs'].values], y_train['Rain'])
model_gbc_fog.fit(X_train[featureScores.nlargest(6,'Score')['Specs'].values], y_train['Fog'])
logmodel_thunderstorm.fit(X_train[featureScores.nlargest(6,'Score')['Specs'].values], y_train['Thunderstorm'])

# predict the validation set
y_pred_gbc_rain = model_gbc_rain.predict(X_val[featureScores.nlargest(6,'Score')['Specs'].values])
y_pred_gbc_fog = model_gbc_fog.predict(X_val[featureScores.nlargest(6,'Score')['Specs'].values])
y_pred_logmodel_thunderstorm = logmodel_thunderstorm.predict(X_val[featureScores.nlargest(6,'Score')['Specs'].values])

# define a user input function
def user_input_features():
# (['VisibilityLowMiles', 'PrecipitationSumInches',
    #    'HumidityLowPercent', 'HumidityAvgPercent', 'VisibilityAvgMiles','WindHighMPH'],
    #   dtype='object')
    VisibilityLowMiles = st.sidebar.slider('VisibilityLowMiles', 0.0, 10.0, 5.0)
    PrecipitationSumInches = st.sidebar.slider('PrecipitationSumInches', 0.0, 10.0, 5.0)
    HumidityLowPercent = st.sidebar.slider('HumidityLowPercent', 0.0, 100.0, 50.0)
    HumidityAvgPercent = st.sidebar.slider('HumidityAvgPercent', 0.0, 100.0, 50.0)
    VisibilityAvgMiles = st.sidebar.slider('VisibilityAvgMiles', 0.0, 10.0, 5.0)
    WindHighMPH = st.sidebar.slider('WindHighMPH', 0.0, 20.0, 10.0)
    data = {'VisibilityLowMiles': VisibilityLowMiles,
            'PrecipitationSumInches': PrecipitationSumInches,
            'HumidityLowPercent': HumidityLowPercent,
            'HumidityAvgPercent': HumidityAvgPercent,
            'VisibilityAvgMiles': VisibilityAvgMiles,
            'WindHighMPH': WindHighMPH}
    features = pd.DataFrame(data, index=[0])
    return features

# store the user input into a variable
input_df = user_input_features()

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
