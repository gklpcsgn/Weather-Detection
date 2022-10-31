import pandas as pd
import numpy as np

def preprocess(df):
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
    

    from sklearn.preprocessing import StandardScaler 
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    X_train = pd.DataFrame(X_train, columns=df.columns.drop(["Rain","Fog","Thunderstorm"]))
    X_val = pd.DataFrame(X_val, columns=df.columns.drop(["Rain","Fog","Thunderstorm"]))
    y_train = pd.DataFrame(y_train, columns=["Rain","Fog","Thunderstorm"])
    y_val = pd.DataFrame(y_val, columns=["Rain","Fog","Thunderstorm"])

    return  X_train, X_test, y_train, y_test, X_val, y_val