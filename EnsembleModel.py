#create a class for ensemble model
import pickle
import joblib
import numpy as np
class EnsembleModel:
    def __init__(self):
        self.model_rain = joblib.load('best_rain.pkl')
        self.model_fog = joblib.load('best_fog.pkl')
        self.model_thunderstorm = joblib.load('best_thunderstorm.pkl')
    def predict(self, X):
        #take the first column of X
        
        predictions = np.column_stack([
            self.model_rain.predict(X),
            self.model_fog.predict(X),
            self.model_thunderstorm.predict(X)
        ])
        return predictions
