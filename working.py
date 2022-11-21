from fastapi import FastAPI
import uvicorn


from EnsembleModel import EnsembleModel
import numpy as np

app = FastAPI()
model=EnsembleModel()

@app.get("/")
def index():
    return {"message": "Hello, This is our YAP470 Project"}

@app.get('/{name}')
def get_name(name: str):
    return {'Hello, {name}. We are ready to predict the weather'}  

@app.get('/predict')
def predict(data: dict):
    data = np.array([data['data']])
    prediction = model.predict(data)
    return {'prediction': prediction.tolist()}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

