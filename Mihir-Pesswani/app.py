import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from BankNotes import BankNote

app = FastAPI()
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.get('/{name}')
def get_name(name: str):
    return {'Welcome': f'{name}'}


@app.post('/predict')
def predict_species(data: BankNote):
    data = data.dict()
    print(data)
    print('Hello')
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    print(classifier.predict([[variance, skewness, curtosis, entropy]]))
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if(prediction[0] > 0.5):
        prediction = "Fake Note"
    else:
        prediction = "It is a Bank Note"

    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)