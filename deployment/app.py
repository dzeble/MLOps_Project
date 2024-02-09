
# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Quality import WineQuality
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("rfc.pkl","rb")
classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To My Wine Quality Predictor': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_quality(data:WineQuality):

    fixed_acidity = data.fixed_acidity
    volatile_acidity = data.volatile_acidity
    citric_acid = data.citric_acid
    residual_sugar = data.residual_sugar
    chlorides = data.chlorides
    free_sulfur_dioxide = data.free_sulfur_dioxide
    total_sulfur_dioxide = data.total_sulfur_dioxide
    density = data.density
    pH = data.pH
    sulphates = data.sulphates
    alcohol = data.alcohol
    red_wine = data.red_wine


        # Convert the features to a NumPy array
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                          chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                          pH, sulphates, alcohol, red_wine]])

   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict(features)

    prediction = prediction[0].item()

    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload