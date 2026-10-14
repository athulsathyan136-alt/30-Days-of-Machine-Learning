from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel,Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR /"model"/"iris_model.joblib"

if not MODEL_PATH.exists():
    raise RuntimeError("model file not found")

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
target_name = model_package["target_names"]

app = FastAPI(
    title='Iris Flower Prediction API',
    description="Predict an Iris flower species from four measurements.",
    version='1.0.0'
)

class FlowerInput(BaseModel):
    sepal_length:float =Field(gt=0)
    sepal_width:float =Field(gt=0)
    petal_length:float =Field(gt=0)
    petal_width:float =Field(gt=0)

@app.get('/')
def home():
    return{
       "message": "Iris Flower Prediction API is running",
        "docs": "/docs",
        "health": "/health"
    }
@app.get('/health')
def health():
    return{
      "status": "healthy",
        "model_loaded": True  
    }
@app.post('/predict')
def predict(flower:FlowerInput):
    measurement = [[flower.sepal_length,flower.sepal_width,
                    flower.petal_length,flower.petal_width]]

    predicted_class = int(model.predict(measurement)[0])

    predicted_species = target_name[predicted_class]

    probability = model.predict_proba(measurement)[0]

    probability_results = {
        name:round(float(class_probability),4)

        for name, class_probability  in zip(target_name,probability)

    }

    return{
         "predicted_class": predicted_class,
        "predicted_species": predicted_species,
        "class_probabilities": probability_results
    }