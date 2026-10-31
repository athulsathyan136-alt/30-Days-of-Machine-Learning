# Day 19 — Machine Learning Prediction API

## Overview

This project demonstrates how to train a Machine Learning classification model and serve predictions through a REST API using **FastAPI**.

The project uses the Iris dataset and a Random Forest Classifier to predict the species of an Iris flower based on four measurements.

## Features

- Train a Random Forest classification model.
- Split data into training and testing sets.
- Evaluate model accuracy.
- Save the trained model using Joblib.
- Build a REST API with FastAPI.
- Validate prediction inputs using Pydantic.
- Test API endpoints using Swagger UI.

## Technologies Used

- Python
- NumPy and scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Pydantic
- Iris dataset

## Project Structure

```text
Day19_ML_Prediction_API/
├── api.py
├── train_model.py
├── requirements.txt
├── README.md
└── model/
    └── iris_model.joblib
```

The `model/` directory and `iris_model.joblib` file are generated when the training script runs successfully.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/athulsathyan136-alt/30-Days-of-Machine-Learning.git
```

Open the repository and navigate to the `Day19_ML_Prediction_API` directory.

### 2. Install dependencies

Activate your virtual environment and run:

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This trains the Random Forest model, evaluates its accuracy, and saves it to `model/iris_model.joblib`.

### 4. Start the API

```bash
python -m uvicorn api:app --reload
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API homepage |
| GET | `/health` | Check API health |
| POST | `/predict` | Predict Iris flower species |

The exact response format depends on the implementation in `api.py`.

## Interactive API Documentation

After starting the server, open:

http://127.0.0.1:8000/docs

Use Swagger UI to inspect the endpoints and test predictions.

## Sample Prediction Input

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Submit this input to `POST /predict` if these field names match the request schema defined in `api.py`.

## Machine Learning Workflow

1. Load the Iris dataset.
2. Separate features and target labels.
3. Split the dataset into training and testing sets.
4. Train a Random Forest Classifier.
5. Evaluate predictions using accuracy score.
6. Save the trained model using Joblib.
7. Load the model in FastAPI.
8. Accept prediction requests and return results.

## Learning Objectives

- Understand the process of training and saving an ML model.
- Learn how to serve a model through a REST API.
- Understand API endpoints and request validation.
- Connect machine learning workflows with backend development.

## Author

Athul Sathyan

## License

This project is intended for educational and portfolio purposes.
