
# Day 18: Machine Learning Model Saving and Loading

## Project Overview

This project demonstrates how to train a machine-learning
model, save it to a file, load it again, and make predictions.

## Objective

Predict Iris flower species from four measurements.

## Technologies Used

- Python
- Pandas (not required by the current implementation)
- NumPy (used internally by Scikit-learn)
- Scikit-learn
- Joblib

## Dataset

The Iris dataset contains 150 flower samples with four
features and three species.

## Project Structure

```text
Day18_Model_Saving_and_Loading/
├── model/
│   └── iris_model.joblib
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
```

## How to Run

Install the dependencies:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train and save the model:

```powershell
..\.venv\Scripts\python.exe .\train_model.py
```

Load the saved model and make a prediction:

```powershell
..\.venv\Scripts\python.exe .\predict.py
```

## Key Concepts

- Training a Random Forest classifier
- Train/test splitting
- Evaluating model accuracy
- Saving a trained model with Joblib
- Loading a saved model
- Making predictions on new measurements

## Learning Outcome

The project demonstrates how a trained model can be
reused without retraining every time a prediction is needed.
