
# Day 26: Model Calibration

## Overview
This project compares a logistic regression classifier with a
probability-calibrated version using the Breast Cancer Wisconsin
dataset available in scikit-learn.

## Concepts
- Classification probabilities
- `predict()` versus `predict_proba()`
- Sigmoid probability calibration
- Brier score and log loss
- Train/test separation and cross-validation

## Project Structure
- `train_model.py`: trains, evaluates, and saves models
- `predict.py`: loads the calibrated model and predicts
- `model/`: saved model artifact
- `reports/`: evaluation report
- `requirements.txt`: dependencies

## Setup
Run from this directory using the parent repository's virtual environment:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train and Evaluate

```powershell
..\.venv\Scripts\python.exe train_model.py
```

## Make a Prediction

```powershell
..\.venv\Scripts\python.exe predict.py
```

## Evaluation
The project compares accuracy, Brier score, and log loss on the
same held-out test set. Lower Brier score and log loss generally
indicate better probability estimates.

Calibration does not guarantee improved accuracy or better
probabilities in every dataset.

## Disclaimer
Educational project only. This is not a medical diagnostic system.
