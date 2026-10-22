
# Day 27: Ensemble Learning

## Overview
Compare Logistic Regression, Random Forest, and a Soft Voting
Classifier on the Breast Cancer Wisconsin dataset.

## Concepts Learned
- Ensemble learning
- Hard voting vs. soft voting
- Classification metrics
- Confusion matrix
- Saving and loading a trained model

## Project Structure
- `train_model.py`: trains and compares the models
- `predict.py`: loads the saved ensemble for predictions
- `model/`: stores the saved model
- `reports/`: stores evaluation results
- `requirements.txt`: Python dependencies

## Setup
Run these commands from this directory:

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
All three models use the same train/test split. Accuracy, precision,
recall, F1 score, confusion matrices, and classification reports
are saved in `reports/ensemble_report.txt`.

An ensemble does not always outperform its individual models.

## Disclaimer
Educational project only. Not intended for clinical diagnosis.
