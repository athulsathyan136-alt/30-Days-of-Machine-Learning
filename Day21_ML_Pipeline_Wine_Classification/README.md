
# Day 21: ML Pipeline - Wine Classification

## Overview

An end-to-end machine-learning project that classifies
wine samples using a Scikit-learn Pipeline.

## Features

- Loads the built-in Wine dataset
- Splits data into training and testing sets
- Scales numerical features with StandardScaler
- Trains a Logistic Regression classifier
- Evaluates accuracy, precision, recall and F1-score
- Generates a confusion matrix
- Saves the complete fitted pipeline using Joblib
- Reuses the saved pipeline for predictions
- Saves a text evaluation report

## Technologies

- Python
- Scikit-learn
- Joblib

## Project Structure

    Day21_ML_Pipeline_Wine_Classification/
    ├── model/
    │   └── wine_pipeline.joblib
    ├── reports/
    │   └── evaluation_report.txt
    ├── train_pipeline.py
    ├── predict.py
    ├── requirements.txt
    └── README.md

## Run the Project

Install dependencies:

    ..\.venv\Scripts\python.exe -m pip install -r requirements.txt

Train and save the pipeline:

    ..\.venv\Scripts\python.exe .\train_pipeline.py

Load the saved pipeline and predict:

    ..\.venv\Scripts\python.exe .\predict.py

## Learning Outcomes

- Building reusable ML pipelines
- Scaling features without test-data leakage
- Training a Logistic Regression classifier
- Evaluating multiclass predictions
- Saving and loading a complete preprocessing/model pipeline
