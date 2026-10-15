# Day 20: Interactive ML Prediction Web App

## Overview

An interactive web application that predicts Iris flower
species using a trained Random Forest classifier.

## Features

- Interactive measurement sliders
- Machine-learning predictions
- Class probability display
- Saved model loaded using Joblib
- Web interface powered by Streamlit

## Technologies

- Python
- Scikit-learn
- Joblib
- Streamlit

## Project Structure

    Day20_Streamlit_ML_Web_App/
    ├── model/
    │   └── iris_model.joblib
    ├── train_model.py
    ├── app.py
    ├── requirements.txt
    └── README.md

## Run the Project

Install dependencies:

    ..\.venv\Scripts\python.exe -m pip install -r requirements.txt

Train the model:

    ..\.venv\Scripts\python.exe .\train_model.py

Start the web app:

    ..\.venv\Scripts\python.exe -m streamlit run .\app.py

## Learning Outcomes

- Training and evaluating an ML model
- Saving and loading models
- Building an interactive interface
- Collecting user inputs
- Displaying predictions and probabilities