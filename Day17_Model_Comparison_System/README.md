# Day 17 — Model Comparison System

## 📌 Project Overview

The **Model Comparison System** is a beginner-friendly Machine Learning project that compares multiple classification algorithms and identifies the best-performing model based on accuracy.

This project uses the **Iris dataset** from Scikit-learn to understand how different Machine Learning algorithms perform on the same dataset.

## 🎯 Objectives

- Understand how to train multiple Machine Learning models.
- Learn how to split a dataset into training and testing sets.
- Understand feature scaling and why it matters.
- Compare model performance using accuracy.
- Identify the best-performing model.

## 🧠 Machine Learning Algorithms

This project compares four classification algorithms:

1. **Logistic Regression** — predicts classes using a linear decision boundary.
2. **K-Nearest Neighbors (KNN)** — predicts a class based on nearby training examples.
3. **Decision Tree Classifier** — makes predictions using a sequence of decision rules.
4. **Random Forest Classifier** — combines multiple decision trees to make predictions.

## 📊 Dataset

**Dataset:** Iris dataset

- Samples: 150
- Features: 4
- Classes: Setosa, Versicolor, Virginica

The four input features represent flower measurements, including sepal length, sepal width, petal length, and petal width.

## 🛠️ Technologies Used

- Python
- NumPy
- Scikit-learn
- Machine Learning
- Git and GitHub

## 📁 Project Structure

```text
Day17_Model_Comparison_System/
│
├── compare_models.py
└── README.md
```

## ⚙️ How to Run

### 1. Open the project folder

Open your `30-Days-of-Machine-Learning` project in VS Code.

### 2. Run the Python script

From the main project directory, run:

```powershell
..\.venv\Scripts\python.exe Day17_Model_Comparison_System\compare_models.py
```

If your terminal is already inside the `Day17_Model_Comparison_System` folder, run:

```powershell
..\.venv\Scripts\python.exe compare_models.py
```

### 3. Review the results

The program compares the four algorithms and reports their accuracy scores and the best-performing model.

**Note:** The script can run only when its required Python libraries load successfully. A Windows Application Control restriction previously blocked a compiled NumPy/SciPy component in this environment.

## 📈 Key Learning Outcomes

- Loading a dataset using Scikit-learn.
- Separating features (`X`) from target labels (`y`).
- Using `train_test_split()`.
- Applying `StandardScaler` to appropriate models.
- Training models with `.fit()`.
- Making predictions with `.predict()`.
- Evaluating predictions using `accuracy_score()`.
- Comparing model results programmatically.

## 🚀 Future Improvements

- Compare additional classification algorithms.
- Evaluate models using precision, recall, and F1-score.
- Display results in a comparison table.
- Add confusion matrices.
- Save the best-performing model for future predictions.

## 👨‍💻 Author

**Athul Sathyan**

GitHub: [athulsathyan136-alt](https://github.com/athulsathyan136-alt)

---

*Part of my 30 Days of Machine Learning learning journey.*
