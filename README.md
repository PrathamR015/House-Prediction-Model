# House Price Prediction

## 📌 Project Overview

This project predicts **house prices** based on multiple features such as living area, number of rooms, quality, year built, and garage capacity using a **Random Forest Regression model**.  
This is still in progress. I am first testing the model with test data and will scale up eventually.

---

## 🎯 Objectives

- Build a machine learning model to predict house prices
- Use **Random Forest Regressor** to handle non-linear relationships
- Analyze **feature importance**
- Evaluate performance using standard regression metrics

---

## 🧠 Machine Learning Algorithm Used

**Random Forest Regressor**

- Ensemble learning method
- Combines multiple decision trees
- Reduces overfitting
- Handles non-linear data efficiently

---

## 📂 Project Structure

```

House-Price-Prediction/
│
├── house_prices.csv        # Dataset
├── house_price_model.py    # Model training & evaluation code
├── README.md               # Project documentation
└── requirements.txt        # Required libraries

```

---

## 📊 Dataset Description

The dataset is inspired by Kaggle’s **House Prices** dataset.

| Feature Name | Description                         |
| ------------ | ----------------------------------- |
| LotArea      | Size of the lot in square feet      |
| OverallQual  | Overall material and finish quality |
| OverallCond  | Overall condition of the house      |
| YearBuilt    | Year the house was built            |
| TotalRooms   | Total number of rooms               |
| FullBath     | Number of full bathrooms            |
| GarageCars   | Garage capacity                     |
| GrLivArea    | Above ground living area (sq ft)    |
| SalePrice    | Target variable (house price)       |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/USERNAME/house-price-prediction.git
cd house-price-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
pandas
numpy
scikit-learn
matplotlib
```

---

## ▶️ How to Run the Project

```bash
python house_price_model.py
```

The script will:

- Train the Random Forest model
- Evaluate performance
- Predict prices for new data
- Display feature importance graph

---

## 📈 Model Evaluation Metrics

The model is evaluated using:

- **MAE (Mean Absolute Error)**
- **MSE (Mean Squared Error)**
- **R² Score**

These metrics measure prediction accuracy and model reliability.

---

## 📌 Feature Importance

The project includes a **feature importance visualization** to understand which factors most influence house prices.

Typical important features:

- `GrLivArea`
- `OverallQual`
- `GarageCars`
- `YearBuilt`

---

## 🧮 Mathematical Explanation (Brief)

Random Forest works by:

1. Creating multiple decision trees using random subsets of data
2. Each tree predicts a house price
3. Final prediction is the **average of all tree predictions**

[
\hat{y} = \frac{1}{N} \sum_{i=1}^{N} y_i
]

This reduces variance and improves generalization.

---

## 🧪 Sample Prediction

```python
new_house = [[9000, 7, 5, 2010, 8, 2, 2, 1800]]
predicted_price = model.predict(new_house)
```

---

## 🚀 Future Improvements

- Hyperparameter tuning (GridSearchCV)
- SHAP / Permutation feature importance
- Flask / Streamlit web app
- Use full Kaggle dataset
- Cross-validation

---

## 👨‍💻 Author

**Pratham Raval**

---
