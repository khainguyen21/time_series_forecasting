# 🌀 CO₂ Time Series Forecasting

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This project demonstrates a simple pipeline for forecasting atmospheric CO₂ concentrations using machine learning models on time series data. It includes preprocessing, feature engineering, model training, and performance evaluation.

---

## 📂 Dataset

The project expects a file named **`co2.csv`** in the working directory with the following columns:

- **`time`**: Timestamps of measurements (string or datetime format)
- **`co2`**: Atmospheric CO₂ concentration (numerical)

Missing values in the `co2` column are filled via linear interpolation.

---

## ⚙️ Workflow Overview

1. **Data Preprocessing**
   - Converts `time` column to `datetime`
   - Interpolates missing CO₂ values
   - Generates lagged features and multi-step targets using a sliding window

2. **Feature Engineering**
   - Uses past `window_size` time steps to predict the next `target_size` values

3. **Model Training**
   - Trains:
     - `LinearRegression` for each target step
     - `RandomForestRegressor` for the first target step
     
4. **Evaluation**
   - Prints:
     - 📉 Mean Absolute Error (MAE)
     - 📊 Mean Squared Error (MSE)
     - 📈 R² Score

5. **Visualization (optional)**
   - Raw CO₂ data over time
   - Training vs Testing vs Prediction curves *(commented out in code)*

---

## 🧪 Example Configuration

```python
window_size = 5
target_size = 3
train_ratio = 0.8
