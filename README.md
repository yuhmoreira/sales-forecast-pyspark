# 📈 Sales Forecasting with PySpark – Retail Demand Prediction

This project implements a **time series forecasting pipeline in PySpark** to predict daily product demand for an online retail store.  
Using transactional invoice data, the notebook builds, evaluates, and optimizes regression models that forecast **daily quantity sold per product and country**.

🔍 **Goal:**  
Predict the **daily quantity sold for each (Country, StockCode, Date)** and show that a well‑designed machine learning pipeline can **significantly outperform a simple time‑series baseline**, both in standard metrics and in **value‑weighted business impact**.

---

## 📊 Dataset

The project uses the **Online Retail** dataset, which contains all transactions from **01/12/2010 to 09/12/2011** for a UK‑based online store.

- **Source:** [Online Retail – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail) / [Kaggle](https://www.kaggle.com/datasets/tunguz/online-retail/).
- **Raw rows:** 541,909 invoice line items  
- **Columns (original):**
  - `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`
- **Target (after aggregation):**  
  - `NetDailyQuantity` – total quantity sold per `(Country, StockCode, InvoiceDate)`
- **Train / Test split (time‑based):**
  - **Train:** all rows with `InvoiceDate` ≤ **2011‑09‑01**  
    → 197,016 rows (~64%)
  - **Test:** all rows with `InvoiceDate` > **2011‑09‑01**  
    → 112,999 rows (~36%)

## 🧠 Workflow & Analysis

### Key Steps:

- **Data Preparation & Aggregation**: Cleaned the raw transactional dataset and aggregated it at a daily/weekly level by product and country, summing quantities and computing average unit prices.
- **Feature Engineering**: Extracted calendar features (**Year**, **Month**, **Week**, **Day**, **DayOfWeek**) from invoice dates to capture seasonality. Created **lagged features**, **rolling window statistics** (mean, std), and **difference calculations** on rolling data to capture trends and momentum in sales patterns.
- **Train/Test Split (Time-Based)**: Split the data chronologically using a **date cutoff** to respect temporal ordering and prevent data leakage.
- **Model Training**: Built and compared multiple regression models in PySpark:
  - **Random Forest Regressor**
  - **Linear Regression**
  - **Gradient Boosting Trees (GBT)**
- **Performance Evaluation**: Assessed model performance using **MAE** (Mean Absolute Error), **RMSE** (Root Mean Squared Error) and **R²** (coefficient of determination) on the test set to measure forecast accuracy and variance explained.
- **Model Segmentation Strategy**: Split the modeling approach into two segments, **high-volume country (UK)** and **other countries**, to capture region-specific patterns and improve overall prediction accuracy.
- **Business Insight & Cost Reduction**: Compared model predictions against a **moving average (MA) baseline**. Demonstrated **cost reduction** by quantifying the improvement in forecast accuracy (lower MAE) of the ML model vs. the MA approach, translating to better inventory planning and reduced overstock/understock costs.

---

## 📈 Key Findings

- **United Kingdom:**
  - Train: 170,079 rows | Test: 95,337 rows
  - **MAE = 0.51**, **RMSE = 11.21**, **R² = 0.9790**
- **Other Countries:**
  - Train: 26,937 rows | Test: 17,662 rows
  - **MAE = 7.09**, **RMSE = 22.90**, **R² = 0.7734**

The project also computes detailed **KPIs by country** (MAE, MAPE, bias) and compares:

- Global Linear Regression vs Segmented Linear Regression
- Both vs the moving‑average baseline

### 7. Evaluation & Diagnostics

For the best models, the notebook includes:

- **Standard metrics** on the test set:
  - MAE, RMSE, R² (global and by country)
- **Baseline vs model comparison**:
  - Global KPIs:
    - Baseline: `MAE ≈ 18.18`, `MAPE ≈ 2.49`, negative bias
    - Linear Regression v2: `MAE ≈ 1.50`, `MAPE ≈ 0.21`, near‑zero bias
- **Value‑weighted absolute error** (using revenue as weight):
  - Baseline: total error ≈ **4,908,743**
  - Segmented Linear Regression: total error ≈ **810,486**
  - → **83.49% reduction** in total value‑weighted forecast error
  - For the UK alone: ≈ **95.96% reduction**

Model diagnostics & plots:

- Time series: **Actual vs Predicted vs Baseline** (global and UK‑only)
- Scatter plots: **Actual vs Predicted** with y = x diagonal
- Residuals:
  - Residuals vs predicted value
  - Histogram of residuals
- Error over time:
  - Daily MAE across the test period
- Error by segment:
  - MAE per country (table + bar chart)
- Feature importance:
  - Linear Regression coefficients (showing that rolling medians/means and calendar features dominate)

---

## 📈 Key Findings

- **Massive improvement over baseline:**
  - Global value‑weighted forecast error reduced by **83.49%** vs a moving‑average baseline.
  - For the UK, value‑weighted error reduced by **~96%**.

- **Simple model, rich features win:**
  - A regularized **Linear Regression** with carefully designed **lag + rolling features** achieved:
    - **MAE = 1.50**, **R² = 0.9648** on the test set.
  - Tree‑based models (Random Forest, GBT) could not match this performance.

- **Segmentation helps:**
  - Splitting by **United Kingdom vs Others** improved results for **24 out of 31 countries** (lower MAE vs global model).
  - After segmentation, the model beats the baseline in **18 out of 31 countries**.

---

## 🧰 Technologies

- **Python** 3.10+  
- **Apache Spark / PySpark** 3.5.1  
  - `pyspark.sql` (DataFrame API, Window functions)
  - `pyspark.ml` (Pipeline, Feature Engineering, Regression models)
- **PySpark ML Models**: Linear Regression, Random Forest, Gradient Boosting Trees (GBT)
- **Pandas**, **NumPy** (for local data manipulation and analysis)
- **Matplotlib**, **Seaborn** (for visualization)
- **Jupyter Notebook** / **DataCamp Workspace**

---

## 🔁 Reproducibility

To run this project locally:

1.  Clone the repo:
    ```bash
    git clone https://github.com/yuhmoreira/sales-forecast-pyspark.git
    cd sales-forecast-pyspark
    ```
2.  Install dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ensure your [Kaggle API key](https://www.kaggle.com/docs/api) is set up. The notebook will automatically download the dataset.

---
