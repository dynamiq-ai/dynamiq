---
name: math_analytics_pipeline
version: "2.0.0"
description: Deep mathematical and statistical analysis pipelines for data science workflows. Covers data loading, cleaning, statistical analysis, mathematical modeling, time series forecasting, and automated reporting.
author: Dynamiq Team
tags: [mathematics, statistics, data-science, analytics, modeling, forecasting, machine-learning]
dependencies:
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - statsmodels
  - matplotlib
  - seaborn
  - plotly
supporting_files:
  - scripts/data_pipeline.py
  - scripts/statistical_analysis.py
  - scripts/time_series.py
  - scripts/ml_models.py
  - templates/analysis_report_template.html
---

# Math Analytics Pipeline Skill

## Overview

This skill provides comprehensive workflows for deep mathematical and statistical analysis. It covers the complete data science pipeline from raw data ingestion to mathematical modeling, statistical inference, and automated report generation.

**Core Capabilities:**
- **Data Engineering**: Load, clean, transform, and validate datasets
- **Exploratory Analysis**: Descriptive statistics, distributions, correlations
- **Statistical Testing**: Hypothesis testing, ANOVA, regression analysis
- **Mathematical Modeling**: Optimization, differential equations, numerical methods
- **Time Series**: Forecasting, decomposition, ARIMA, Prophet
- **Machine Learning**: Classification, regression, clustering, model evaluation
- **Visualization**: Statistical plots, interactive dashboards
- **Reporting**: Automated analysis reports with insights

---

## Decision Tree: Choose Your Analysis Workflow

### 1. Exploratory Data Analysis (EDA)
**When:** Initial data exploration, understanding distributions and patterns
→ Follow **EDA Workflow** below

### 2. Statistical Inference
**When:** Hypothesis testing, confidence intervals, significance testing
→ Follow **Statistical Testing Workflow** below

### 3. Predictive Modeling
**When:** Forecasting, classification, regression problems
→ Follow **ML Pipeline Workflow** below

### 4. Time Series Analysis
**When:** Sequential data, temporal patterns, forecasting
→ Follow **Time Series Workflow** below

### 5. Mathematical Optimization
**When:** Constrained optimization, linear programming, numerical solutions
→ Follow **Optimization Workflow** below

---

## Workflow 1: Exploratory Data Analysis (EDA)

### Step 1: Data Loading and Validation

```python
import pandas as pd
import numpy as np
from scripts.data_pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline()

# Load data with validation
df = pipeline.load_data(
    source="data.csv",  # or "database_query", "api_endpoint"
    validate_schema=True,
    handle_missing="auto"  # or "drop", "impute", "flag"
)

# Quick overview
pipeline.data_summary(df)
# Returns: shape, dtypes, missing values, memory usage
```

### Step 2: Data Cleaning and Transformation

```python
# Clean data
df_clean = pipeline.clean_data(
    df,
    remove_duplicates=True,
    handle_outliers="iqr",  # IQR method
    normalize_columns=["age", "income"],  # Standardize
    encode_categorical=["category", "region"]  # One-hot encoding
)

# Feature engineering
df_clean = pipeline.engineer_features(
    df_clean,
    interactions=[("feature1", "feature2")],  # Interaction terms
    polynomials={"feature1": 2},  # Polynomial features
    date_features=["timestamp"],  # Extract day, month, year, etc.
)
```

### Step 3: Descriptive Statistics

```python
from scripts.statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(df_clean)

# Comprehensive statistics
stats = analyzer.describe_all(
    include_distributions=True,
    test_normality=True,
    compute_correlations=True
)

# Key metrics:
# - Central tendency: mean, median, mode
# - Dispersion: std, variance, IQR, range
# - Shape: skewness, kurtosis
# - Distribution tests: Shapiro-Wilk, Anderson-Darling
# - Correlation matrix with significance tests
```

### Step 4: Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution plots
analyzer.plot_distributions(
    columns=["age", "income", "score"],
    plot_type="histogram",  # or "kde", "violin", "box"
    save_path="distributions.png"
)

# Correlation heatmap
analyzer.plot_correlation_matrix(
    method="pearson",  # or "spearman", "kendall"
    annotate=True,
    save_path="correlation.png"
)

# Pairplot for relationships
analyzer.plot_pairplot(
    columns=["feature1", "feature2", "feature3", "target"],
    hue="category",
    save_path="pairplot.png"
)
```

---

## Workflow 2: Statistical Testing

### Hypothesis Testing Framework

```python
from scripts.statistical_analysis import HypothesisTester

tester = HypothesisTester(df_clean)

# T-Test (comparing two groups)
result = tester.t_test(
    group1=df_clean[df_clean["group"] == "A"]["metric"],
    group2=df_clean[df_clean["group"] == "B"]["metric"],
    alternative="two-sided",  # or "greater", "less"
    alpha=0.05
)
# Returns: t-statistic, p-value, confidence interval, effect size, conclusion

# ANOVA (comparing multiple groups)
anova_result = tester.anova(
    data=df_clean,
    dependent_var="outcome",
    groups="treatment_group",
    post_hoc="tukey"  # Post-hoc test if significant
)

# Chi-Square Test (categorical associations)
chi_result = tester.chi_square_test(
    data=df_clean,
    var1="category1",
    var2="category2"
)
```

### Regression Analysis

```python
# Linear Regression
regression = tester.linear_regression(
    data=df_clean,
    target="y",
    features=["x1", "x2", "x3"],
    include_diagnostics=True
)

# Returns:
# - Coefficients with standard errors
# - R-squared, Adjusted R-squared
# - F-statistic and p-value
# - Residual diagnostics (normality, homoscedasticity, autocorrelation)
# - VIF for multicollinearity

# Logistic Regression (binary outcome)
logistic = tester.logistic_regression(
    data=df_clean,
    target="is_churn",
    features=["tenure", "monthly_charges", "total_charges"],
    include_odds_ratios=True
)
```

---

## Workflow 3: Machine Learning Pipeline

### Step 1: Train-Test Split and Preprocessing

```python
from scripts.ml_models import MLPipeline

ml_pipeline = MLPipeline()

# Split data
X_train, X_test, y_train, y_test = ml_pipeline.split_data(
    df_clean,
    target="target",
    test_size=0.2,
    stratify=True,  # For classification
    random_state=42
)

# Preprocessing pipeline
X_train_processed, X_test_processed = ml_pipeline.preprocess(
    X_train, X_test,
    scaling="standard",  # or "minmax", "robust"
    handle_categorical="onehot",  # or "label", "target"
    feature_selection="mutual_info",  # or "chi2", "f_test", None
    n_features=20  # Top N features
)
```

### Step 2: Model Training and Selection

```python
# Train multiple models
models = ml_pipeline.train_models(
    X_train_processed, y_train,
    models=[
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "neural_network"
    ],
    cross_validation=5,
    scoring="roc_auc",  # or "accuracy", "f1", "mse"
    hyperparameter_tuning=True
)

# Compare models
comparison = ml_pipeline.compare_models(
    models,
    X_test_processed,
    y_test,
    metrics=["accuracy", "precision", "recall", "f1", "roc_auc"]
)

# Best model
best_model = comparison["best_model"]
```

### Step 3: Model Evaluation

```python
# Comprehensive evaluation
evaluation = ml_pipeline.evaluate_model(
    best_model,
    X_test_processed,
    y_test,
    plot_confusion_matrix=True,
    plot_roc_curve=True,
    plot_feature_importance=True,
    save_plots=True
)

# Returns:
# - Classification report
# - Confusion matrix
# - ROC curve and AUC
# - Precision-Recall curve
# - Feature importances
# - Cross-validation scores
# - Learning curves
```

### Step 4: Prediction and Interpretation

```python
# Make predictions
predictions = ml_pipeline.predict(
    best_model,
    X_test_processed,
    return_probabilities=True
)

# Feature importance analysis
importance = ml_pipeline.analyze_feature_importance(
    best_model,
    feature_names=X_train.columns.tolist(),
    plot=True
)

# SHAP values for interpretability
shap_values = ml_pipeline.explain_predictions(
    best_model,
    X_test_processed,
    method="shap",  # or "lime"
    sample_size=100
)
```

---

## Workflow 4: Time Series Analysis

### Step 1: Time Series Preprocessing

```python
from scripts.time_series import TimeSeriesAnalyzer

ts_analyzer = TimeSeriesAnalyzer()

# Load time series data
df_ts = pd.read_csv("time_series.csv", parse_dates=["date"])
df_ts = df_ts.set_index("date")

# Check stationarity
stationarity = ts_analyzer.test_stationarity(
    df_ts["value"],
    tests=["adf", "kpss", "pp"]  # Multiple tests
)

# Make stationary if needed
df_ts_stationary, transformations = ts_analyzer.make_stationary(
    df_ts["value"],
    method="auto"  # auto-detect best transformation
)
```

### Step 2: Decomposition and Seasonality

```python
# Decompose time series
decomposition = ts_analyzer.decompose(
    df_ts["value"],
    model="additive",  # or "multiplicative"
    period=12  # Monthly seasonality
)

# Extract components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Detect seasonality
seasonality_test = ts_analyzer.detect_seasonality(
    df_ts["value"],
    max_lag=48
)
```

### Step 3: Forecasting Models

```python
# ARIMA Model
arima_model = ts_analyzer.fit_arima(
    df_ts["value"],
    order=(1, 1, 1),  # (p, d, q) or "auto" for auto-selection
    seasonal_order=(1, 1, 1, 12)  # (P, D, Q, s) for seasonal
)

# Forecast
arima_forecast = ts_analyzer.forecast(
    arima_model,
    steps=12,  # 12 periods ahead
    return_confidence_interval=True
)

# Prophet Model (Facebook)
prophet_model = ts_analyzer.fit_prophet(
    df_ts,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.05,
    holidays=us_holidays  # Optional holidays dataframe
)

prophet_forecast = ts_analyzer.forecast_prophet(
    prophet_model,
    periods=12
)

# LSTM Model (Deep Learning)
lstm_model = ts_analyzer.fit_lstm(
    df_ts["value"],
    lookback=30,  # Use last 30 periods
    layers=[50, 50],  # LSTM layer sizes
    epochs=100
)

lstm_forecast = ts_analyzer.forecast_lstm(
    lstm_model,
    steps=12
)
```

### Step 4: Forecast Evaluation

```python
# Compare forecasting models
comparison = ts_analyzer.compare_forecasts(
    actual=df_ts["value"][-12:],  # Last 12 periods as test
    forecasts={
        "ARIMA": arima_forecast,
        "Prophet": prophet_forecast,
        "LSTM": lstm_forecast
    },
    metrics=["mae", "mse", "rmse", "mape", "mase"]
)

# Plot forecasts
ts_analyzer.plot_forecasts(
    actual=df_ts["value"],
    forecasts=comparison["forecasts"],
    save_path="forecast_comparison.png"
)
```

---

## Workflow 5: Mathematical Optimization

### Linear Programming

```python
from scripts.ml_models import Optimizer

optimizer = Optimizer()

# Define optimization problem
result = optimizer.linear_program(
    objective_coefficients=[3, 5],  # Maximize 3x + 5y
    constraints=[
        {"coefficients": [1, 0], "bound": 4, "type": "<="},
        {"coefficients": [0, 2], "bound": 12, "type": "<="},
        {"coefficients": [3, 2], "bound": 18, "type": "<="}
    ],
    bounds=[(0, None), (0, None)],  # x, y >= 0
    method="highs"  # or "simplex", "interior-point"
)

# Returns: optimal values, objective value, status
```

### Nonlinear Optimization

```python
# Minimize a function
def objective(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2

def constraint(x):
    return x[0] + x[1] - 5  # x + y = 5

result = optimizer.minimize(
    objective,
    initial_guess=[0, 0],
    constraints=[{"type": "eq", "fun": constraint}],
    bounds=[(0, 10), (0, 10)],
    method="SLSQP"
)
```

### Curve Fitting and Parameter Estimation

```python
# Fit nonlinear model to data
def model(x, a, b, c):
    return a * np.exp(b * x) + c

params, covariance = optimizer.curve_fit(
    model,
    x_data=df["x"],
    y_data=df["y"],
    initial_params=[1, 0.1, 0],
    bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
)

# Parameter confidence intervals
param_errors = np.sqrt(np.diag(covariance))
```

---

## Advanced Features

### Automated Report Generation

```python
from scripts.data_pipeline import ReportGenerator

generator = ReportGenerator()

# Generate comprehensive analysis report
report = generator.create_report(
    data=df_clean,
    analysis_type="full",  # or "eda", "ml", "time_series"
    include_sections=[
        "executive_summary",
        "data_overview",
        "statistical_analysis",
        "visualizations",
        "model_results",
        "recommendations"
    ],
    template="templates/analysis_report_template.html",
    output_format="html",  # or "pdf", "markdown"
    output_path="analysis_report.html"
)
```

### Anomaly Detection

```python
from scripts.statistical_analysis import AnomalyDetector

detector = AnomalyDetector()

# Detect outliers using multiple methods
anomalies = detector.detect_anomalies(
    df_clean["metric"],
    methods=["zscore", "iqr", "isolation_forest", "lof"],
    threshold=3,  # Z-score threshold
    contamination=0.1  # Expected anomaly proportion
)

# Returns: anomaly indices, scores, method agreement
```

### Dimensionality Reduction

```python
# PCA
pca_result = ml_pipeline.reduce_dimensions(
    X_train_processed,
    method="pca",
    n_components=10,  # or "auto" for explained variance threshold
    plot_variance=True
)

# t-SNE for visualization
tsne_result = ml_pipeline.reduce_dimensions(
    X_train_processed,
    method="tsne",
    n_components=2,
    perplexity=30
)

# UMAP (better than t-SNE for large datasets)
umap_result = ml_pipeline.reduce_dimensions(
    X_train_processed,
    method="umap",
    n_components=2,
    n_neighbors=15
)
```

---

## Best Practices

### Data Quality Checks

**ALWAYS perform these checks before analysis:**

1. **Missing Data**: Check percentage and patterns
2. **Outliers**: Use IQR or Z-score methods
3. **Data Types**: Ensure correct types (int, float, datetime)
4. **Duplicates**: Remove or investigate
5. **Consistency**: Check value ranges and logical constraints

### Statistical Significance

**Guidelines for p-values:**
- p < 0.001: Very strong evidence
- p < 0.01: Strong evidence
- p < 0.05: Moderate evidence (standard threshold)
- p < 0.10: Weak evidence
- p ≥ 0.10: Insufficient evidence

**Effect Size Matters:** Always report effect sizes alongside p-values:
- Cohen's d for t-tests
- Eta-squared for ANOVA
- R-squared for regression

### Model Selection

**Choose based on:**
1. **Problem Type**: Classification vs Regression vs Clustering
2. **Data Size**: Large data → complex models, Small data → simple models
3. **Interpretability**: Linear models > Tree-based > Neural networks
4. **Performance**: Use cross-validation, not just test accuracy
5. **Computational Cost**: Training time vs prediction time

### Cross-Validation Strategy

```python
# K-Fold for general use
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Stratified K-Fold for imbalanced classification
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Time Series Cross-Validation
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)
```

---

## Common Analysis Patterns

### Pattern 1: A/B Test Analysis

```python
# Load data
df_ab = pipeline.load_data("ab_test.csv")

# Statistical test
test_result = tester.t_test(
    group1=df_ab[df_ab["variant"] == "A"]["conversion_rate"],
    group2=df_ab[df_ab["variant"] == "B"]["conversion_rate"],
    alpha=0.05
)

# Interpret
if test_result["p_value"] < 0.05:
    print(f"Variant B performs {test_result['effect_size']:.2%} better")
else:
    print("No significant difference between variants")
```

### Pattern 2: Customer Churn Prediction

```python
# Prepare data
features = ["tenure", "monthly_charges", "total_charges", "num_products"]
target = "churn"

# Train model
ml_pipeline.train_models(
    df[features], df[target],
    models=["logistic_regression", "random_forest", "xgboost"],
    cross_validation=5,
    class_weight="balanced"  # Handle imbalanced classes
)

# Feature importance
importance = ml_pipeline.analyze_feature_importance(best_model, features)
```

### Pattern 3: Sales Forecasting

```python
# Time series analysis
df_sales = df.set_index("date")
ts_analyzer = TimeSeriesAnalyzer()

# Fit multiple models
models = {
    "ARIMA": ts_analyzer.fit_arima(df_sales["sales"], order="auto"),
    "Prophet": ts_analyzer.fit_prophet(df_sales),
    "LSTM": ts_analyzer.fit_lstm(df_sales["sales"])
}

# 3-month forecast
forecasts = {name: ts_analyzer.forecast(model, steps=90)
             for name, model in models.items()}

# Select best model by MAPE
best_forecast = min(forecasts.items(), key=lambda x: x[1]["mape"])
```

---

## Error Handling

```python
try:
    # Analysis code
    result = analyzer.perform_analysis(data)
except ValueError as e:
    # Data validation error
    print(f"Data error: {e}")
    # Suggest: check data types, missing values, outliers
except ConvergenceWarning:
    # Model didn't converge
    # Solution: increase iterations, scale features, try different algorithm
except np.linalg.LinAlgError:
    # Matrix singularity (multicollinearity)
    # Solution: remove correlated features, use regularization
```

---

## Dependencies Installation

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn plotly
pip install xgboost lightgbm catboost  # Gradient boosting
pip install prophet tensorflow  # Time series and deep learning
```

## Sources

- [HTML to PPTX conversion](https://github.com/maximecaruchet/html2pptx)
- [Aspose.Slides for Python](https://products.aspose.com/slides/python-net/)
- [PptxGenJS](https://gitbrent.github.io/PptxGenJS/docs/html-to-powerpoint.html)
