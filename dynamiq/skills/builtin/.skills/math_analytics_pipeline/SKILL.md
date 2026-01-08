---
name: math_analytics_pipeline
version: "2.0.0"
description: Deep mathematical and statistical analysis for data science workflows
tags: [mathematics, statistics, data-science, analytics]
dependencies:
  - pandas
  - numpy
  - scipy
---

# Math Analytics Pipeline Skill

## Quick Analysis Workflow

When analyzing data:

1. **Load and Validate**
   ```python
   import pandas as pd
   df = pd.read_csv('data.csv')
   df.info()  # Check data types and missing values
   ```

2. **Descriptive Statistics**
   ```python
   stats = df.describe()
   correlations = df.corr()
   ```

3. **Key Metrics**
   - Total, average, median for numeric columns
   - Growth rates: `(current - previous) / previous * 100`
   - Percentiles: 25th, 50th, 75th

4. **Insights Generation**
   - Identify top/bottom performers
   - Detect trends (increasing/decreasing)
   - Flag outliers (values > 2 std deviations)

## Example: Sales Analysis

```python
def analyze_sales(file_path):
    import pandas as pd
    import numpy as np

    df = pd.read_csv(file_path)

    # Basic stats
    total_revenue = df['revenue'].sum()
    avg_revenue = df['revenue'].mean()
    growth = df['revenue'].pct_change().mean() * 100

    # Top performers
    top_months = df.nlargest(3, 'revenue')[['month', 'revenue']]

    # Regional analysis
    if 'region' in df.columns:
        regional = df.groupby('region')['revenue'].agg(['sum', 'mean', 'count'])

    return {
        'total': total_revenue,
        'average': avg_revenue,
        'growth': growth,
        'top_months': top_months,
        'regional': regional if 'region' in df.columns else None
    }
```

Use PythonCodeExecutor to run analysis.
