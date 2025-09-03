# hr-analytics
Analyzing employee data through EDA and correlation analysis to find the factors that attribute to attrition to eventually building a predictive model to identify at-risk employees that can be used for future insights.

This project uses two datasets, "Employee.csv" and "Performance.csv".
Both can be found here: [HR Analytics Employee Attrition and Performance](https://www.kaggle.com/datasets/mahmoudemadabdallah/hr-analytics-employee-attrition-and-performance)

## Methods
1. **Exploratory Data Analysis (EDA):**
- Examined numeric variables like age, salary, tenure, and promotion history
- Explored categorical variables such as gender, job role, department, business travel, and ethnicity
2. **Visualizations**
- Boxplots, scatter plots, pairplots, heatmaps, and stacked bar charts were used to pinpoint patterns and correlations
3. **Statistical Tests**
- Conducted nonparametric tests to determine which variables significantly relate to attrition
- Results showed that variables such as Years at Company, Years in Current Role, Years Since Last Promotion were the most significant
4. **Predictive Modeling:**
- Built Logistic Regression and Random Forest models to predict attrition

## How to Use

1. **Install dependencies:**

```bash
pip install pandas matplotlib seaborn scipy numpy scikit-learn imbalanced-learn
```

2.  **Run Jupyter Notebook (for full report)**:

```bash
jupyter notebook hr_analytics.ipynb
```

3. **Run all scripts (to generate plots, tests, and models automatically)**:

```bash
python eda.py
python correlation.py
python attrition_modeling.py
```