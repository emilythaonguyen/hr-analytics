import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, normaltest, probplot

# Load Dataset
df_employee = pd.read_csv('Employee.csv')
df_performance = pd.read_csv('PerformanceRating.csv')

# Merge datasets
df_combined = pd.merge(df_employee, df_performance, on='EmployeeID')
print(df_combined.shape)
df_combined.head()

# Missing values
print(df_combined.isnull().sum())

# Map binary values
df_combined['Attrition'] = df_combined['Attrition'].map({'Yes':1, 'No': 0})
df_combined['OverTime'] = df_combined['OverTime'].map({'Yes': 1, 'No':0})


# Convert ordinal values
from pandas.api.types import CategoricalDtype
levels = CategoricalDtype(
    categories=[1,2,3,4,5],
    ordered=True
)

cols_to_convert = [
    'EnvironmentSatisfaction',
    'JobSatisfaction',
    'RelationshipSatisfaction',
    'WorkLifeBalance',
    'SelfRating',
    'ManagerRating',
    'Education',
]
for col in cols_to_convert:
    df_combined[col] = df_combined[col].astype(levels)

# convert remaining categorical columns
categorical_cols = ['BusinessTravel', 'Department', 'State', 'Ethnicity', 'EducationField', 'JobRole', 
                    'MaritalStatus', 'StockOptionLevel', 'TrainingOpportunitiesWithinYear',
                    'TrainingOpportunitiesTaken'
                    ]
for col in categorical_cols:
    df_combined[col] = df_combined[col].astype('category')

# Date columns
df_combined['HireDate'] = pd.to_datetime(df_combined['HireDate'])
df_combined['ReviewDate'] = pd.to_datetime(df_combined['ReviewDate'])

# Fixing 'Marketing' typo
df_combined['EducationField'] = df_combined['EducationField'].str.strip().str.title()

# Merge Marketing variants
df_combined['EducationField'] = df_combined['EducationField'].replace({
    'Marketing ': 'Marketing',
    'Marketting': 'Marketing' 
})

# Drop unused columns
drop_columns = ['FirstName', 'LastName', 'PerformanceID']

df_combined = df_combined.drop(drop_columns, axis=1)

# Save cleaned data
df_combined.to_csv('cleaned_employee_data.csv', index=False)

# EDA
# Set visual style
sns.set(style='whitegrid', palette='hls')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the dataset
df = df_combined.copy()

print(df.shape)
df.info()
df.describe()

# Function to print out summary for qualitative variables 
# and proportion along w/plots.
def ql_stats(df, col):
    """
    Prints the summary and percentage of each category 
    in the specified column w/count plots.

    Parameters: 
    df (DataFrame): The DataFrame containing the data.
    col (str): The name of the column to summarize.
    """
    print(f"\n--- Categorical Summary: {col} ---")
    counts = df[col].value_counts(dropna=False)
    percentages = df[col].value_counts(normalize=True, dropna=False) * 100

    summary = pd.DataFrame({
        'Count': counts,
        'Percentages': percentages.round(2)
    })
    print(summary)
    print(f"Unique categories: {df[col].nunique(dropna=False)}")
    print(f"Most frequent: {df[col].mode()[0]}")

    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()

# Same concept but for quantitative variables
def qn_stats(df, col):
    """
    Prints out summary for quantitative columns and creates 
    histogram + KDE plots.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    col (str): The name of the numeric column to summarize.
    """
    print(f"\n--- Numerical Summary: {col} ---")
    desc = df[col].describe()
    print(desc)

    # Boxplot
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.show()

    print(f"Mode: {df[col].mode()[0]}")
    print(f"Skewness: {skew(df[col].dropna()):.2f}")
    print(f"Kurtosis: {kurtosis(df[col].dropna()):.2f}")
    
    # Histogram + KDE
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

    # Normality Test
    stat, p=normaltest(df[col])
    print(f"\nD'Agostino and Pearson Test:")
    print(f" Statistic = {stat:.4f}, p-value = {p:.4f}")
    if p < 0.05:
        print("Data is not normally distributed.")
    else:
        print("Data is normally distributed.")

    # QQ Plot
    plt.figure(figsize=(6,6))
    probplot(df[col], dist='norm', plot=plt)
    plt.title(f'QQ-Plot of {col}')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel('Sample Quantiles')
    plt.grid(True)
    plt.show()

# Same thing but with datetime columns
def dt_stats(df, col):
    """
    Summarizes datetime columns and plots time series trends.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    col (str): The name of the numeric column to summarize.
    """
    print(f"\n--- Datetime Summary: {col} ---")
    print(f"Min date: {df[col].min()}")
    print(f"Max date: {df[col].max()}")
    print(f"Unique dates: {df[col].nunique(dropna=False)}")

    # Yearly counts
    dt_yearly = df.set_index(col).resample('Y').size()
    plt.figure(figsize=(14, 7))
    dt_yearly.plot(marker='o')
    plt.title(f"Yearly Count of {col}")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Drop Employee ID from the columns
df = df.drop(columns=['EmployeeID'])

# Temporarily turns Attrition and OverTime variables as categories
df['Attrition'] = df['Attrition'].astype('category')
df['OverTime'] = df['OverTime'].astype('category')

# Loop through all columns and run stats, tests, and plots
for col in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        dt_stats(df, col)
    elif pd.api.types.is_numeric_dtype(df[col]):
        qn_stats(df, col)
    else:
        ql_stats(df, col)