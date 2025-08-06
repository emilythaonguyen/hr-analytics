import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
                    'MaritalStatus', 'StockOptionLevel'
                    ]
for col in categorical_cols:
    df_combined[col] = df_combined[col].astype('category')

# one-hot encode categorical variables
df_encoded = pd.get_dummies(df_combined, drop_first=True)

# Date columns
df_combined['HireDate'] = pd.to_datetime(df_combined['HireDate'])
df_combined['ReviewDate'] = pd.to_datetime(df_combined['ReviewDate'])

# Drop unused columns
drop_columns = ['FirstName', 'LastName', 'PerformanceID']

df_combined = df_combined.drop(drop_columns, axis=1)

# Save cleaned data
df_combined.to_csv('cleaned_employee_data.csv', index=False)

# EDA
# Set visual style
sns.set(style='whitegrid', palette='dark')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the dataset
df = df_combined.copy()

print(df.shape)
df.info()
df.describe()

# Create function to print out basic data stats 
# for qualitative variables
def qual_summary(df, column):
    """
    Prints the count and percentage of each category in the specified column.
    Parameters: 
    df (DataFrame): The DataFrame containing the data.
    column (str): The name of the column to summarize.
    """
    total = len(df)
    counts = df[column].value_counts()
    percentages = counts / total * 100

    print(f"\nColumn: {column}")
    for category, count in counts.items():
        print(f"{category}: {count} ({percentages[category]:.2f}%)")

# Same concept but for qualitative variables
def quant_summary(df, column):
    """
    Prints summary stats for a numeric column:
    (mean, median, mode, min, max, and range)

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    column (str): The name of the numeric column to summarize.
    """
    series = df[column].dropna() # for missing values
    mean = series.mean()
    median = series.median()
    mode = series.mode().values[0]
    min_val = series.min()
    max_val = series.max()
    range_val = max_val - min_val
    print(f"\nSummary of '{column}:")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"Range: {range_val}")

# Attrition Count
sns.countplot(x='Attrition', data=df)
plt.title('Employee Attrition Count')
plt.show()
qual_summary(df, 'Attrition')

# Distribution of Age
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()
quant_summary(df, 'Age')

# Gender Count
sns.countplot(x='Gender', data=df)
plt.title('Employee Gender Count')
plt.show()
qual_summary(df, 'Gender')

# Department Count
sns.countplot(x='Department', data=df)
plt.title('Department Count')
plt.show()
qual_summary(df, 'Department')

# 

# Distribution of Salary
sns.histplot(df['Salary'], bins=20, kde=True)
plt.title('Distribution of Salary')
plt.show()
quant_summary(df, 'Salary')

# Distribution of Hire Date
sns.histplot(df['HireDate'], bins=20, kde=True)
plt.title('Distribution of Hire Date')
plt.show()
quant_summary(df, 'HireDate')