import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df_employee = pd.read_csv('Employee.csv')
df_performance = pd.read_csv('PerformanceRating.csv')

# Check the columns to identify a join key
print("Employee Columns:", df_employee.columns.tolist())
print("Performance Columns:", df_performance.columns.tolist())

# Merge datasets
df_combined = pd.merge(df_employee, df_performance, on='EmployeeID')
print(df_combined.shape)
df_combined.head()

# Display columns
df_combined.info()
df_combined.describe(include='all')

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
sns.set(style='whitegrid', palette='Set2')
plt.rcParams['figure.figsize'] = (10, 6)

# Load the dataset
df = df_combined.copy()

# Attrition Count
print("\nAttrition Value Counts:\n", df['Attrition'].value_counts())

sns.countplot(x='Attrition', data=df)
plt.title('Employee Attrition Count')
plt.show()

# Distribution of Age
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Department Count
sns.countplot(x='Department', data=df)
plt.title('Department Count')
plt.show()

# Distribution of Salary
sns.histplot(df['Salary'], bins=20, kde=True)
plt.title('Distribution of Salary')
plt.show()

# Distribution of Hire Date
sns.histplot(df['HireDate'], bins=20, kde=True)
plt.title('Distribution of Hire Date')
plt.show()


# Attrition vs Department
sns.countplot(x='Department', hue='Attrition', data=df)
plt.title('Attrition by Department')
plt.xticks(rotation=45)
plt.show()

# Attrition vs Gender
sns.countplot(x='Gender', hue='Attrition', data=df)
plt.title('Attrition by Gender')
plt.xticks(rotation=45)
plt.show()

# Attrition vs Education Level
sns.countplot(x='Education', hue='Attrition', data=df)
plt.title('Attrition by Education')
plt.xticks(rotation=45)
plt.show()

# Attrition vs OverTime
sns.countplot(x='OverTime', hue='Attrition', data=df)
plt.title('Attrition by Overtime')
plt.xticks(rotation=45)
plt.show()

# Attrition vs Salary
g = sns.FacetGrid(df, col='Attrition', height=4, aspect=1.2)
g.map(sns.histplot, 'Salary', bins=20, kde=True, color='skyblue')
g.set_axis_labels("Salary", "Count")
plt.suptitle('Salary Distribution by Attrition', y=1.05)
plt.show()

# Attrition vs Job Satisfaction
sns.countplot(x='JobSatisfaction', hue='Attrition', data=df)
plt.title('Attrition by Job Satisfaction')
plt.xlabel('Job Satisfaction (1 = Very Dissatisfied, 5 = Very Satisfied)')
plt.ylabel('Number of Employees')
plt.legend(title='Attrition', labels=['Stayed', 'Left'])
plt.show()

# Attrition vs Relationship Satisfaction
sns.countplot(x='RelationshipSatisfaction', hue='Attrition', data=df)
plt.title('Attrition by Relationship Satisfaction')
plt.xlabel('Relationship Satisfaction (1 = Very Dissatisfied, 5 = Very Satisfied)')
plt.ylabel('Number of Employees')
plt.legend(title='Attrition', labels=['Stayed', 'Left'])
plt.show()

# Attrition vs Work Life Balance
sns.countplot(x='WorkLifeBalance', hue='Attrition', data=df)
plt.title('Attrition vs Work Life Balance')
plt.xlabel('WorkLifeBalance')
plt.ylabel('Number of Employees')
plt.legend(title="Attrition", labels=['Stayed', 'Left'])
plt.show()

# Attrition vs HireDate
g = sns.FacetGrid(df, col='Attrition', height=4, aspect=1.2)
g.map(sns.histplot, 'HireDate', bins=20, kde=True, color='skyblue')
g.set_axis_labels("HireDate", "Count")
plt.suptitle('HireDate by Attrition', y=1.05)
plt.show()

# 