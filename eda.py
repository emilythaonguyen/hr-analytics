import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, normaltest, probplot
import os

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

    plt.figure(figsize=(12, 6))
    sns.countplot(x=col, data=df)
    
    if col in ['Ethnicity', 'EducationField', 'JobRole']:
        # rotate to make space for x labels
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.3)
    else:
        plt.tight_layout()
    
    # Save plot as JPG
    plt.title(f'Distribution of {col}')
    filename = f'Distribution_of_{col}.jpg'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

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
    filename = f"Boxplot_of_{col}.jpg"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Mode: {df[col].mode()[0]}")
    print(f"Skewness: {skew(df[col].dropna()):.2f}")
    print(f"Kurtosis: {kurtosis(df[col].dropna()):.2f}")
    
    # Histogram + KDE
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    filename = f"Distribution_of_{col}.jpg"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
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
    filename = f'QQ-Plot_of_{col}.jpg'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
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
    print(f"Range: {df[col].max() - df[col].min()}")
    print(f"Median: {df[col].median()}")
    print(f"Mode: {df[col].mode()[0]}")
    print(f"Unique dates: {df[col].nunique(dropna=False)}")

    #Counts per year
    year_counts = df[col].dt.year.value_counts().sort_index()
    print("\nCounts per year:")
    print(year_counts)

    # Yearly counts
    dt_yearly = df.set_index(col).resample('Y').size()
    plt.figure(figsize=(14, 7))
    dt_yearly.plot(marker='o')
    plt.title(f"Yearly Count of {col}")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    filename=f'Yearly_Count_of_{col}.jpg'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
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


import scipy.stats as stats

def nparametric_tests(df, target='Attrition'):
    """
    Run nonparametric tests statistical tests between a target variable and all other variables in the DataFrame.

    For numeric variables, performs the Mann-Whitney U test to compare distributions between target groups.
    For categorical variables, preforms the Chi-square test of independence to assess association with the target.
    
    Parameters:
        df (pd.DataFrame): the DataFrame that contains the data.
        target (str): The name of the target variable to test against other columns.
        
    Returns: 
        list: A list of dictionaries containing the variable name, test type, test statistic, and p-value for each variable.
    """
    results = []

    for col in df.columns:
        if col == target:
            continue # skip target column
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # Mann-Whitney U test for numerical vs binary catgorical target
            group1 = df[df[target]==0][col].dropna()
            group2 = df[df[target]==1][col].dropna()
            stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            results.append({'Variable': col, 'Test': 'Mann-Whitney U', 'Statistic': stat, 'p-value': p})
        
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
            # Chi-square test for categorical variables/target
            contigency_table = pd.crosstab(df[col], df[target])
            chi2, p, dof, ex = stats.chi2_contingency(contigency_table)
            results.append({'Variable': col, 'Test': 'Chi-square', 'Statistic': chi2, 'p-value': p})
    
    return pd.DataFrame(results)

attrition_results = nparametric_tests(df)
print(attrition_results)

# use attrition_results to print out interpretations
def interpret_results(variable, test_name, statistic, p_value, alpha=0.05):
    """
    Prints out the interpretation and stats in a readable format.

    Parameters:
    variable (str): Name of the variable being tested.
    test_name (str): Name of the statistical test used.
    statistic (float): Test statistic value.
    p-value (float): P-value from the test.
    alpha (float): Significance level threshold (default is 0.05).
    """
    print(f"---{variable}---")
    print(f"Test: {test_name}")
    print(f"Statistic: {statistic:.4g}")
    print(f"p_value: {p_value:.4g}")
    print(f"Significance level: {alpha} (95% confidence)")
    print(f"Null hypothesis (H0): There is no relationship/difference between {variable} and Attrition.")
    print(f"Alternative hypothesis (Ha): There is a relationship/difference between {variable} and Attrition.")
    if p_value < alpha:
        print(f"\nSince the p-value is less than {alpha}, we reject the null hypothesis and conclude that there is a statistically significant relationship/difference between {variable} and Attrition.\n")
    else:
        print(f"\nSince the p-value is greater than or equal to {alpha}, we fail to reject the null hypothesis and conclude that there is not enough evidence to say there is a statistically significant relationship/difference between {variable} and Attrition.\n")

for idx, row in attrition_results.iterrows():
    interpret_results(row['Variable'], row['Test'], row['Statistic'], row['p-value'])


# check which ones are nonsignificant and significant
significant_vars = []
nonsignificant_vars = []

for _, row in attrition_results.iterrows():
    variable = row['Variable']
    p_value = row['p-value']
    if p_value < 0.05:
        significant_vars.append(variable)
    else:
        nonsignificant_vars.append(variable)

print('Quick Summary:')
print("---Significant Variables:---")
for var in significant_vars:
    print(f"{var}")
print("---Non-Significant Variables:---")
for var in nonsignificant_vars:
    print(f"{var}")