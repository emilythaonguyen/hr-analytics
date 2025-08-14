import pandas as pd
from pandas.api.types import CategoricalDtype
from univariate import ql_stats, qn_stats, dt_stats


def run_eda():
    # Load the dataset
    df = pd.read_csv('cleaned_employee_data.csv')

    # Drop Employee ID from the columns
    df = df.drop(columns=['EmployeeID'])

    # recast types for EDA
    # ordinal columns
    levels = CategoricalDtype(categories=[1,2,3,4,5], ordered=True)
    ordinal_cols = [
    'EnvironmentSatisfaction', 
    'JobSatisfaction', 
    'RelationshipSatisfaction', 
    'WorkLifeBalance', 
    'SelfRating', 
    'ManagerRating', 
    'Education']

    for col in ordinal_cols:
        df[col] = df[col].astype(levels)
    
    # retyping the non-ordinal category columns
    categorical_cols = ['BusinessTravel', 'Department', 'State', 'Ethnicity', 'EducationField', 
        'JobRole', 'MaritalStatus', 'StockOptionLevel', 'TrainingOpportunitiesWithinYear',
        'TrainingOpportunitiesTaken']
    
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # date columns
    df['HireDate'] = pd.to_datetime(df['HireDate'])
    df['ReviewDate'] = pd.to_datetime(df['ReviewDate'])

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

if __name__ == "__main__":
    run_eda()