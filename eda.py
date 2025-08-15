import pandas as pd
from pandas.api.types import CategoricalDtype
from univariate import ql_stats, qn_stats, dt_stats

def run_eda():
    """
    Perform exploratory data analysis (EDA) on a dataset.
    Calls the correct function based on the type of data.

    Parameters
    ----------
    df: DataFrame
        The cleaned dataset to analyze.

    Returns
    -------
    None
    """
    # load the dataset
    df = pd.read_csv('cleaned_employee_data.csv')

    # drop Employee ID column from the dataset
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

    # convert date columns back to datetime
    df['HireDate'] = pd.to_datetime(df['HireDate'])
    df['ReviewDate'] = pd.to_datetime(df['ReviewDate'])

    # make binary columns categorical
    df['Attrition'] = df['Attrition'].astype('category')
    df['OverTime'] = df['OverTime'].astype('category')

    # loops through all columns 
    # and runs univariate functions for each data type.
    for col in df.columns:
        # datetime columns
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dt_stats(df, col)
        # quantitative columns
        elif pd.api.types.is_numeric_dtype(df[col]):
            qn_stats(df, col)
        # qualitative columns
        else:
            ql_stats(df, col)

if __name__ == "__main__":
    run_eda()