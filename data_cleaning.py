# import libraries
import pandas as pd
from pandas.api.types import CategoricalDtype

def clean_data():
    """
    Load, merge, and clean employee and performance datasets, 
    then save the cleaned dataset to a CSV file

    Steps
    -----
    1. Load data from 'Employee.csv' and 'PerfromanceRating.csv'
    2. Merge datasets on 'EmployeeID'
    3. Map binary columns ('Yes'/'No' -> 1/0)
    4. Convert ordinal columns to ordered categorical types
    5. Convert nominal categorical columns to 'category' dtype
    6. Convert date columns to datetime
    7. Fix typos in 'EducationField'
    8. Drop unused columns
    9. Save to 'cleaned_employee_data.csv'

    Returns
    -------
    None
    """
    # load the data
    df_employee = pd.read_csv('Employee.csv')
    df_performance = pd.read_csv('PerformanceRating.csv')

    # merge both datasets
    df = pd.merge(df_employee, df_performance, on='EmployeeID')
    
    # map binary columns
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

    # convert ordinal columns
    levels = CategoricalDtype(categories=[1,2,3,4,5], ordered=True)
    cols_to_convert = [
        'EnvironmentSatisfaction', 
        'JobSatisfaction', 'RelationshipSatisfaction', 
        'WorkLifeBalance', 'SelfRating', 
        'ManagerRating', 'Education',
    ]
    for col in cols_to_convert:
        df[col] = df[col].astype(levels)

    # convert the rest of the categorical columns
    categorical_cols = [
        'BusinessTravel', 'Department', 'State', 'Ethnicity', 'EducationField', 
        'JobRole', 'MaritalStatus', 'StockOptionLevel', 'TrainingOpportunitiesWithinYear',
        'TrainingOpportunitiesTaken'
    ]
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # make sure date columns are datetime
    df['HireDate'] = pd.to_datetime(df['HireDate'])
    df['ReviewDate'] = pd.to_datetime(df['ReviewDate'])

    # fix typos in EducationField
    # realized this error when i was running the summary for education field
    df['EducationField'] = df['EducationField'].str.strip().str.title()
    df['EducationField'] = df['EducationField'].replace({
        'Marketing ': 'Marketing',
        'Marketting': 'Marketing'
    })

    # drop unused columns 
    df = df.drop(['FirstName', 'LastName', 'PerformanceID'], axis=1)
    

    # check summary before saving
    print("\nSummary:")
    df.info()
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nPreview:")
    print(df.head())

    # save cleaned data
    df.to_csv('cleaned_employee_data.csv', index=False)
    print("Data cleaned and saved as cleaned_employee_data.csv\n")

    df = df.drop(columns=['EmployeeID'])

    return df

if __name__ == "__main__":
    clean_data()