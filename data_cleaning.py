# import libraries
import pandas as pd

def clean_data(dataset_type="all"):
    """
    Load, merge, and clean employee and performance datasets.
    Can return one of three modified versions of the original dataset.

    Parameters
    ----------
    dataset_type : str, default = "all"
        "hr_only" : HR data only, no performance columns (1470 employees)
        "with_reviews": HR + performance, only employees with reviews (1280 employees)
        "all" : HR + performance, includes all employees (1470 employees)
    
    Returns
    -------
    DataFrame
        Main dataframe for analyzation, includes with or without performance columns
        depending on dataset_type.      
    """
    # load the data
    df_employee = pd.read_csv('data/Employee.csv')
    df_performance = pd.read_csv('data/PerformanceRating.csv')

    # change ReviewDate to datetime
    df_performance["ReviewDate"] = pd.to_datetime(df_performance["ReviewDate"])
    # sort performance by employee + review date
    df_performance = df_performance.sort_values(["EmployeeID", "ReviewDate"])
    # keep the most recent review for each employee
    df_performance = df_performance.groupby("EmployeeID").tail(1)
    
    # merge both datasets based on dataset_type
    if dataset_type == 'hr_only':
        df = df_employee.copy()
    elif dataset_type == "with_reviews":
        # inner join to keep only employees with reviews
        df = pd.merge(df_employee, df_performance, on='EmployeeID', how='inner')
    elif dataset_type == 'all':
        # left join to keep all employees with & without reviews
        df = pd.merge(df_employee, df_performance, on='EmployeeID', how='left')
    else:
        raise ValueError("Invalid dataset_type. Must be in ['hr_only', 'with_reviews', 'all']")

    # map binary columns
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

    # mapping ordinal columns
    satisfaction_map = {
        1: 'Very Dissatisfied',
        2: 'Dissatisfied',
        3: 'Neutral',
        4: 'Satisfied',
        5: 'Very Satisfied'
    }

    performance_map = {
        1: 'Unacceptable',
        2: 'Needs Improvement',
        3: 'Meets Expectations',
        4: 'Exceeds Expectation',
        5: 'Above and Beyond'
    }

    education_map = {
        1: 'No Formal Qualifications',
        2: 'High School',
        3: 'Bachelors',
        4: 'Masters',
        5: 'Doctorate'
    }

    # ensure Education is present for mapping
    if 'Education' not in df.columns:
        # if somehow lost during merge, bring it from df_employee
        df = df.merge(df_employee[['EmployeeID', 'Education']], on='EmployeeID', how='left')

   # define ordinal mappings by column
    ordinal_mappings = {
        'EnvironmentSatisfaction': satisfaction_map,
        'JobSatisfaction': satisfaction_map,
        'RelationshipSatisfaction': satisfaction_map,
        'WorkLifeBalance': satisfaction_map,
        'SelfRating': performance_map,
        'ManagerRating': performance_map,
        'Education': education_map
    }

    # apply mappings
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            df[col] = pd.Categorical(df[col], categories=list(mapping.values()), ordered=True)
    
    # convert the rest of the categorical columns
    categorical_cols = [ 
        'Gender', 'BusinessTravel', 'Department', 'State',
          'Ethnicity', 'EducationField', 
        'JobRole', 'MaritalStatus', 'StockOptionLevel']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # make sure any date columns are in datetime
    df['HireDate'] = pd.to_datetime(df['HireDate'])

    # fix typos in EducationField
    df['EducationField'] = df['EducationField'].str.strip().str.title()
    df['EducationField'] = df['EducationField'].replace({
        'Marketing ': 'Marketing',
        'Marketting': 'Marketing'
    })
    # make sure EducationField is categorical
    df['EducationField'] = df['EducationField'].astype('category')

    # drop unused columns 
    if dataset_type == 'hr_only':
        df = df.drop(columns=['FirstName', 'LastName'])
    else:
        df = df.drop(columns=['FirstName', 'LastName', 'PerformanceID'])

    # check summary before saving
    print("\nSummary:")
    df.info()
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nPreview:")
    print(df.head())
    
    # save cleaned data (with EmployeeID in tact)
    df.to_csv(f'data/cleaned_{dataset_type}_employee_data.csv', index=False)
    print(f"Data cleaned and saved as cleaned_{dataset_type}_employee_data.csv\n")

    # drop EmployeeId for analysis
    df = df.drop(columns=['EmployeeID'])

    return df

if __name__ == "__main__":
    clean_data('hr_only')
    clean_data('with_reviews')
    clean_data('all')