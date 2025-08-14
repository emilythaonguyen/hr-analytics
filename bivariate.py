import pandas as pd
from pandas.api.types import CategoricalDtype
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
    # recast types for nonparametric testing
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
    print(f"Significance level: {alpha} ({(1-alpha)*100}% confidence)")
    print(f"Null hypothesis (H0): There is no relationship/difference between {variable} and Attrition.")
    print(f"Alternative hypothesis (Ha): There is a relationship/difference between {variable} and Attrition.")
    if p_value < alpha:
        print(f"\nSince the p-value is less than {alpha}, we reject the null hypothesis and conclude that there is a statistically significant relationship/difference between {variable} and Attrition.\n")
    else:
        print(f"\nSince the p-value is greater than or equal to {alpha}, we fail to reject the null hypothesis and conclude that there is not enough evidence to say there is a statistically significant relationship/difference between {variable} and Attrition.\n")

def significant_list(results_df, alpha=0.05, save=True):
    """
    Prints a quick summary of significant and non-significant variables
    Optionally saves the lists to CSV files.
    """
    significant = results_df[results_df['p-value'] < alpha]['Variable'].tolist()
    nonsignificant = results_df[results_df['p-value'] >= alpha]['Variable'].tolist()

    print("\nQuick Summary:")
    print("--- Significant Variables ---")
    for var in significant:
        print(var)
    print("--- Non-Significant Variables ---")
    for var in nonsignificant:
        print(var)

    if save:
        pd.DataFrame({'Significant Variables': significant}).to_csv('significant_variables.csv', index=False)
        pd.DataFrame({'Non-Significant Variables': nonsignificant}).to_csv('nonsignificant_variables.csv', index=False)
        print("\nCSV files saved: significant_variables.csv & nonsignificant_variables.csv")