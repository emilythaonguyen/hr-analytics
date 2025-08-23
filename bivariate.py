import pandas as pd
import scipy.stats as stats


def nparametric_tests(df, target='Attrition', alpha=0.05, ordinal_cols=None):
    """
    Run a specific nonparametric test based on the dtype of the variable.
    numeric variables: Mann-Whitney 
    categorical variables: Chi-Square
    ordinal (if present): Spearman

    Parameters
    ----------
    df : DataFrame
        the DataFrame that contains the data to test on.
    target : str, default='Attrition'
        The binary target variable being tested for association.
    alpha : float
        Significance level threshold.
    ordinal_cols = list, optional
        Names of columns that are ordinal
    
    Returns
    -------
    DataFrame
        Summary of variables, tests used, test statistics, p-values, and significance,
        sorted from most significant to least significant (by p-values).
    """
    results = []
    
    if ordinal_cols is None:
        ordinal_cols = []
    
    for col in df.columns:
        if pd.api.types.is_datetime64_dtype(df[col]) or col == target:
            continue # skip target and datetime columns

        # for numeric and ordinal columns
        if pd.api.types.is_numeric_dtype(df[col]):
            group0 = df.loc[df[target] == 0, col].dropna()
            group1 = df.loc[df[target] == 1, col].dropna()

            if len(group0) > 0 and len(group1) > 0:
                if ordinal_cols is not None and col in ordinal_cols:
                    # ordinal → spearman
                    corr, p = stats.spearmanr(df[col], df[target])
                    results.append({
                        'Variable': col,
                        'Test': 'Spearman',
                        'Statistic/Corr': corr,
                        'p-value': p,
                        'Significant': p < alpha
                })
                else:
                    # numeric → mann-whitney
                    stat, p = stats.mannwhitneyu(group0, group1, alternative='two-sided')
                    results.append({
                        'Variable': col,
                        'Test': 'Mann-Whitney U',
                        'Statistic/Corr': stat,
                        'p-value': p,
                        'Significant': p < alpha
                    })

        else:
            # chi-square test for categorical variables
            contingency_table = pd.crosstab(df[col], df[target])
            chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
            results.append({'Variable': col, 
                            'Test': 'Chi-square', 
                            'Statistic/Corr': chi2, 
                            'p-value': p,
                            'Significant': p < alpha
                            })
            
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(by='p-value', ascending=True).reset_index(drop=True)

    return results_df

def interpret_results(variable, test_name, statistic, p_value, alpha=0.05):
    """
    Interpret statistical test results for a given variable.

    Parameters
    ----------
    variable : str
        Name of the variable tested
    test_name : str
        Name of the statistic test used
    statistic : float
        Test statistic value
    p_value : float
        p-value from the test
    alpha : float, default=0.05
        Significance level threshold.
    """
    print(f"--- {variable} ---")
    print(f"Test: {test_name}")
    print(f"Statistic/Corr: {statistic:.4g}")
    print(f"p_value: {p_value:.4g}")
    print(f"Significance level: {alpha} ({(1-alpha)*100}% confidence)")
    print(f"Null hypothesis (H0): There is no relationship/difference between {variable} and Attrition.")
    print(f"Alternative hypothesis (Ha): There is a relationship/difference between {variable} and Attrition.")
    if p_value < alpha:
        print(f"\nSince the p-value is less than {alpha}, we reject the null hypothesis and conclude that there is a statistically significant relationship/difference between {variable} and Attrition.\n")
    else:
        print(f"\nSince the p-value is greater than or equal to {alpha}, we fail to reject the null hypothesis and conclude that there is not enough evidence to say there is a statistically significant relationship/difference between {variable} and Attrition.\n")

