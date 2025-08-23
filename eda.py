import pandas as pd
from data_cleaning import clean_data
from univariate import ql_stats, qn_stats, dt_stats

def run_eda(dataset_type='all'):
    """
    Perform exploratory data analysis (EDA) on a dataset.
    Calls the correct function based on the type of data.

    Parameters
    ----------
    dataset_type : str, default = 'all'
        The name of which cleaned dataset we want to do eda on
    """
    # load the dataset
    df = clean_data(dataset_type)

    for col in df.columns:
        # datetime columns
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dt_stats(df, col)
        # quantitative columns
        elif pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = df[col].nunique()
            if unique_vals == 2:
                ql_stats(df, col)
            else:
                qn_stats(df, col)
        # qualitative columns
        else:
            ql_stats(df, col)

run_eda('hr_only')
run_eda('with_reviews')