import pandas as pd
from pandas.api.types import CategoricalDtype
from bivariate import nparametric_tests, interpret_results
from bivariate_viz import bivar_numeric_plot, bivar_categorical_plot, bivar_spearman_plot, bivar_binary_plot, bivar_datetime_plot
from data_cleaning import clean_data



# load the data
df = clean_data('all')

numeric_cols = []
ordinal_cols = {}
spearman_cols = []
nominal_cols = []
datetime_cols = []
binary_cols = []

for col in df.columns:
    if col == 'Attrition':
        continue

    if pd.api.types.is_datetime64_any_dtype(df[col]):
        datetime_cols.append(col)
    
    elif isinstance(df[col].dtype, CategoricalDtype):
        if df[col].dtype.ordered:
            ordinal_cols[col] = list(df[col].cat.categories)
            df[col] = df[col].cat.codes
            spearman_cols.append(col)
        else:
            nominal_cols.append(col)
    
    elif pd.api.types.is_numeric_dtype(df[col]):
        if df[col].nunique(dropna=True) == 2:
            binary_cols.append(col)
        else:
            numeric_cols.append(col)
            spearman_cols.append(col)

    else:
        nominal_cols.append(col)

bivar_numeric_plot(df, numeric_cols)
bivar_categorical_plot(df, nominal_cols)
bivar_categorical_plot(df, ordinal_cols)
bivar_spearman_plot(df, spearman_cols)
bivar_binary_plot(df, binary_cols)
bivar_datetime_plot(df, datetime_cols)

df_hr_only = clean_data('hr_only') 
df_with_reviews = clean_data('with_reviews')

# convert ordinal columns to codes for nonparametric tests
ordinal_columns = []
for col in df_with_reviews.columns:
    if isinstance(df_with_reviews[col].dtype, CategoricalDtype) and df_with_reviews[col].dtype.ordered:
        df_with_reviews[col] = df_with_reviews[col].cat.codes
        ordinal_columns.append(col)

# run the nonparametric tests
# for hr_only features
results_hr = nparametric_tests(df_hr_only, target='Attrition', alpha=0.05, ordinal_cols=None)
print(results_hr)

# for hr + performance features
results_perf = nparametric_tests(df_with_reviews, target='Attrition', alpha=0.05, ordinal_cols=ordinal_columns)
print(results_perf)
# interpret results
# hr only
for _, row in results_hr.iterrows():
    interpret_results(row['Variable'], row['Test'], row['Statistic/Corr'], row['p-value'])
# hr + performance
for _, row in results_perf.iterrows():
    interpret_results(row['Variable'], row['Test'], row['Statistic/Corr'], row['p-value'])
