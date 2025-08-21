import pandas as pd
from pandas.api.types import CategoricalDtype
from bivariate import nparametric_tests, interpret_results
from bivariate_viz import bivar_numeric_plot, bivar_nominal_plot, bivar_ordinal_plot, bivar_binary_plot, bivar_datetime_plot
from data_cleaning import clean_data


# load the data
df = clean_data()

numeric_cols = []
ordinal_cols = {}
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
        else:
            nominal_cols.append(col)
    
    elif pd.api.types.is_numeric_dtype(df[col]):
        if df[col].nunique(dropna=True) == 2:
            binary_cols.append(col)
        else:
            numeric_cols.append(col)

    else:
        nominal_cols.append(col)

bivar_numeric_plot(df, numeric_cols)
bivar_nominal_plot(df, nominal_cols)
bivar_ordinal_plot(df, ordinal_cols)
bivar_binary_plot(df, binary_cols)
bivar_datetime_plot(df, datetime_cols)

# drop hiredate and reviewdate (not needed for correlation)
df = df.drop(columns=['HireDate', 'ReviewDate'])

# run the nonparametric tests
attrition_results = nparametric_tests(df)
print(attrition_results)

# interpret results
for _, row in attrition_results.iterrows():
    interpret_results(row['Variable'], row['Test'], row['Statistic'], row['p-value'])