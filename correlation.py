import pandas as pd
from pandas.api.types import CategoricalDtype
from bivariate import nparametric_tests, interpret_results
from bivariate_viz import bivar_numeric_plot, bivar_nominal_plot, bivar_ordinal_plot, bivar_binary_plot, bivar_datetime_plot
from data_cleaning import clean_data


# load the data
df = clean_data()

# # drop EmployeeID from the dataset
# df = df.drop(columns=['EmployeeID'])

# # recast types for nonparametric testing
# # ordinal columns
# levels = CategoricalDtype(categories=[1,2,3,4,5], ordered=True)
# ordinal_cols = [
# 'EnvironmentSatisfaction', 
# 'JobSatisfaction', 
# 'RelationshipSatisfaction', 
# 'WorkLifeBalance', 
# 'SelfRating', 
# 'ManagerRating', 
# 'Education']

# for col in ordinal_cols:
#     df[col] = df[col].astype(levels)

# # retyping the non-ordinal category columns
# categorical_cols = ['BusinessTravel', 'Department', 'State', 'Ethnicity', 'EducationField', 
#     'JobRole', 'MaritalStatus', 'StockOptionLevel', 'TrainingOpportunitiesWithinYear',
#     'TrainingOpportunitiesTaken']

# for col in categorical_cols:
#     df[col] = df[col].astype('category')

# # change date variables to datetime
# date_cols = ['HireDate', 'ReviewDate']
# for col in date_cols:
#     df[col] = pd.to_datetime(df[col])

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

# unique_vals = df[col].nunique()
#         if unique_vals == 2:
#             binary_cols.append(col)
# print(ordinal_cols)
print(binary_cols)
# print(numeric_cols)
# print(nominal_cols)

bivar_numeric_plot(df, numeric_cols)
bivar_nominal_plot(df, nominal_cols)
bivar_ordinal_plot(df, ordinal_cols)
bivar_binary_plot(df, binary_cols)
bivar_datetime_plot(df, datetime_cols)

# run the nonparametric tests
attrition_results = nparametric_tests(df)
print(attrition_results)

# interpret results
for _, row in attrition_results.iterrows():
    interpret_results(row['Variable'], row['Test'], row['Statistic'], row['p-value'])

# a list with only the significant variables
# significant_vars = attrition_results.query("Significant == True")['Variable'].tolist()

# # a list separating numeric and categorical variables
# numeric_vars = []
# categorical_vars = []

# for col in significant_vars:
#     if pd.api.types.is_numeric_dtype(df[col]) or \
#     (pd.api.types.is_categorical_dtype(df[col]) and df[col].cat.ordered):
#         numeric_vars.append(col)
#     else:
#         categorical_vars.append(col)

# # plots directories
# os.makedirs('plots/pairplots', exist_ok=True)
# os.makedirs('plots/heatmaps', exist_ok = True)
# os.makedirs('plots/stackedbars', exist_ok = True)

# # paths for pairplot and heatmap
# pairplot_path = os.path.join('plots/pairplots', 'pairplot.png')
# heatmap_path = os.path.join('plots/heatmaps', 'heatmap.png')

# if numeric_vars:
#     plot_pairplot(df, numeric_vars, hue='Attrition', save_path=pairplot_path)
#     plot_correlation_heatmap(df, numeric_vars, save_path=heatmap_path)

# for cat_var in categorical_vars:
#     stackedbar_path = os.path.join('plots/stackedbars', f'{cat_var}_stackedbar.png')
#     plot_stacked_bar(df, cat_var, target='Attrition', save_path=stackedbar_path)