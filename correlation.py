import pandas as pd
from bivariate import nparametric_tests, interpret_results
from bivariate_viz import plot_pairplot, plot_correlation_heatmap, plot_stacked_bar
import os

# load the data
df = pd.read_csv('cleaned_employee_data.csv')

# run the nonparametric tests
attrition_results = nparametric_tests(df)
print(attrition_results)

# interpret results
for _, row in attrition_results.iterrows():
    interpret_results(row['Variable'], row['Test'], row['Statistic'], row['p-value'])

# a list with only the significant variables
significant_vars = attrition_results.query("Significant == True")['Variable'].tolist()

# a list separating numeric and categorical variables
numeric_vars = []
categorical_vars = []

for col in significant_vars:
    if pd.api.types.is_numeric_dtype(df[col]) or \
    (pd.api.types.is_categorical_dtype(df[col]) and df[col].cat.ordered):
        numeric_vars.append(col)
    else:
        categorical_vars.append(col)

# plots directories
os.makedirs('plots/pairplots', exist_ok=True)
os.makedirs('plots/heatmaps', exist_ok = True)
os.makedirs('plots/stackedbars', exist_ok = True)

# paths for pairplot and heatmap
pairplot_path = os.path.join('plots/pairplots', 'pairplot.png')
heatmap_path = os.path.join('plots/heatmaps', 'heatmap.png')

if numeric_vars:
    plot_pairplot(df, numeric_vars, hue='Attrition', save_path=pairplot_path)
    plot_correlation_heatmap(df, numeric_vars, save_path=heatmap_path)

for cat_var in categorical_vars:
    stackedbar_path = os.path.join('plots/stackedbars', f'{cat_var}_stackedbar.png')
    plot_stacked_bar(df, cat_var, target='Attrition', save_path=stackedbar_path)
