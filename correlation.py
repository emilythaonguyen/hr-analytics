import pandas as pd
from bivariate import nparametric_tests, interpret_results, significant_list



df = pd.read_csv('cleaned_employee_data.csv')

attrition_results = nparametric_tests(df)
print(attrition_results)

# interpret results
for _, row in attrition_results.iterrows():
    interpret_results(row['Variable'], row['Test'], row['Statistic'], row['p-value'])

# quick summary
significant_list(attrition_results, alpha=0.05, save=True)