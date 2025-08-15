import pandas as pd
from bivariate import nparametric_tests, interpret_results


# load the data
df = pd.read_csv('cleaned_employee_data.csv')

# run the nonparametric tests
attrition_results = nparametric_tests(df)
print(attrition_results)

# interpret results
for _, row in attrition_results.iterrows():
    interpret_results(row['Variable'], row['Test'], row['Statistic'], row['p-value'])