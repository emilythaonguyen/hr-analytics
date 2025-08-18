import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def bivar_numeric_plot(df, variables, target='Attrition'):
    """
    Generate plots to visualize dataset correlation 
    between numeric variables against the target variable.

    Parameters
    ----------
    df : DataFrame
        The DataFrame that contains the data.
    variables : list
        List containing variables to plot against the target.
    target : str, default = 'attrition'
        The variable being compared against the others.
    """
    # create directory
    os.makedirs('plots/pairplots', exist_ok=True)
    os.makedirs('plots/heatmaps', exist_ok=True)
    os.makedirs('plots/boxplots', exist_ok=True)
    
    # boxplots
    for var in variables:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[target], y=df[var], palette='Set2')
        plt.title(f'{var} Distribution by {target}')
        plt.tight_layout()
        filename = os.path.join('plots/boxplots', f'{var}_by_{target}_boxplot.png')
        plt.savefig(filename, dpi=300)
        plt.show()
        plt.close()


    # pairplot
    if variables:
        sns.pairplot(df[variables + [target]], hue=target, diag_kind='kde', palette='Set1')
        plt.suptitle('Pairplot of Numeric Variables by Attrition', y=1.02)
        filename = os.path.join('plots/pairplots', 'Pairplot.png')
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show(block=False)
        plt.close()

    # correlation heatmap
    if variables:
        plt.figure(figsize=(12,8))
        corr = df[variables + [f'{target}']].corr()
        sns.heatmap(corr, annot=True, cmap='RdBu', fmt='.2f')
        plt.title('Correlation Heatmap')
        filename = os.path.join('plots/heatmaps', 'Heatmap.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def bivar_nominal_plot(df, variables, target='Attrition'):
    """
    Generate plots to visualize dataset correlation 
    between numeric variables against the target variable.

    Parameters
    ----------
    df : DataFrame
        The DataFrame that contains the data.
    variables : list
        List containing variables to plot against the target.
    target : str, default = 'attrition'
        The variable being compared against the others.
    """
    # create directory
    os.makedirs('plots/stackedbars', exist_ok=True)

    # stacked bar plot for nominal variables
    for var in variables:
        cross_tab = pd.crosstab(df[var], df[target])
        cross_tab_pct = cross_tab.div(cross_tab.sum(1), axis=0)
        cross_tab_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')
        plt.ylabel('Proportion')
        plt.title(f'Attrition by {var}')
        plt.legend(title=target)
        filename = os.path.join('plots/stackedbars', f"Attrition by {var}")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def bivar_ordinal_plot(df, variables, target='Attrition'):
    """
    Generate plots to visualize dataset correlation
    between ordinal variables against the target variable.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    variables : dict
        Dictionary containing ordinal variables with list of ordered levels
    target : str 
        Binary target column name
    """
    # make directory
    os.makedirs('plots/barplots', exist_ok=True)

    for var in variables:
        # Proportion of target by level
        ct = df.groupby(var)[target].mean()
        ct.plot(kind='bar', figsize=(6,4), color='skyblue')
        plt.ylabel(f'{target} Rate')
        plt.title(f'{target} Rate by {var}')
        plt.tight_layout()
        filename = os.path.join('plots/barplots', f'{target}_Rate_by_{var}.png')
        plt.savefig(filename, dpi=300)
        plt.show()
        plt.close()

def bivar_binary_plot(df, variables, target='Attrition'):
    """
    Generate bar plots for binary variables vs binary target

    Parameters
    ----------
    df : DataFrame
        Dataset containing binary variables and target
    binary_cols : list
        List of binary column names
    target : str
        Binary target column name
    """
    # create directory
    os.makedirs('plots/binary_barplots', exist_ok=True)

    # bar plot
    for var in variables:
        ct = df.groupby(var)[target].mean()

        plt.figure(figsize=(5,4))
        ct.plot(kind='bar', color='skyblue')
        plt.ylabel(f'Mean {target}')
        plt.xlabel(var)
        plt.title(f'{target} Rate by {var}')
        plt.tight_layout()
        filename = os.path.join('plots/binary_barplots', f'{var}_binary_bar.png')
        plt.savefig(filename, dpi=300)
        plt.show()
        plt.close()

def bivar_datetime_plot(df, variables, target='Attrition', freq='Y'):
    """
    Generate line plots for datetime variables showing mean target over time 

    Parameters
    ----------
    df : DataFrame
        Dataset containing datetime variables and target
    variables : list
        List of datetime column names
    target : str
        Binary target column name
    freq : str
        Frequency to aggregate by ('Y' = year, 'M' = month, etc.)
    """
    # make directory
    os.makedirs('plots/lineplots', exist_ok=True)

    for var in variables:
        # aggregate mean target by time period
        df_temp = df.copy()
        df_temp[var] = pd.to_datetime(df_temp[var], errors='coerce')
        df_grouped = df_temp.groupby(pd.Grouper(key=var, freq=freq))[target].mean().dropna()

        plt.figure(figsize=(8,4))
        df_grouped.plot(marker='o', linestyle='-')
        plt.ylabel(f'Mean {target}')
        plt.xlabel(var)
        plt.title(f'{target} Rate Over Time ({var})')
        plt.tight_layout()
        filename = os.path.join('plots/lineplots', f'{var}_lineplot.png')
        plt.savefig(filename, dpi=300)
        plt.show()
        plt.close()



# def plot_pairplot(df, numeric_vars, hue=None, save_path=None):
#     """
#     Creates a seaborn pairplot for numeric variables.

#     Parameters
#     __________
#     df : DataFrame
#         Data containing the variable
#     numeric_vars : list
#         List of numeric variable names to plot
#     hue : str, optional
#         Column name for coloring points
#     save_path: str, optional
#         Path to save the plot as a file
#     """
#     # build the list of columns to use
#     hue_columns = numeric_vars.copy()
#     if hue: 
#         hue_columns.append(hue)
    
#     #subset the dataframe
#     data_subset = df[hue_columns]

#     # create the pairplot
#     sns.pairplot(data_subset, hue=hue, diag_kind="kde", corner=True)
#     plt.suptitle("Pairplot of Numeric Features", y=1.02)

#     # save plot
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
#     plt.show()

# def plot_correlation_heatmap(df, numeric_vars, save_path=None):
#     """
#     Plots a heatmap of correlations for numeric variables

#     Parameters
#     __________
#     df : DataFrame
#         Data containing the variables.
#     numeric_vars : list
#         List of numeric variable names to plot
#     save_path : str, optional
#         Path to save the plot as a file.
#     """
#     corr = df[numeric_vars].corr()

#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
#     plt.title("Correlation Heatmap of Numeric Variables")
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
#     plt.show()

# def plot_stacked_bar(df, cat_var, target, normalize=True, save_path=None):
#     """
#     Plots a stacked bar chart for a categorical variable against the target.

#     Parameters
#     ----------
#     df : DataFrame
#         Data containing the variables.
#     cat_var : str
#         Categorical variable name.
#     target : str
#         Target variable name.
#     normalize ; bool, default=True
#         Whether to show proportions instead of raw counts.
#     save_path : str, optional
#         Path to save the plot as a file.
#     """
#     # create crosstab
#     ctab = pd.crosstab(df[cat_var], df[target])

#     # normalize if needed
#     if normalize:
#         ctab = ctab.div(ctab.sum(axis=1), axis=0)

#     # plot stacked bar
#     ctab.plot(kind='bar', stacked=True, figsize=(8,6), colormap="Set2")
#     plt.xlabel(cat_var)
#     plt.ylabel("Proportion")
#     plt.title(f"Stacked Bar Chart of {cat_var} vs {target}")
#     plt.legend(title=target)
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()