import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
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

def bivar_spearman_plot(df, variables, target='Attrition'):
    """
    Generate spearman plot to visualize dataset correlation
    between ordinal + numeric variables against the target variable.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    variables : list
        Dictionary containing ordinal + numeric variables with list of ordered levels
    target : str 
        Binary target column name
    """
    # make directory
    os.makedirs('plots/heatmaps', exist_ok=True)
    
    # keep only columns that exist in df
    cols = [col for col in variables if col in df.columns] + [target]
    
    if cols:
        plt.figure(figsize=(12,8))
        corr = df[cols].corr(method='spearman')
        sns.heatmap(corr, annot=True, cmap='RdBu', fmt='.2f', center=0)
        plt.title('Spearman Correlation Heatmap')
        filename = os.path.join('plots/heatmaps', 'Spearman_Heatmap.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
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