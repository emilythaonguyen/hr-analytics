import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairplot(df, numeric_vars, hue=None, save_path=None):
    """
    Creates a seaborn pairplot for numeric variables.

    Parameters
    __________
    df : DataFrame
        Data containing the variable
    numeric_vars : list
        List of numeric variable names to plot
    hue : str, optional
        Column name for coloring points
    save_path: str, optional
        Path to save the plot as a file
    """
    # build the list of columns to use
    hue_columns = numeric_vars.copy()
    if hue: 
        hue_columns.append(hue)
    
    #subset the dataframe
    data_subset = df[hue_columns]

    # create the pairplot
    sns.pairplot(data_subset, hue=hue, diag_kind="kde", corner=True)
    plt.suptitle("Pairplot of Numeric Features", y=1.02)

    # save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()

def plot_correlation_heatmap(df, numeric_vars, save_path=None):
    """
    Plots a heatmap of correlations for numeric variables

    Parameters
    __________
    df : DataFrame
        Data containing the variables.
    numeric_vars : list
        List of numeric variable names to plot
    save_path : str, optional
        Path to save the plot as a file.
    """
    corr = df[numeric_vars].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title("Correlation Heatmap of Numeric Variables")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_stacked_bar(df, cat_var, target, normalize=True, save_path=None):
    """
    Plots a stacked bar chart for a categorical variable against the target.

    Parameters
    ----------
    df : DataFrame
        Data containing the variables.
    cat_var : str
        Categorical variable name.
    target : str
        Target variable name.
    normalize ; bool, default=True
        Whether to show proportions instead of raw counts.
    save_path : str, optional
        Path to save the plot as a file.
    """
    # create crosstab
    ctab = pd.crosstab(df[cat_var], df[target])

    # normalize if needed
    if normalize:
        ctab = ctab.div(ctab.sum(axis=1), axis=0)

    # plot stacked bar
    ctab.plot(kind='bar', stacked=True, figsize=(8,6), colormap="Set2")
    plt.xlabel(cat_var)
    plt.ylabel("Proportion")
    plt.title(f"Stacked Bar Chart of {cat_var} vs {target}")
    plt.legend(title=target)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()