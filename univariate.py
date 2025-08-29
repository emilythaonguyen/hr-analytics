import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, normaltest, probplot
import os

# set the style for all plots
sns.set(style='whitegrid', palette='tab20c')
plt.rcParams['figure.figsize'] = (10, 6)

# function to print out summary for qualitative variables 
# and proportion along w/plots.
def ql_stats(df, col):
    """
    Prints the summary and proportion of each category in the selected column,
    then plot and save a count plot.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data
    col : str
        The name of the categorical column to analyze
    """

    # summary
    counts = df[col].value_counts(dropna=False)
    percentages = df[col].value_counts(normalize=True, dropna=False) * 100
    print(f"\n--- Categorical Summary: {col} ---")
    summary = pd.DataFrame({
        'Count': counts,
        'Percentages': percentages.round(2)
    })
    print(summary)
    print(f"Unique categories: {df[col].nunique(dropna=False)}")
    print(f"Most frequent: {df[col].mode()[0]}")

    # make directory for count plots
    dir = "plots/countplots"
    os.makedirs(dir, exist_ok=True)
    # countplot
    plt.figure(figsize=(12, 6))
    sns.countplot(x=col, data=df, hue=col)
    if col in ['Ethnicity', 'EducationField', 'JobRole']:
        # rotate to make space for x labels
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.3)
    else:
        plt.tight_layout()
    plt.title(f'Distribution of {col}')
    # save the plot to the new directory
    filename = os.path.join(dir, f'Distribution_of_{col}.jpg')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# same concept but for quantitative variables
def qn_stats(df, col):
    """
    Prints out summary for numeric columns, generate 
    box plots, histograms, QQ plots, and then runs a normality test.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    col : str
        The name of the numeric column we want to analyze.
    """
    # summary
    print(f"\n--- Numerical Summary: {col} ---")
    desc = df[col].describe()
    print(desc)

    # make directory for boxplots
    dir = "plots/boxplots"
    os.makedirs(dir, exist_ok=True)

    # boxplot
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    filename = os.path.join(dir, f"Boxplot_of_{col}.jpg")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    # mode
    print(f"Mode: {df[col].mode()[0]}") 
    # skewness + kurtosis
    print(f"Skewness: {skew(df[col].dropna()):.2f}")
    print(f"Kurtosis: {kurtosis(df[col].dropna()):.2f}")
    
    # make directory for histogram
    dir = "plots/histograms"
    os.makedirs(dir, exist_ok=True)
    # histogram + kde
    sns.histplot(df[col], kde=True, stat='density', bins=20)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    filename = os.path.join(dir, f'Distribution_of_{col}.jpg')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    # normality test
    stat, p=normaltest(df[col])
    print(f"\nD'Agostino and Pearson Test:")
    print(f"Statistic = {stat:.4f}, p-value = {p:.4f}")
    if p < 0.05: # 95% confidence
        print("Data is not normally distributed.")
    else:
        print("Data is normally distributed.")

    # make directory for QQ plot
    dir = "plots/qq_plots"
    os.makedirs(dir, exist_ok=True)
    # QQ plot
    plt.figure(figsize=(6,6))
    probplot(df[col], dist='norm', plot=plt)
    plt.title(f'QQ-Plot of {col}')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel('Sample Quantiles')
    plt.grid(True)
    filename = os.path.join(dir, f'QQ-Plot_of_{col}.jpg')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# same thing but with datetime columns
def dt_stats(df, col):
    """
    Summarize datetime column and generate a yearly count plot.
        
    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    col: str 
        The name of the datetime column to summarize.
    """
    # datetime summary
    print(f"\n--- Datetime Summary: {col} ---")
    print(f"Min date: {df[col].min()}")
    print(f"Max date: {df[col].max()}")
    print(f"Range: {df[col].max() - df[col].min()}")
    print(f"Median: {df[col].median()}") 
    print(f"Mode: {df[col].mode()[0]}")
    print(f"Unique dates: {df[col].nunique(dropna=False)}")

    #counts per year
    year_counts = df[col].dt.year.value_counts().sort_index()
    print("\nCounts per year:")
    print(year_counts)

    # make directory for time series plot
    dir = "plots/time_series_plots"
    os.makedirs(dir, exist_ok=True)
    sns.color_palette(palette='tab20c')
    # yearly counts - time series plot
    dt_yearly = df.set_index(col).resample('YE').size()
    plt.figure(figsize=(14, 7))
    dt_yearly.plot(marker='o')
    plt.title(f"Yearly Count of {col}")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    filename=os.path.join(dir, f'Yearly_Count_of_{col}.jpg')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()