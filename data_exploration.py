import pandas as pd
import numpy as np
from scipy import stats
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from pathlib import Path
from general_purpose import check_y, x_variables
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


path = Path("/Users/msivecova/desktop/scorecard_dev/cc.csv")
inp_data = pd.read_csv(path)

def univariate_analysis(x):
    
    '''
    Calculates location and dispersion measures
    
    Parameters
    ----------
    x: Numeric
       Variable to be analyzed. Applicable only for numeric variables.
       
    Returns
    -------
    Series
        Series with location and dispersion measures

    '''
            
    if not is_numeric_dtype(x):
        raise TypeError('Invalid type provided. Only allowed type is numeric.')

    out_stats = pd.Series(data=[ len(x)
                             ,x.isnull().sum()
                             ,x.mean()
                             ,x.median()
                             ,x.mode().max()
                             ,x.var()
                             ,x.std()
                             ,x.skew()
                             ,x.kurt()
                             ,x.max() - x.min()
                             ,x.quantile(0.75) - x.quantile(0.25)
                             ,x.mean() - ((stats.norm.ppf(0.975) * x.std()) / np.sqrt(len(x)))
                             ,x.mean() + ((stats.norm.ppf(0.975) * x.std()) / np.sqrt(len(x)))
                             ,((x.std() / x.mean()) * 100)]
                     ,index = ['# Observations'
                                ,'# Missing values'
                                ,'Mean'
                                ,'Median'
                                ,'Modus'
                                ,'Variance'
                                ,'Standard Deviation'
                                ,'Skewness'
                                ,'Kurtosis'
                                ,'Range'
                                ,'IQR'
                                ,'Upper 95% CI'
                                ,'Lower 95% CI'
                                ,'Coefficient of variation'])

    return out_stats.map('{:.4f}'.format)

def plot_cat_var(df, y, x_sel = None, x_skip = None, stacked = False, rows = None, cols = None):
    
    '''
    Plots all categorical variables within dataset
    
    Parameters
    ----------
    df: DataFrame
        Input DataFrame
    y:  String 
        Name of dependent variable
    x_sel: List
        List of selected variables to be analyzed
    x_skip: List
        List of viariables to be skipped
    stacked: Bool
        True if variables should be stacked
    rows: Int
        Number of rows in stacked graph. Positive number required.
    cols: Int
        Number of columns in stacked graph. Positive numer required.        
    
    Returns
    -------
    Plots
        Objects are showed
    
    '''
    
    'Creates a copy of input dataset'
    df = df.copy(deep = True)
    
    'Checks input data'    
    df = check_y(df,y)
    
    'Creates a list of input variables'
    x_vars = x_variables(df, y, x_sel, x_skip)
    
    'Subselects only categorical variable'
    x_selected = [i for i in x_vars if is_categorical_dtype(df[i])]
    
    'Empty list'
    if len(x_selected) == 0:
        warnings.warn('Numer of categorical variables within dataset is 0.')
    
    'Separate or Stacked plot'
    if not stacked:
        for i, col in enumerate(x_selected):
            plt.subplots()
            plt.title(col)
            sns.countplot(data = df, x = col)
            plt.xlabel('')
    else: 
        'Setting up the stacked graph'
        if (not (isinstance(rows,int) or isinstance(cols,int))) or (rows < 0) or (cols < 0) :
            rows = len(x_selected)
            cols = 1

        fig, axes = plt.subplots(nrows = rows, ncols = cols)
        for i, col in enumerate(x_selected):
            plt.subplot(rows,cols,i+1)
            plt.title(col)
            sns.countplot(data = df, x = col)
            plt.xlabel('')
            
    plt.tight_layout()
    return plt.show()

def plot_num_var(df, y, x_sel = None, x_skip = None,plt_type = 'hist' , stacked = False, rows = None, cols = None):
    
    'Creates a copy of input dataset'
    df = df.copy(deep = True)
    
    'Checks input data'
    df = check_y(df,y)
    
    'Creates a list of input variabes'
    x_vars = x_variables(df, y, x_sel, x_skip)
    
    'Subselects only numerical variable'
    x_selected = [i for i in x_vars if is_numeric_dtype(df[i])]
    
    'Empty list'
    if len(x_selected) == 0:
        warnings.warn('Number of numerical variables within dataset is 0.')
        
    'Checks if type '
    
    
    

