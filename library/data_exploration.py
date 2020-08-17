import pandas as pd
import numpy as np
from scipy import stats
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from general_purpose import check_y, x_variables
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def univariate_analysis(x):
    
    '''
    Calculates location and dispersion measures
    
    Parameters
    ----------
    x: Numeric
       Variable to be analyzed. Applicable only for numeric variables.
       
    Returns
    -------
    Series:
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
    Plots:
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

def missing_values(df, y, x_sel = None, x_skip = None, threshold = 0):
    '''
    Calculates missing rates for all variables and returns variables above specified threshold
    
    Parameters
    ----------
    df: DataFrame
        Input DataFrame
    y:  String
        Response variable name
    x_sel: List
        List of selected variables to be analyzed
    x_skip: List
        List of viariables to be skipped
    threshold: int,float
        Minimum limit for variable to be returned. Default is 0.
            
    Returns
    -------
    DataFrame: DataFrame with variables names and proportion of missing values.
    
    '''
    
    if not isinstance(threshold,(float,int)):
        raise TypeError(f'Specified type for threshold: {type(threshold)} not supported.')
    
    if not (0 <= threshold <= 1):
        warnings.warn(f'Threshold value {threshold} out of boundaries. Changed to default value.')
        threshold = 0
    
    'Copy of input dataset'
    df = df.copy(deep = True)
    
    'Check input data'
    df = check_y(df,y)
    
    'Create list of explanatory variables'
    x_vars = x_variables(df,y,x_sel,x_skip)
    
    'Missing rate'
    miss_rt  = lambda x: x.isnull().sum() / len(x)
    na_prct = df[x_vars].apply(miss_rt).reset_index(name = 'Missing_rate').rename(columns = {'index': 'Variable'})
    na_prct = na_prct.loc[na_prct['Missing_rate'] > threshold]
    na_prct['Missing_rate'] = na_prct['Missing_rate'].mul(100).map('{:.2f}'.format) + '%'    
    
    return na_prct
    
def identical_values(df, y, x_sel = None, x_skip = None, threshold = 0.5):
    
    '''
    Calculates proportion of identical values and returns variables above specified threshold
    
    Parameters
    ----------
    df: DataFrame
        Input dataframe
    y:  String
        Name of dependent variable
    x_sel: List
        List of selected variables to be analyzed
    x_skip: List
        List of viariables to be skipped
    threshold: int,float
        Minimum limit for variable to be returned. Default is 0.
    
    Returns
    ------
    DataFrame:
        Variables names with identical rates above threshold
    '''
    
    if not isinstance(threshold, (float, int)):
        raise TypeError(f'Specified type for threshold: {type(threshold)} not supported.')
    
    if not (0 <= threshold <= 1):
        warnings.warn(f'Threshold value {threshold} out of boundaries. Changed to default value.')
        threshold = 0.5
     
    'Copy of input dataset'
    df = df.copy(deep = True)
    
    'Check input data'
    df = check_y(df,y)
    
    'Create list of explanatory variables'
    x_vars = x_variables(df,y,x_sel,x_skip)
    
    'Identical rate'
    ident_rt = lambda x: x.value_counts().max() / len(x)
    ident_prct = df[x_vars].apply(ident_rt).reset_index(name = 'Identical_rate').rename(columns = {'index':'Variable'})
    ident_prct = ident_prct.loc[ident_prct['Identical_rate'] > threshold]
    ident_prct['Identical_rate'] = ident_prct['Identical_rate'].mul(100).map('{:.2f}'.format) + '%'
    
    return ident_prct
   
def outlier_detection(df, y, x_sel = None, x_skip = None, method = 'std'):
    
    'Copy of input dataset'
    df = df.copy(deep = True)
    
    'Check input data'
    df = check_y(df,y)

    'Create list of explanatory variables'
    x_vars = x_variables(df,y,x_sel,x_skip)
    
    'Standard deviation'
    if method.lower() == 'std':
        
        outliers_limits = pd.DataFrame()
        outliers_limits['ll'] = df[x_vars].mean() - 3 * df[x_vars].std()
        outliers_limits['ul'] = df[x_vars].mean() + 3 * df[x_vars].std()
        
        return outliers_limits
    
    
            
    

