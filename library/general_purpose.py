import pandas as pd
import warnings 
from pandas.api.types import is_numeric_dtype

def check_y(df,y):
    
    '''
    Checks if response variable is binary type
    
    Parameters
    ----------
    df : DataFrame.
         Input DataFrame.
    y :  String
         Response variable name. Only binomial type allowed.
            
    Returns
    -------
    DataFrame
        Input DataFrame.
    
    '''
    
    if not isinstance(df,pd.DataFrame):
        raise TypeError('Invalid type provided. Supported format for df argument is DataFrame.')
        
    if not isinstance(y,str):
        raise TypeError('Invalid type provided. Supported format for y argument is string.')
    
    if y not in df.columns:
        raise Exception(f'Column {y} not in provided DataFrame.')
    
    if df[y].isnull().any():
        warnings.warn(f'Missing values present in column {y}.')
    
    if is_numeric_dtype(df[y]):
        df[y] = df[y].apply(lambda x: x if pd.isnull(x) else int(x))
        
    if list(df[y].drop_duplicates()) != [0,1]:
        raise ValueError(f'Column {y} is not binomial type')
            
    return df      
      
def x_variables(df, y, x_sel = None, x_skip = None):
    
    '''
    Generates list of explanatory variables based on input conditions.
    
    Parameters
    ----------
    df : DataFrame
         Input DataFrame with variables.
    y :  String
         Name of dependent variable.
    x_sel : List, optional
        Selection of explanatory variables. The default is None.
    x_skip : List, optional
        Explanatory variables to be skipped. The default is None.

    Returns
    -------
    List
        List of explanatory variables used for modelling.

    '''
    
    'Checks input data and response variable'
    df = check_y(df,y)
    
    'Stores all column names into list'
    x_all = list(set(df.columns).difference([y]))
    
    'Checks if both options are not selected'
    if (x_sel is not None and x_skip is not None):
         raise ValueError('Both, variable selection and filtering is not allowed. Please use neither or either.')
         
    'Predetermined selection or elimination'
    if x_sel is not None:
        
        if not isinstance(x_sel,list):
            raise TypeError('Invalid type provided. Supported type is list.')
        
        if df.columns.isin(x_sel).sum() != len(x_sel):
            raise ValueError('Invalid list of provided variables. Variable not in dataset present.')
        
        return x_sel
        
    elif x_skip is not None:
        
        if not isinstance(x_skip,list):
            raise TypeError('Invalid type provided. Supported type is list.')
            
        if df.columns.isin(x_skip).sum() != len(x_skip):
            raise ValueError('Invalid list of provided variables. Variable not in dataset present.')
        
        return set(x_all).difference(set(x_skip))

    else:
        return x_all
