import pandas as pd
import warnings 
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_datetime64_dtype

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
        
    if sorted(list(df[y].drop_duplicates())) != [0,1]:
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
    
    'Check input data'
    df = check_y(df,y)
   
    'Stores all column names into list'
    x_all = list(set(df.columns).difference([y]))
    
    'Checks if both options are not selected'
    if (x_sel is not None and x_skip is not None):
         raise ValueError('Both, variable selection and filtering is not allowed. Please use neither or either.')
         
    'Predetermined selection or elimination'
    if x_sel is not None:
        if df.columns.isin(x_sel).sum() != len(x_sel):
            raise ValueError('Invalid list of provided variables. Variable not in dataset present.')
        
        return x_sel
        
    elif x_skip is not None:
        if df.columns.isin(x_skip).sum() != len(x_skip):
            raise ValueError('Invalid list of provided variables. Variable not in dataset present.')
        
        return set(x_all).difference(set(x_skip))

    else:
        return x_all
     
def select_numeric_vars(df, y, x_sel = None, x_skip = None):
    
    '''
    Creates a list of numeric variables [int,float]
    
    Parameters
    ----------
    df: DataFrame
        Input DataFrame
    y:  String
        Response variable
    x_sel : List, optional
        Selection of explanatory variables. The default is None.
    x_skip : List, optional
        Explanatory variables to be skipped. The default is None.

    Returns
    -------
    List
        List of numeric variables used for modelling.
    
    '''
        
    'Create selection of variables'
    x_vars = x_variables(df,y,x_sel,x_skip)
    
    'Creates list of numerical values'
    x_numeric = df[x_vars].apply(pd.to_numeric,errors = 'ignore').select_dtypes(['int64','float64']).columns
        
    return x_numeric

def select_categorical_vars(df, y, x_sel = None, x_skip = None):
    
    '''
    Creates a list of categorical variables 
    
    Parameters
    ----------
    df: DataFrame
        Input DataFrame
    y:  String
        Response variable
    x_sel : List, optional
        Selection of explanatory variables. The default is None.
    x_skip : List, optional
        Explanatory variables to be skipped. The default is None.

    Returns
    -------
    List
        List of categorical variables used for modelling.
    '''
    
    'Create selection of variables'
    x_vars = x_variables(df,y,x_sel,x_skip)
    
    'Creates list of categorical variables'
    x_categoric = df[x_vars].apply(pd.to_numeric,errors = 'ignore').select_dtypes(object).columns
    
    return x_categoric

def select_datetime_vars(df, y, x_sel = None, x_skip = None):
    
    '''
    Creates a list of datetime variables 
    
    Parameters
    ----------
    df: DataFrame
        Input DataFrame
    y:  String
        Response variable
    x_sel : List, optional
        Selection of explanatory variables. The default is None.
    x_skip : List, optional
        Explanatory variables to be skipped. The default is None.

    Returns
    -------
    List
        List of categorical variables used for modelling.
    '''
    
    'Create selection of variables'
    x_vars = x_variables(df, y, x_sel, x_skip)
    
    'Creates a list of datetime variables'
    x_datetime = [i for i in x_vars if is_datetime64_dtype(df[i])]
    
    return x_datetime
