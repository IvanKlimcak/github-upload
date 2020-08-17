import pandas as pd
import numpy as np
from general_purpose import check_y, x_variables

def good_bad(df, y):
    
    '''
    Calculates number of good and bad cases for response binary variable.
    
    Parameters
    ----------
    df: DataFrame
        Input DataFrama
    y:  String
        Response binary variable
        
    Returns
    ------
    Series
        Number of bad, good and total cases
    '''
        
    good_bad = {'bad': (sum(df[y])), 'good' : (len(df[y]) - sum(df[y])), 'total': (len(df[y]))}

    return pd.Series(good_bad)

def _iv_resolution(iv):
    
    '''
    Categorize information value results.
    
    Returns
    -------
    String 
        Predictive power commentary
    
    '''
    
    if iv < 0.02 :
        return 'Not useful'
    elif iv <= 0.1:
        return 'Weak predictive power'
    elif iv <= 0.3:
        return 'Medium predictive power'
    elif iv <= 0.5:
        return 'Strong predictive power'
    else:
        return 'Suspicious predictive power'

def iv_ks(df, x, y):
    
    '''
    Information value
    
    Parameters
    ----------
    df: DataFrame
        Input DataFrame
    x:  String
        Name of explanatory variable
    y:  String
        Name of response variable
    
    Returns
    -------
    Float
        Information value
        
    '''
    
    iv_df = pd.DataFrame(df.groupby(x).apply(good_bad,y = y).replace(0,0.01))
        
    iv_df['bad_D'] =  (iv_df['bad'] / sum(iv_df['bad']))
    iv_df['good_D'] = (iv_df['good'] / sum(iv_df['good']))
    
    iv_df['bad_cum'] = (iv_df['bad']).cumsum()
    iv_df['good_cum'] = (iv_df['good']).cumsum()
        
    iv_df['IV'] = (iv_df['good_D'] - iv_df['bad_D']) * np.log(iv_df['good_D'] / iv_df['bad_D'])
    
    return iv_df['IV'].sum()
    
def iv_apply(df, y, x_sel = None, x_skip = None, order = True):
    
    '''
    Calculates information value for selected response variable
    
    Parameters
    ----------
    df: DataFrame
        Input DataFrame
    y:  String 
        Name of response variable
    x_sel: List
            List of selected variables
    x_skip: List
              List of variables to be skipped
    order: Bool
              Sort by information value from highest to lowest
    
    Returns
    -------
    DataFrame
        Output DataFrame with Information value
    
    '''
    
    'Creates a copy of input dataset'
    df = df.copy(deep = True)
    
    'Checks dependent variable'
    df = check_y(df,y)
    
    'List of independent variables'
    x_vars  = x_variables(df,y,x_sel,x_skip)

    'Creates output dataset'    
    out_df = pd.DataFrame({'variable' : x_vars
                             ,'information_value' : [round(iv_ks(df,i,y),6) for i in x_vars]})
    
    'Adding resolution'
    out_df['predictive_power'] = list(map(_iv_resolution,out_df['information_value']))
    
    'Ordering'
    if order:
        out_df = out_df.sort_values(by = 'information_value',ascending=False)
    
    return out_df



