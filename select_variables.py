import pandas as pd
from general_purpose import check_y, x_variables
import warnings
    
def filter_var(df, y, x = None, remove_var = None, keep_var = None, return_reason = True, threshold_missing = 0.50, threshold_identical = 0.50):
        
    'Create copy of input dataset'
    df = df.copy(deep = True)
    
    'Check response variable'
    df = check_y(df,y)
    
    'Creates list of explanatory variable'
    x = x_variables(df,y,x,remove_var)
              
    'Keeping variable'
    if keep_var is not None:
        if not isinstance(keep_var,list):
            raise TypeError('Invalid type provided. Supported format is list.')
        
        if df.columns.isin(keep_var).sum() != len(keep_var):
            raise ValueError('Variable not present in dataset.')
        
        x_to_filter = list(set(x) - set(keep_var))
    else:
        x_to_filter = x
    
    'Missing rate'
    miss_rate = lambda x: x.isnull().sum() / len(x)
    
    na_perc = df[x_to_filter].apply(miss_rate).reset_index(name = 'missing_rate').rename(columns = {'index':'Variable'})
                                 
    na_perc = na_perc.loc[na_perc['missing_rate'] > threshold_missing]
    
    na_perc = na_perc.assign(reason = lambda x: [f'Missing rate is greater than {round(threshold_missing,2)}' for i in x.missing_rate])
    
    na_perc = na_perc.rename(columns = {'missing_rate':'value'})
    
    'Identical rate'
    ident_rate = lambda x: x.value_counts().max() / len(x)
    
    ident_perc = df[x_to_filter].apply(ident_rate).reset_index(name = 'identical_rate').rename(columns = {'index':'Variable'})
        
    ident_perc = ident_perc.loc[ident_perc['identical_rate'] > threshold_identical]
    
    ident_perc = ident_perc.assign(reason = lambda x: [f'Identical rate is geater than {round(threshold_identical,2)}' for i in x.identical_rate])
    
    ident_perc = ident_perc.rename(columns = {'identical_rate':'value'})
    
    'Excluded variables'
    x_excluded = pd.concat([na_perc, ident_perc])['Variable'].drop_duplicates()
    
    if keep_var is not None:
        x_kept = list(set(x_to_filter) - set(x_excluded)) + keep_var + [y]
    else:
       x_kept = list(set(x_to_filter) - set(x_excluded)) + [y]
        
    if return_reason:
        return df[x_kept], na_perc.append(ident_perc,ignore_index=True)
    else:
        return df[x_kept]
    
    

