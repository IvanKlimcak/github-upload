import pandas as pd
from pathlib import Path
from general_purpose import check_y, x_variables
import warnings


path = Path("/Users/msivecova/desktop/scorecard_dev/cc.csv")
inp_data = pd.read_csv(path)



def missing_values(df, y, x_sel = None, x_skip = None, threshold = 0):
    
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
    
    
            
x_to_filter = outlier_detection(inp_data,'default_time')

x_to_filter = x_to_filter.reset_index().rename(columns = {'index':'variable'})
    



