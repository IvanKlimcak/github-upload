
from general_purpose import check_y,x_variables
import numpy as np
 
def correlation_measures(df, y, x_sel = None, x_skip = None, order = True):
    
    '''
    Calculates correlation measures for all variables within DataFrame with exception of response variable
    
    Parameters
    ----------
    df: DataFrame
        Input DataFrame.
    y:  String
        Name of response variable.
    x_sel: List
           List of selected variables. Default is None.
    x_skip: List
            List of variables to be skipped. Default is None.
    
    '''
        
    def _corr_evaluation(x):
        if abs(x) >= 0.7:
            return "High correlation"
        elif 0.2 <= abs(x) < 0.7:
            return "Moderate correlation"
        else:
            return "Low correlation"
    
    'Check inputs'
    df = check_y(df, y)
    
    'Creates a list of variables for correlation analysis'
    x = x_variables(df,y,x_sel,x_skip)
    
    'Creates a copy of input file'
    df = df.copy(deep = True)
    
    'Calculates correlation'
    corr_mat = df[x].corr()

    'Creates outputs for correlation'
    corr_mat = corr_mat.where(np.triu(np.ones(corr_mat.shape).astype(bool)))    
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','corr_coef']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2']]
    corr_mat['evaluation'] = list(map(_corr_evaluation,corr_mat['corr_coef']))
    corr_mat['corr_coef'] = corr_mat['corr_coef'].mul(100).map('{:.2f}'.format) + '%'

    if order:
        corr_mat = corr_mat.sort_values(by='corr_coef',ascending = False)
    
    return corr_mat