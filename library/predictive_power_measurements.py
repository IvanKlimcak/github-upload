import pandas as pd
import numpy as np
from general_purpose import x_variables
from Binning import *
import scorecardpy as sc

df = sc.germancredit()
df['creditability'].loc[df['creditability']=='good'] =  0
df['creditability'].loc[df['creditability']== 'bad'] = 1


def correlation_measures(df, y, x_sel = None, x_skip = None, order = True):
    
    '''
    Calculates correlation measures for all variables within DataFrame with exception of response variable
    
    Parameters
    ----------
    df: DataFrame
        Input DataFrame.
    y:  String
        Name of response variable.
    order: Bool.optional
        Ordered correlation
    '''
    def _corr_evaluation(x):
        if abs(x) >= 0.7:
            return "High correlation"
        elif 0.2 <= abs(x) < 0.7:
            return "Moderate correlation"
        else:
            return "Low correlation"
    
    'Creates a copy of input file'
    df = df.copy(deep = True)
 
    'Select variables'
    x_vars = x_variables(df, y, x_sel, x_skip)
     
    'Calculates correlation for numeric variables'
    corr_mat = df[x_vars].corr(method = 'pearson')

    'Creates outputs for correlation'
    corr_eval = corr_mat.where(np.triu(np.ones(corr_mat.shape).astype(bool)))    
    corr_eval = corr_mat.stack().reset_index()
    corr_eval.columns = ['variable_1','variable_2','corr_coef']
    corr_eval = corr_eval.loc[corr_eval['variable_1'] != corr_eval['variable_2']]
    corr_eval['evaluation'] = list(map(_corr_evaluation,corr_eval['corr_coef']))
    
    if order:
        corr_eval = corr_eval.sort_values(by='corr_coef',ascending = False)    
    
    corr_eval['corr_coef'] = corr_eval['corr_coef'].mul(100).map('{:.2f}'.format) + '%'

    return corr_eval

def forward_selection_pvalues(x_train, y_train):
    '''
    Forward selection of variables based on p-value in logistic regression.
    
    Parameters
    ----------
    x_train : DataFrame
        Training DataFrame with explanatory variables.
    y_train : DataFrame
        Training DataFrame with response variable.

    Returns
    -------
    col_list : List
        Statistically significant variables.

    '''
    
    'Select explanatory variables'
    x_vars = set(x_train.columns)
    
    col_list = []
    'Calculate forward logistic regression'
    for x in x_vars:
        
        col_list.append(x)
        x_train_s = sm.add_constant(x_train.loc[:,col_list])
        lr = sm.Logit(y_train.astype(float),x_train_s.astype(float))
        lr = lr.fit()
        
        for i,j in zip(lr.pvalues.index[1:],lr.pvalues.values[1:]):
            if j > 0.05:
                col_list.remove(i)
                
    'Regression with statistically significant variables'
    x_train_sv = sm.add_constant(x_train.loc[:,col_list])
    lr_sel = sm.Logit(y_train.astype(float), x_train_sv.astype(float))
    lr_sel = lr_sel.fit()
    
    print(lr_sel.summary2())
    
    return col_list    
        