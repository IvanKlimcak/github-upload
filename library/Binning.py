import math
import pandas as pd
import numpy as np
from general_purpose import select_numeric_vars, select_categorical_vars
    
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
    

def coarse_classifying(df, col, split_num):
    
    '''
    Coarse classifying for columns with large number of unique values. Reduce computational time.
    
    Parameters 
    ----------
    df: DataFrame
        Input DataFrame
    col: Column
        Input column
    split_num: Int
        Fineness of coarse classifying. The increase of fineness increase computational time. Default is 100.
        
    Returns
    -------
    List
        Coarse breakpoints
    '''
    
    if not isinstance(split_num,int):
        raise ValueError('Splitting number is unsupported type. Use int.')
    
    'Creates a copy of input dataset'
    df = df.copy(deep = True)
    
    'Lenght of dataset'
    count = len(df[col])
    
    'Split number'
    n = math.floor(count/split_num)
    
    'Split index'
    split_index = [i*n for i in range(1,split_num)]

    'All values sorted'
    values = sorted(list(df[col]))
    
    'Extracting splitted values'
    split_value = [values[i] for i in split_index]
    
    'Distinct list of values to be sorted'
    split_value = sorted(list(set(split_value)))
    
    return split_value

def assign_coarse_classifying(x, split_value):
    
    '''
    Assign coarse classifying to variable.    
    '''
    
    n = len(split_value)
    
    if x <= min(split_value):
        return min(split_value)
    
    elif x >= max(split_value):
        return 10e10
    
    else:
        for i in range(n-1):
            if split_value[i] < x <= split_value[i+1]:
                return split_value[i+1]


def bin_bad_rate(df, col, y, total_bad = False):
    
    '''
    Calculates bad rate per each distinct value.
    
    Parameters
    ----------
    df: DataFrame
        Input DataFrame
    col: String
        Input column
    target: String
        Response variable
    total_bad: Bool
        Returns overall bad rate. Default is False
    
    Returns
    -------
    Tuple
        Dictionary of bad rates per unique value and DataFrame with good, bad and total number of observations.
    
    '''
    
    'Copy of input dataset'
    df = df.copy(deep = True)
    
    'Calculates bad rate'
    df_bin = df.groupby(col).apply(good_bad,y = y).assign(bad_rate = lambda x: x.bad / x.total).reset_index()
    
    'Calculates total bad rate'
    overall = good_bad(df,y)
    total_br = overall.bad / overall.total
    
    dict_bad = dict(zip(df_bin[col],df_bin['bad_rate']))
    
    if total_bad == False:
        return (dict_bad, df_bin)
    else:
        return (dict_bad, df_bin, total_br)
    
def chi2_binning(df, overall_br):
    
    '''
    Calculates Chi-squared statistics for binning
    
    Parameters
    ---------
    df: DataFrame
        Input DataFrame
    overall_br: Float,Int
        Overall default rate
    
    Returns 
    -------
    Float
        Chi-squared statistics
    '''
    
    df = df.copy(deep = True)
    df['expected'] = df['total'] * overall_br
    df['chi'] = (df['expected'] - df['bad']) ** 2 / df['expected']
    
    return df['chi'].sum()


def assign_bin(x, cut_offs):
    
    'Assigns bins based on predefined cut-off points'    
    
    bin_num = len(cut_offs) + 1
    
    if x <= cut_offs[0]:
        return 'Bin 0'
    elif x > cut_offs[-1]:
        return 'Bin {}'.format(bin_num-1)
    
    else:
        for i in range(0, bin_num - 1):
            if cut_offs[i] < x <= cut_offs[i+1]:
                return 'Bin {}'.format(i+1)
            

def Chi_merge(df, col, y, max_bins = 5, min_binpct = 0, split_num = 100):
    '''
    Creates binning for a given variable 

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    col : String
        Explanatory variable name.
    y : String
        Response variable.
    max_bins : Int, optional
        Maximum number of available bins. The default is 5.
    min_binpct : Float, optional
        Minimum proportion within bin. The default is 0.
    split_num : Int, optional
        Number of distinct values for variable. The default is 100.

    Returns
    -------
    cut_offs : List
        Cut-off points for binning.

    '''
    
    'Copy of input dataset'
    df = df.copy(deep = True)
    
    n = len(set(df[col]))
    
    if n > split_num:
        split_value = coarse_classifying(df, col, split_num)
        df['col_map'] = df[col].map(lambda x: assign_coarse_classifying(x,split_value))
    
    else:
        df['col_map'] = df[col]
    
    'Calculating bad rate per bin'
    (dict_bad, df_br, overall_br) = bin_bad_rate(df, 'col_map', y, True)    

    'Creating groups for binning'
    col_map_unique = sorted(set(df_br['col_map']))    
    group_interval = [[i] for i in col_map_unique]
    
    'Itteratively decreases number of cutoff points based on Chi-squared'
    while(len(group_interval) > max_bins):
        chi_list = []
        
        'Calculates Chi-squared for 2 groups next to each other'
        for i in range(len(group_interval) - 1):
            group = group_interval[i] + group_interval[i+1]
            df_group = df_br[df_br['col_map'].isin(group)]
            chi2 = chi2_binning(df_group,overall_br)
            chi_list.append(chi2)
        
        'Selects those with minimum Chi-squared statistics and merge with the value which follows'
        min_chi2 = chi_list.index(min(chi_list))
        group_interval[min_chi2] = group_interval[min_chi2] + group_interval[min_chi2 + 1]
        group_interval.remove(group_interval[min_chi2+1])
    
    'Final cut-offs'
    group_interval = [sorted(i) for i in group_interval]
    cut_offs = [max(i) for i in group_interval[:-1]]
        
    return cut_offs


def bin_num_vars(df, y, x_sel = None, x_skip = None, max_bins = 5, min_binpct = 0):
    
    '''
    Numeric variables binning 
    
    Parameters 
    ---------
    df: DataFrame
        Input DataFrame
    y:  String
        Response variable
    x_sel: List
        Custom selection of explanatory variables
    x_skip: List
        Custom elimination of explanatory variables
    max_bins: Int
        Maximum number of available bins
    min_binpct Float
        Minimum proportion within the bin
        
    Returns
    ------
    bins_num: List
        Details of binning per variable
    IV_num: DataFrame
        DataFrame with information values
    Df: DataFrame
        Output DataFrame with WOE values
    '''
    
    'Copy of input dataset'
    df = df.copy(deep = True)
    
    'Select numeric variables'
    x_num = select_numeric_vars(df, y, x_sel, x_skip)
    
    'Overall statistics'
    overall = good_bad(df, y)
    
    bins_num = []
    IV_list_num = []
    
    for x in x_num:
        
        'Calculate cut-offs'
        cut_offs = Chi_merge(df, x, y, max_bins, min_binpct)
        cut_offs.insert(0,float('-inf'))
        cut_offs.append(float('inf'))
        df_bins = df.groupby(pd.cut(df[x], cut_offs))
        
        df_out = pd.DataFrame()
        
        df_out['min'] = df_bins[x].min()
        df_out['max'] = df_bins[x].max()
        
        df_out['total'] = df_bins[y].count()
        df_out['total_cum'] = df_out['total'].cumsum()
        df_out['total_rate'] = df_out['total'] / overall.total
        
        df_out['bad'] = df_bins[y].sum()
        df_out['bad_cum'] = df_out['bad'].cumsum()
        df_out['bad_rate'] = df_out['bad'] / df_out['total']
        
        df_out['good'] = df_out['total'] - df_out['bad']
        df_out['good_cum'] = df_out['good'].cumsum()
        df_out['good_rate'] = df_out['good'] / df_out['total']
        
        df_out['bad_attr'] = df_out['bad'] / overall.bad
        df_out['good_attr'] = df_out['good'] / overall.good
        
        df_out['woe'] = np.log(df_out['bad_attr'] / df_out['good_attr'])
        df_out['iv_bin'] = (df_out['bad_attr'] - df_out['good_attr']) * df_out['woe']
        df_out['IV'] = df_out['iv_bin'].sum().round(3)
        
        df[x] = df[x].map(df_out['woe'])
        
        IV_list_num.append(df_out['IV'].sum().round(3))
        bins_num.append(df_out)
    
    'Information value DataFrame'    
    IV_num = pd.DataFrame({'col': x_num, 'IV': IV_list_num}).sort_values(by= 'IV', ascending = False)
    
    return bins_num, IV_num, df

def bin_cat_vars(df, y, x_sel = None, x_skip = None, max_bins = 5, min_binpct = 0):
    
    '''
    Categorical variables binning
    Parameters
    ----------
    df: DataFrame
        Input DataFrame
    y:  String
        Response variable
    x_sel: List
        Custom selection of explanatory variables
    y_skip: List
        Custom elimination of explanatory variables
    max_bins: Int
        Maximum number of available bins. Default is 5
    min_binpct: Float
        Minimum proportion within the bin. Default is 0
    
    Returns
    -------
    bins_num: List
        Details of binning per variable
    IV_num: DataFrame
        DataFrame with information values
    Df: DataFrame
        Output DataFrame with WOE values
    
    '''
    
    'Copy of input DataFrame'
    df = df.copy(deep = True)
    
    'Select categorical variables'
    x_cat = select_categorical_vars(df, y)
    
    'Overall statistics'
    overall = good_bad(df, y)
    
    bins_cat = []
    IV_list_cat = []
    
    for x in x_cat:
        
        'Calculate IV for categoric variables'
        df_bins = df.groupby([x])
    
        df_out  = pd.DataFrame()
        
        df_out['total'] = df_bins[y].count()
        df_out['total_cum'] = df_out['total'].cumsum()
        df_out['total_rate'] = df_out['total'] / overall.total
        
        df_out['bad'] = df_bins[y].sum()
        df_out['bad_cum'] = df_out['bad'].cumsum()
        df_out['bad_rate'] = df_out['bad'] / overall.bad
        
        df_out['good'] = df_out['total'] - df_out['bad']
        df_out['good_cum'] = df_out['good'].cumsum()
        df_out['good_rate'] = df_out['good'] / df_out['total']
        
        df_out['bad_attr'] = df_out['bad'] / overall.bad
        df_out['good_attr'] = df_out['good'] / overall.good
        
        df_out['woe'] = np.log(df_out['bad_attr'] / df_out['good_attr'])
        df_out['iv_bin'] = (df_out['bad_attr'] - df_out['good_attr']) * df_out['woe']
        df_out['IV'] = df_out['iv_bin'].sum().round(3)
        
        df[x] = df[x].map(df_out['woe'])
        
        IV_list_cat.append(df_out['iv_bin'].sum().round(3))
        bins_cat.append(df_out)
    
    'Information value DataFrame'
    IV_cat = pd.DataFrame({'col':x_cat, 'IV': IV_list_cat}).sort_values(by = 'IV', ascending = False)
    
    return bins_cat, IV_cat, df
     
def woe_monotonic(bin_df):
    
    'Variables list'
    col_name = [i.index.name for i in bin_df]
    
    woe_mon = []
    
    for i in range(len(col_name)):
        w = bin_df[i].woe.to_list()
        
        if len(w) == 2:
            woe_nm = [(w[i] > w[i+1]) for i in range(len(w)-1)]
        
            if True in woe_nm:
                woe_mon.append('False')
            else:
                woe_mon.append('True')
    
        else:
            woe_nm = [((w[i] > w[i+1] and w[i] > w[i-1]) or (w[i] < w[i-1] and w[i] < w[i+1])) for i in range(len(w)-1)]
        
            if True in woe_nm:
                woe_mon.append('False')
            else:
                woe_mon.append('True')
        
    Monotonic_df = pd.DataFrame({'col': col_name, 'Monotonic': woe_mon})
    
    return Monotonic_df
