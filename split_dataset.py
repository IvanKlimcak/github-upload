import pandas as pd
from general_purpose import check_y
 
def df_split(df, y, ratio = 0.70, split_by_y = True, seed = 123):
    
    df = check_y(df,y)
    
    if not isinstance(ratio,(float,int)):
        raise TypeError('Ratio is not in acceptable format. Use float.')
    
    if not isinstance(seed,int):
        raise TypeError('Seed is not in acceptable format. Use integer.')
        
    if not (0 <= ratio <= 1):
        raise ValueError('Ratio out of boundaries. Accepted ratio = <0,1>.')
    
    if seed <= 0:
        raise ValueError('Seed out of boundaries. Accepted seed > 0.')
    
    if ratio == 1: 
        print('[INFO] All observation will be in training sample.')
    
    if ratio == 0:
        print('[INFO] All observations will be in testing sample.')
    
    if split_by_y:
        print(f'[INFO] Splitted by dependent variable {y}.')
                
        df_train = df.groupby(y).apply(lambda x: x.sample(frac = ratio, random_state=seed)).sort_index()
        df_test = df.loc[set(df.index).difference(set(df_train.index))]
        
        return df_train, df_test
    
    else:
        print('[INFO] Splitted randomly.')
        
        df_train = df.sample(frac = ratio, random_state = seed).sort_index()
        df_test = df.loc[set(df.index).difference(set(df_train.index))].sort_index()    
        
        return df_train, df_test