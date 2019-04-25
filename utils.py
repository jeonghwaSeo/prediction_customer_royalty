import pandas as pd
import datetime

def train_test(df_train, df_test):
    df_train['outliers'] = 0
    df_train.loc[df_train['target'] < -30, 'outliers'] = 1
    df_test['target'] = -10000
    df_test['outliers'] = -1
    df = pd.concat([df_train, df_test], axis=0)
    
    df.loc[df['first_active_month'].isnull(), 'first_active_month'] = datetime.datetime.today()
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['quarter'] = df['first_active_month'].dt.quarter
    df['sofar_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    
    for col in ['feature_1', 'feature_2', 'feature_3']:
        df[col + '_ratio'] = df[col]/df['sofar_time']
        df[col + '_multiply'] = df[col]*df['sofar_time']
    
    df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']
    df['feature_mean'] = df['feature_sum']/3
    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)
    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)
    df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)
    
    # mapping:numerical to categorical
    df['feature_1_cat'] = df['feature_1'].map({1:'A', 2:'B', 3:'C', 4:'D', 5:'E'})
    df['feature_2_cat'] = df['feature_2'].map({1:'A', 2:'B', 3:'C'})
    df['feature_3_cat'] = df['feature_3'].map({1:'Y', 0:'N'})
    
    return df
    
def feature_engineering (df, df_):
    
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    
    # mapping : categorical to numerical
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 
    
    # mapping : numerical to categorical
    df['category_2'] = df['category_2'].map({1.0:'A', 2.0:'B', 3.0:'C', 4.0:'D', 5.0:'E'}) 
 
    # purchase_amount sum by category_2, 3
    
    for col in ['category_2','category_3']:
        df[col+'_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
        df[col+'_sum'] = df.groupby([col])['purchase_amount'].transform('sum')
        df[col+'_max'] = df.groupby([col])['purchase_amount'].transform('max')
        df[col+'_min'] = df.groupby([col])['purchase_amount'].transform('min')
        #aggs[col+'_mean'] = ['mean']
     
    
    # additional features
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['quarter'] = df['purchase_date'].dt.quarter
    df['month'] = df['purchase_date'].dt.month
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']

    # additional features
    df['duration'] = df['purchase_amount']*df['month_diff']
    df['amount_month_ratio'] = df['purchase_amount']/df['month_diff']
   
    # feature aggregation list
    aggs = {}
    for col in ['month','hour','weekofyear', 'dayofweek','year','subsector_id','merchant_id'
                ,'merchant_category_id', 'category_2', 'category_3']:
        aggs[col] = ['nunique']

    aggs['purchase_amount'] = ['sum','max','min','mean','var']
    aggs['installments'] = ['sum','max','min','mean','var']
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['max','min','mean','var']
    aggs['month_diff'] = ['mean']
    aggs['authorized_flag'] = ['sum', 'mean']
    aggs['weekend'] = ['sum', 'mean']
    aggs['category_1'] = ['sum', 'mean']
    aggs['card_id'] = ['size']
    aggs['duration']=['mean','min','max','var','skew']
    aggs['amount_month_ratio']=['mean','min','max','var','skew']
    
    # feature aggregation
    new_columns = [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
    df_group = df.groupby('card_id').agg(aggs)
    
    # reindexing 
    df_group.columns = new_columns
    df_group.reset_index(drop=False,inplace=True)
    
    # add features using added features
    df_group['purchase_date_diff'] = (df_group['purchase_date_max'] - df_group['purchase_date_min']).dt.days
    df_group['purchase_date_average'] = df_group['purchase_date_diff']/df_group['card_id_size']
    df_group['purchase_date_uptonow'] = (datetime.datetime.today() - df_group['purchase_date_max']).dt.days
    df_ = df_.merge(df_group,on='card_id',how='left')
    
    return df_
