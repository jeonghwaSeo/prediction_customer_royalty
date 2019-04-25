import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

csv1 = open('sample_submission.csv', 'r')
df_sub = pd.read_csv(csv1, header=0, index_col=None)

csv2 = open('historical_transactions.csv', 'r')
df_txn = pd.read_csv(csv2, header=0, index_col=None)

csv3 = open('merchants.csv', 'r')
df_mer = pd.read_csv(csv3, header=0, index_col=None)

csv4 = open('new_merchant_transactions.csv', 'r')
df_newtxn = pd.read_csv(csv4, header=0, index_col=None)

csv5 = open('train.csv', 'r')
df_train = pd.read_csv(csv5, header=0, index_col=None)

csv6 = open('test.csv', 'r')
df_test = pd.read_csv(csv6, header=0, index_col=None)

df_txn.head(3)
df_mer.head(3)
df_newtxn.head(3)
df_train.head(3)

sns.distplot(df_train.target)

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
   
df = train_test(df_train, df_test)

df.isnull().sum()
df_mer.isnull().sum()
df_newtxn.isnull().sum()
df_txn.merchant_id.value_counts()
df_txn.category_2.value_counts()
df_txn.category_2.value_counts()

