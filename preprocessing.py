import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import utils as utils
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
   
df = utils.train_test(df_train, df_test)

df.isnull().sum()
df_mer.isnull().sum()
df_newtxn.isnull().sum()
df_txn.merchant_id.value_counts()
df_txn.category_2.value_counts()
df_txn.category_2.value_counts()

df_ = feature_engineering (df_txn, df)
df_ = feature_engineering (df_newtxn, df_)

df_train = df_.loc[df_['target']!= -10000]
df_test = df_.loc[df_['target']== -10000]

for df in [df_train,df_test]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['dayofweek'] = df['first_active_month'].dt.dayofweek
    df['weekofyear'] = df['first_active_month'].dt.weekofyear
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    df['hist_first_buy'] = (df['purchase_date_min_x'] - df['first_active_month']).dt.days
    df['new_hist_first_buy'] = (df['purchase_date_min_y'] - df['first_active_month']).dt.days
    for f in ['purchase_date_max_x','purchase_date_min_x','purchase_date_max_y', 'purchase_date_min_y']:
        df[f] = df[f].astype(np.int64) * 1e-9
    df['card_id_total'] = df['card_id_size_y']+df['card_id_size_x']
    df['purchase_amount_total'] = df['purchase_amount_sum_x']+df['purchase_amount_sum_y']

df_train.to_csv("train_cleansing.csv", index=False)
df_test.to_csv('test_cleansing.csv', index=False)
