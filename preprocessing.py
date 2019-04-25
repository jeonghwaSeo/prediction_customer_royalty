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

df_txn = hist_feature_engineering (df_txn, )
