import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv('train_cleansing.csv')
df_test = pd.read_csv('test_cleansing.csv')

target = df_train['outliers']
df_train_out = df_train.drop (['target', 'outliers'], axis=1)

df_train_columns = [c for c in df_train_out.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = [c for c in df_train_columns if 'feature_' in c]

param = {
        'num_leaves': 10,
        'max_bin': 119,
        'min_data_in_leaf': 11,
        'learning_rate': 0.02,
        'min_sum_hessian_in_leaf': 0.00245,
        'bagging_fraction': 1.0, 
        'bagging_freq': 5, 
        'feature_fraction': 0.05,
        'lambda_l1': 4.972,
        'lambda_l2': 2.276,
        'min_gain_to_split': 0.65,
        'max_depth': 14,
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
    }
    
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
oof = np.zeros(len(df_train_out))
predictions_likelihood = np.zeros(len(df_test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train_out.values ,target.values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(df_train_out.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(df_train_out.iloc[val_idx][df_train_columns], label=target.iloc[val_idx], categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(df_train_out.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)
    
    predictions_likelihood += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

sns.distplot(predictions_likelihood)

df_train_without = df_train[df_train['outliers']==0]
target_without = df_train_without['target']
del df_train_without['target']

features_without = [c for c in df_train_without.columns if c not in ['card_id', 'first_active_month', 'outliers']]
categorical_feats_without = [c for c in features_without if 'feature_' in c]

param_ ={
         'task': 'train',
         'boosting': 'goss',
         'objective': 'regression',
         'metric': 'rmse',
         'learning_rate': 0.01,
         'subsample': 0.9855232997390695,
         'max_depth': 7,
         'top_rate': 0.9064148448434349,
         'num_leaves': 63,
         'min_child_weight': 41.9612869171337,
         'other_rate': 0.0721768246018207,
         'reg_alpha': 9.677537745007898,
         'colsample_bytree': 0.5665320670155495,
         'min_split_gain': 9.820197773625843,
         'reg_lambda': 8.2532317400459,
         'min_data_in_leaf': 21,
         'verbose': -1
         }
     
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
oof_ = np.zeros(len(df_train_without))
predictions_without = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train_without.values ,df_train_without['outliers'].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(df_train_without.iloc[trn_idx][features_without], label=target_without.iloc[trn_idx], categorical_feature=categorical_feats_without)
    val_data = lgb.Dataset(df_train_without.iloc[val_idx][features_without], label=target_without.iloc[val_idx], categorical_feature=categorical_feats_without)

    num_round = 10000
    clf = lgb.train(param_, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    oof_[val_idx] = clf.predict(df_train_without.iloc[val_idx][features_without], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features_without
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions_without += clf.predict(df_test[features_without], num_iteration=clf.best_iteration) / folds.n_splits

cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="Feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


df_sub_without = pd.DataFrame({"card_id":df_test["card_id"].values})
df_sub_without["target"] = predictions_without
df_sub_without.to_csv("submission_without_outlier.csv", index=False)

df_outlier_likelihood = pd.DataFrame({'card_id':df_test['card_id'].values})
df_outlier_likelihood['target'] = predictions_likelihood
df_outlier_likelihood.to_csv('outlier_likelihood.csv', index=False)
