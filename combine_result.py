import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, log_loss
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

sub_without = pd.read_csv('submission_without_outlier.csv')
out_likelihood = pd.read_csv('outlier_likelihood.csv')
sub_with = pd.read_csv('submission1.csv')
