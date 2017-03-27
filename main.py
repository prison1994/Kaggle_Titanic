# -*- coding: utf-8 -*-

import time 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import preprocess
from featrue import featrue_engineering
from model_selection import model_selection
import tuning_parameters

 
titanic_df = pd.read_csv('/home/jian/DATA_SETS/kaggle/titanic/train.csv')
test_df = pd.read_csv('/home/jian/DATA_SETS/kaggle/titanic/test.csv')
print titanic_df.columns

titanic_df = preprocess.missing_train(titanic_df)
test_df = preprocess.missing_test(test_df)
 

# 调用feature 特征工程
titanic_df = featrue_engineering(titanic_df)
test_df = featrue_engineering(test_df)


features = titanic_df.columns.names
 
y = titanic_df.pop('Survived')
X = titanic_df
 
print(X.dtypes)
print(X.head(3))
#model_selection(X, y)
tuning_parameters.grid_search(X, y)