# -*- coding: utf-8 -*-

import time 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
titanic_df = pd.read_csv('/home/jian/DATA_SETS/kaggle/titanic/train.csv')
test_df = pd.read_csv('/home/jian/DATA_SETS/kaggle/titanic/test.csv')


print(titanic_df.columns)

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
test_df = test_df.drop(['Name','Ticket'], axis=1)

print(titanic_df.info())
print(test_df.info())


fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(15, 8))
sns.countplot(x = 'Embarked', data=titanic_df, ax=axis1)
sns.countplot(x = 'Survived', hue='Embarked', data=titanic_df, ax=axis2)

test = titanic_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
print(test.index)
print(test.columns)
'''
详细分析一下 这个groupby

test = titanic_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()


首先看对象 print(test)  <pandas.core.groupby.DataFrameGroupBy object at 0x7f715700a310>

#需要注意的就是这个 as_index 参数
当 as_index 为True时，查看索引
print(test.index)
print(test.columns)

Index([u'C', u'Q', u'S'], dtype='object', name=u'Embarked')
Index([u'Survived'], dtype='object')

当 as_index 为False，查看索引
print(test.index)
print(test.columns)
 
Int64Index([0, 1, 2], dtype='int64')
Index([u'Embarked', u'Survived'], dtype='object')
'''
##########################################################################################
##########################################################################################
 
sns.barplot(x = 'Embarked', y = 'Survived', data = test, ax = axis3)
#plt.show()
 
 
# 计算该列null值个数 ,求个数 要转为 bool类型
avg_age = titanic_df['Age'].mean()
std_age = titanic_df['Age'].std()
count_nan_age = np.isnan(titanic_df['Age']).sum()
print(count_nan_age)
 
rand1 = np.random.randint(avg_age - std_age, avg_age + std_age, size=count_nan_age)

titanic_df['Age'].ix[np.isnan(titanic_df['Age'])] = rand1



