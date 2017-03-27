# -*- coding: utf-8 -*-
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 探索特征，特征工程，具体来说就是去掉冗余特征，构造出新的特征

def feature_explore(data):
  print(data.head())
  #Embarked
  
  fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(15, 8))
  sns.countplot(x = 'Embarked', data=data, ax=axis1)
  sns.countplot(x = "Survived", hue='Embarked', data=data, ax=axis2)
  
  embark_mean = data[['Survived', 'Embarked']].groupby(['Embarked'], as_index=False).mean()
  sns.barplot(x='Embarked', y='Survived', data=embark_mean, order=['S', 'C', 'Q'], ax=axis3)
  plt.show()
  
  
  # age
    

# 特征工程
def featrue_engineering(data):
  
  embarked_dummies = pd.get_dummies(data['Embarked'])
  embarked_dummies.drop(['S'], axis = 1, inplace=True)
  data = data.join(embarked_dummies)
  data.drop('Embarked', axis=1, inplace=True)
  
  # PClass
  pclass_dummies = pd.get_dummies(data['Pclass'])
  pclass_dummies.columns = ['Class_1','Class_2','Class_3']
  pclass_dummies.drop(['Class_3'], axis = 1, inplace=True)
  data = data.join(pclass_dummies)
  data.drop('Pclass', axis=1, inplace=True)
  
  # 将 SibSp 和 Parch合并成 家人一项
  data['Family'] = data['SibSp'] + data['Parch']
  data['Family'].ix[data['Family'] > 0] = 1
  data['Family'].ix[data['Family'] == 0] = 1
  data.drop(['SibSp', 'Parch'], axis=1, inplace=True)
  # Sex Age 写成 合并成 Children Male Female
  
  data['Sex'].ix[data['Age'] < 18] = 'child' 
  data.drop('Age', axis=1, inplace=True)
  sex_dummies = pd.get_dummies(data['Sex'])
  sex_dummies.drop(['male'], axis=1, inplace=True)
  data = data.join(sex_dummies)
  data.drop('Sex',axis=1, inplace=True)
  
  return data

  
  