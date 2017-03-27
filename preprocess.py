# -*- coding: utf-8 -*-
  
import numpy as np
import pandas as pd
 
# 处理训练数据
def missing_train(data):

  data = data.drop(['PassengerId','Name','Ticket'], axis=1)
  #print(data.info())
   
  data['Embarked'].fillna('S')
  
  data['Fare'] = data['Fare'].astype(int)
  
  avg_age = data['Age'].mean()
  std_age = data['Age'].std()
  count_nan_age = np.isnan(data['Age']).sum()
  rand1 = np.random.randint(avg_age - std_age, avg_age + std_age, size=count_nan_age)
  data['Age'].ix[np.isnan(data['Age'])] = rand1
  data['Age'] = data['Age'].astype(int)
  
  data.drop(['Cabin'], axis=1, inplace=True)
  
  
  return data 

# 处理测试数据
def missing_test(data):
  
  data = data.drop(['Name','Ticket'], axis=1)
 # print(data.info())
  
  # 这个 inplace=True 很重要 否则 空值没写进去
  
  data['Fare'].fillna(data['Fare'].median(), inplace=True)
  data['Fare'] = data['Fare'].astype(int)
  
  avg_age = data['Age'].mean()
  std_age = data['Age'].std()
  count_nan_age = np.isnan(data['Age']).sum()
  rand1 = np.random.randint(avg_age - std_age, avg_age + std_age, size=count_nan_age)
  data['Age'].ix[np.isnan(data['Age'])] = rand1
  data['Age'] = data['Age'].astype(int)
  
  data.drop(['Cabin'], axis=1, inplace=True)
  
  return data
  