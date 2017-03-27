# -*- coding: utf-8 -*-
 
import pandas as pd
import numpy as np
 

 
titanic_df = pd.read_csv('/home/jian/DATA_SETS/kaggle/titanic/train.csv')
test_df = pd.read_csv('/home/jian/DATA_SETS/kaggle/titanic/test.csv')
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
 

titanic_df = missing_train(titanic_df)
test_df = missing_test(test_df)
 
# 特征工程
def featrue_engineering(data):
  
  embarked_dummies = pd.get_dummies(data['Embarked'])
  
  # embarked_dummies.drop(['S'], axis = 1, inplace=True)
  
  data = data.join(embarked_dummies)
  data.drop('Embarked', axis=1, inplace=True)
  
  # PClass
  pclass_dummies = pd.get_dummies(data['Pclass'])
  pclass_dummies.columns = ['Class_1','Class_2','Class_3']
  
  #pclass_dummies.drop(['Class_3'], axis = 1, inplace=True)
  
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
  
  #sex_dummies.drop(['male'], axis=1, inplace=True)
  
  data = data.join(sex_dummies)
  data.drop('Sex',axis=1, inplace=True)
  
  return data

# 调用feature 特征工程
titanic_df = featrue_engineering(titanic_df)
test_df = featrue_engineering(test_df)

 
 
y = titanic_df.pop('Survived')
X = titanic_df


print(test_df.columns)
test_id = test_df.pop('PassengerId')
X_test = test_df

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=30)
rfc.fit(X, y)

print('train over')


y_predict = rfc.predict(X_test)

submission = pd.DataFrame(data= {'PassengerId' : test_id, 'Survived': y_predict})
submission.to_csv("submission.csv", index=False)