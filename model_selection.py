from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


def model_selection(X, y):
  
  rfc = RandomForestClassifier(n_estimators= 200, max_features=.3)
  etc = ExtraTreesClassifier(n_estimators=200, max_features=.3)
  
  print cross_val_score(rfc, X, y, scoring='accuracy', cv = 10)
  print cross_val_score(etc, X, y, scoring='accuracy', cv = 10)

