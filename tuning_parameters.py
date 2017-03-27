from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
 
from sklearn.model_selection import cross_val_score

tuned_parameters = [
                      {'n_estimators' : [15, 20, 30, 50, 75, 100, 200],'max_features': ['sqrt', 'log2', .1, .3, None]}
                    ]

def grid_search(X_train,y):
  
  #for score in scores:
    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10, n_jobs=-1)
    clf.fit(X_train, y)
    print clf.best_params_
    
    

