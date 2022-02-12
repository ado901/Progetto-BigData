
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.svm import SVC
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)

#leggo i csv
df=pd.read_csv("trainingstd.csv")
y=df.pop('Survived')

#setting dei parametri del grid del modello
model = SVC()
cs = [1,5,10,15,20,25,30,35,40]
kernel=['linear', 'rbf']
gamma = [0.1, 1, 10, 100,'scale','auto']
class_weight=['balanced',None]
#creo il parametro da mettere nella randomizedsearch
random_grid = {
'C': cs,
"kernel":kernel,
'gamma': gamma,
'class_weight':class_weight,
}

#inizializzo la grid
svc_grid=GridSearchCV(estimator=model,param_grid=random_grid, cv=5, verbose=3,return_train_score=True,scoring='f1', n_jobs=1)

#fitto
svc_grid.fit(df,y)

#elaboro il dict dei risultati ordinati per rank della random in modo da salvarli in un csv
scores=pd.DataFrame.from_dict(svc_grid.cv_results_)
scores= scores.sort_values(by='rank_test_score')
scores.reset_index(drop=True, inplace=True)

#salvo in csv i risultati
with open("scorestuningSVC.csv","w+") as f1:
    scores.to_csv(f1, index=False)

#semplice print dei parametri dei primi in rank
parameters=scores['params']
for i in parameters.head(10):
    print(i)
    print()

import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)









