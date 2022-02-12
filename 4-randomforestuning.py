
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)

#leggo i csv
df=pd.read_csv("trainingstd.csv")
y=df.pop('Survived')

#setting dei parametri del grid del modello
model = RandomForestClassifier(n_jobs=-1)
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 15)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
#creo il parametro da mettere nella randomizedsearch
random_grid = {'n_estimators': n_estimators,
'max_features': max_features,
"min_samples_split":min_samples_split,
'max_depth': max_depth,
'min_samples_leaf':min_samples_leaf,
'bootstrap': bootstrap,
}

#inizializzo la randomsearch
rf_random = RandomizedSearchCV(estimator = model,param_distributions = random_grid,n_iter = 1000, random_state=35, n_jobs = -1, verbose=3,return_train_score=True, scoring='f1')

#fitto
rf_random.fit(df,y)

#elaboro il dict dei risultati ordinati per rank della random in modo da salvarli in un csv
scores=pd.DataFrame.from_dict(rf_random.cv_results_)
scores= scores.sort_values(by='rank_test_score')
scores.reset_index(drop=True, inplace=True)

#salvo in csv i risultati
with open("randomtuningforest.csv","w+") as f1:
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









