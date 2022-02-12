import pandas as pd
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
import numpy as np
#training e test set
df=pd.read_csv("trainingstd.csv")
y=df.pop('Survived')
test=pd.read_csv('teststd.csv')
y_test=test.pop('Survived')



print('SVC:')
#ricavo i parametri migliori del modello
scoresdf=pd.read_csv('scorestuningSVC.csv')
first=scoresdf.loc[scoresdf['rank_test_score']==2,'params']
first=first.get(1)
bestparameters=eval(first)
#fitting finale e test
param=[[],[],[]]
for i in range(0,10):
    model = SVC(**bestparameters)
    model.fit(df,y)
    y_pred=model.predict(test)
    param[0].append(precision_score(y_test,y_pred))
    param[1].append(recall_score(y_test,y_pred))
    param[2].append(f1_score(y_test,y_pred))
print(f'PRECISION SCORE: \n{np.mean(param[0])}')
print(f'RECALL SCORE: \n{np.mean(param[1])}')
print(f'F1 SCORE: \n{np.mean(param[2])}')

print('\nRandom Forest:')
#ricavo i parametri migliori del modello
scoresdf=pd.read_csv('gridtuningforest.csv')
first=scoresdf.loc[scoresdf['rank_test_score']==2,'params']
first=first.get(1)
bestparameters=eval(first)
#fitting finale e test
param=[[],[],[]]
for i in range(0,10):
    model = RandomForestClassifier(**bestparameters)
    model.fit(df,y)
    y_pred=model.predict(test)
    param[0].append(precision_score(y_test,y_pred))
    param[1].append(recall_score(y_test,y_pred))
    param[2].append(f1_score(y_test,y_pred))
print(f'PRECISION SCORE: \n{np.mean(param[0])}')
print(f'RECALL SCORE: \n{np.mean(param[1])}')
print(f'F1 SCORE: \n{np.mean(param[2])}')

