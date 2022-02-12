from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression #mutual_info_regression
from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np

#carico il training e il test
dftrain=pd.read_csv("trainingclean.csv")
ytrain=dftrain.pop('Survived')
Xtrain=dftrain
dftest=pd.read_csv("testclean.csv")
ytest=dftest.pop('Survived')
Xtest=dftest

#mutual info
bestfeatures = SelectKBest(score_func=mutual_info_regression, k=2)
fit = bestfeatures.fit(Xtrain,ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] #naming the dataframe columns
print('MUTUAL INFO REGRESSION')
print(featureScores)

#chi2
bestfeatures = SelectKBest(score_func=chi2, k=2)
fit = bestfeatures.fit(Xtrain,ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] #naming the dataframe columns
print('CHI2')
print(featureScores)

#normalizzazione delle variabili non codificate con one hot (versione nuova).
#ho provato a fare delle run anche con cabin non normalizzato ma sembrava avere scores appena più bassi
mean=np.mean(Xtrain[["Age","SibSp","Parch","Fare","relatives",'Cabin']], axis=0)
std=np.std(Xtrain[["Age","SibSp","Parch","Fare","relatives",'Cabin']], axis=0)
Xtrain[["Age","SibSp","Parch","Fare","relatives",'Cabin']]= (Xtrain[["Age","SibSp","Parch","Fare","relatives",'Cabin']]-mean)/std
Xtest[["Age","SibSp","Parch","Fare","relatives",'Cabin']]=(Xtest[["Age","SibSp","Parch","Fare","relatives",'Cabin']]-mean)/std

#risalvo i dati rielaborati in nuovi csv
Xtrain['Survived']=ytrain
Xtest['Survived']=ytest
Xtrain.pop('SibSp')
Xtest.pop('SibSp')
print(Xtrain.info())
with open("trainingstd.csv","w+") as f1:
    Xtrain.to_csv(f1, index=False)
with open("teststd.csv","w+") as f1:
    Xtest.to_csv(f1, index=False)
