from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier,BaggingClassifier,RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import warnings
import matplotlib.pyplot as plt
from neuralnetwork import neuralModel
from sklearn.model_selection import KFold

#funzione che fa la kfold sul dict di modelli come parametro
def trainModel(models,df,y):
    reviews={} #dizionario {modello: [scores]} che ritorneremo dalla funzione
    n_fold=10
    #inizializzazione di reviews
    for key in models:
        reviews[key]=[0,0,0]
    reviews['Neural Network']=[0,0,0]
    
    #qui il codice è uguale a quello del prof
    kf =KFold(n_splits=n_fold, random_state=None, shuffle=False)
    for train_index, validation_index in kf.split(df):
        Xtrain = df.iloc[train_index]
        X_validation = df.iloc[validation_index]
        ytrain= y[train_index]
        y_validation = y[validation_index]

        #per ogni modello fitto e prendo gli scores
        for key in models:
            models[key].fit(Xtrain,ytrain)
            y_pred=models[key].predict(X_validation)
            prec= precision_score(y_validation,y_pred)
            recall= recall_score(y_validation,y_pred)
            f1=f1_score(y_validation,y_pred)
            #devo ricavare la media di tutta la kfold per ogni modello
            #-> sommo i parametri a quelli del fold precedente e alla fine di tutto divido per n_fold
            zipped_lists = zip(reviews[key], [prec,recall,f1])
            reviews[key]=[x + y for (x, y) in zipped_lists]
        
        #applico lo stesso per la rete neurale
        nnprecision, nnrecall, nnf1=neuralModel(Xtrain,ytrain,X_validation,y_validation)
        zipped_lists = zip(reviews['Neural Network'], [nnprecision,nnrecall,nnf1])
        reviews['Neural Network']=[x + y for (x, y) in zipped_lists]

    for key in reviews: #divisione per ottenere media
        reviews[key]=[x / n_fold for x in reviews[key]]
    print(reviews)
    return reviews

#warning che non ho capito su adaboost, soppresso
warnings.filterwarnings("ignore")

#carico il training, no testset
df=pd.read_csv("trainingstd.csv")
y=df.pop('Survived')


#elenco dei modelli che voglio confrontare, più avanti verrà aggiunto hard voting e neural network
modelsdict={
    "BAGGING CLASSIFIER":BaggingClassifier(n_estimators = 100),
    "RANDOM FOREST CLASSIFIER":RandomForestClassifier(),
    "DECISION TREE CLASSIFIER":DecisionTreeClassifier(),
    "ADABOOST":AdaBoostClassifier(),
    "GRADIENT BOOSTING": GradientBoostingClassifier(),
    "LOGISTIC REGRESSION":LogisticRegression(solver='liblinear'),
    'SVC':SVC(),
    'SGDClassifier':SGDClassifier(),

}


#hard voting
log_clf = LogisticRegression(solver="liblinear")
ada_clf=AdaBoostClassifier()
gr_clf=GradientBoostingClassifier()
rf_clf= RandomForestClassifier(n_estimators = 100)
svm_clf = SVC()
voting_clf=VotingClassifier(estimators=[('lr', log_clf),('ada', ada_clf),('gr', gr_clf),('rf', rf_clf),('svc', svm_clf)], voting='hard')
modelsdict["VOTING"]=voting_clf

#chiamo la funzione che farà le fold
reviews=trainModel(modelsdict,df,y)

#trasformo in un dataframe il risultato
reviews= pd.DataFrame.from_dict(reviews,orient='index',columns=['Precision','Recall','F1'])
reviews['Model'] = reviews.index
reviews.reset_index(drop=True, inplace=True)

#grafico ordinato per ogni score
precisions= reviews[['Precision','Model']].sort_values('Precision')
recalls=reviews[['Recall', 'Model']].sort_values('Recall')
f1s=reviews[['F1','Model']].sort_values('F1')
print(f1s)
plt.barh(precisions['Model'], precisions['Precision'], color = 'red')
plt.title('Precision Scores')
plt.show()

plt.barh(recalls['Model'], recalls['Recall'], color = 'red')
plt.title('Recall Scores')
plt.show()

plt.barh(f1s['Model'], f1s['F1'], color = 'red')
plt.title('F1 Scores')
plt.show()