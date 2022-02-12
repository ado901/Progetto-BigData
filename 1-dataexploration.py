import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

df= pd.read_csv('dataset.csv')
df.info()
X_train= pd.read_csv('trainingset.csv')
X_train.info()
X_test= pd.read_csv('testset.csv')
X_test.info()

print(X_train.describe(include='all'))
#controllo size del training e quanti valori mancanti abbiamo per feature
print(f"Numero istanze training: {len(X_train)}")
print('ISTANZE VUOTE training: ')
for i in X_train.columns:
    print(f"{i}: {X_train[i].isna().sum()}")
print('FREQUENZE PER OGNI CLASSE')
for i in ['Survived','Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Title']:
    print(X_train.groupby(i).size())


#frequenza passeggeri per feature
for feature in ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']:

    plt.hist(X_train.loc[~X_train[feature].isnull()][feature], bins=50)
    plt.title(feature)
    plt.show()

#mortalità in base al sesso
males= X_train.loc[df['Sex']=='male']
plt.hist(males[males['Survived']==1]['Age'].dropna(),bins=50, color='b',alpha =0.5, label='Sopravvissuti')
plt.hist(males[males['Survived']==0]['Age'].dropna(),bins=50, color='r', alpha=0.5, label='Morti')
plt.title('Maschi')
plt.legend()
plt.show()
females=X_train.loc[X_train['Sex']=='female']
plt.hist(females[females['Survived']==1]['Age'].dropna(),bins=50, color='b',alpha =0.5,label='Survived')
plt.hist(females[females['Survived']==0]['Age'].dropna(),bins=50, color='r', alpha=0.5, label='Dead')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Female')
plt.legend()
plt.show()

#mortalità in base alla classe
plt.hist(X_train[X_train['Survived']==1]['Pclass'].dropna(), bins=50, color='b', alpha =0.5,label='Survived')
plt.hist(X_train[X_train['Survived']==0]['Pclass'].dropna(), bins=50, color='r', alpha =0.5,label='Dead')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class')
plt.legend()
plt.show()

#mortalità in base alla tariffa (Fare)
plt.hist(X_train[X_train['Survived']==1]['Fare'],bins=50,color='b',alpha =0.5,label='Survived')
plt.hist(X_train[X_train['Survived']==0]['Fare'], bins=50, color='r', alpha =0.5,label='Dead')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Fare')
plt.legend()
plt.show()

freqTitle=X_train.groupby('Title').filter(lambda x: len(x)<=2)
df['Title']=df['Title'].replace(freqTitle['Title'].tolist(),'Unusual')
X_train['Title']=X_train['Title'].replace(freqTitle['Title'].tolist(),'Unusual')
X_test['Title']=X_test['Title'].replace(freqTitle['Title'].tolist(),'Unusual')

#potrebbe essere una feature interessante etichettare un passeggero in famiglia o da solo
X_train['relatives'] = X_train['SibSp'] + X_train['Parch']
X_train.loc[X_train['relatives'] > 0, 'Alone'] = 'No'
X_train.loc[X_train['relatives'] == 0, 'Alone'] = 'Yes'

X_test['relatives'] = X_test['SibSp'] + X_test['Parch']
X_test.loc[X_test['relatives'] > 0, 'Alone'] = 'No'
X_test.loc[X_test['relatives'] == 0, 'Alone'] = 'Yes'

df['relatives'] = df['SibSp'] + df['Parch']
df.loc[df['relatives'] > 0, 'Alone'] = 'No'
df.loc[df['relatives'] == 0, 'Alone'] = 'Yes'

#mortalità in base alla presenza di famigliari del passeggero
alone= X_train.loc[X_train['Alone']=='Yes']
plt.hist(alone[alone['Survived']==1]['Age'].dropna(),bins=50, color='b',alpha =0.5, label='Sopravvissuti')
plt.hist(alone[alone['Survived']==0]['Age'].dropna(),bins=50, color='r', alpha=0.5, label='Morti')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Alone')
plt.legend()
plt.show()
notAlone=X_train.loc[X_train['Alone']=='No']
plt.hist(notAlone[notAlone['Survived']==1]['Age'].dropna(),bins=50, color='b',alpha =0.5,label='Survived')
plt.hist(notAlone[notAlone['Survived']==0]['Age'].dropna(),bins=50, color='r', alpha=0.5, label='Dead')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Not Alone')
plt.legend()
plt.show()
print('\n RATIO SOPPRAVVIVENZA PER CLASSE:')
for i in ['Pclass','Sex','Age','Cabin','Embarked','Title','Alone']:
    print(X_train[[i, 'Survived']].groupby([i], as_index=False).mean().sort_values(by=i, ascending=True))
#numero cabine vuote per passeggeri 3 classe
dfclass3=X_train.loc[X_train['Pclass']==3]
print(f"numero passeggeri in terza classe: {len(dfclass3)}")
print(f"numero cabine vuote in terza classe: {len(dfclass3.loc[X_train['Cabin'].isna()])}")
dfclass2=X_train.loc[X_train['Pclass']==2]
print(f"numero passeggeri in seconda classe: {len(dfclass2)}")
print(f"numero cabine vuote in seconda classe: {len(dfclass2.loc[X_train['Cabin'].isna()])}")


#Considerando che quasi la totalità delle terze classi ha una cabina mancante e la restante parte è in buona parte nelle seconde,
# probabilmente è un dato volutamente lasciato vuoto
X_train['Cabin']=X_train['Cabin'].fillna('None')
X_test['Cabin']=X_test['Cabin'].fillna('None')
df['Cabin']=df['Cabin'].fillna('None')

#  associamo i due campi vuoti di Embarked a S, il più frequente
#potremmo anche cancellare direttamente le due righe ma vediamo come va prima
#tra l'altro, essendo anche nella stessa cabina, potrebbero con buona fiducia essere imbarcate nello stesso settore
print(X_train.loc[X_train['Embarked'].isnull()])
X_train['Embarked']=X_train['Embarked'].fillna('S')
X_test['Embarked']=X_test['Embarked'].fillna('S')
df['Embarked']=df['Embarked'].fillna('S')

#i ticket non sembrano essere una informazione significativa (oltre a non sapere il criterio dietro), quindi provo a toglierli
X_train.pop('Ticket')
X_test.pop('Ticket')
df.pop('Ticket')

#riempio i dati di Age mancanti con un valore casuale nel range della media e deviazione standard
meanAge=np.mean(X_train['Age'].dropna())
stdAge=np.std(X_train['Age'].dropna())
low=round(meanAge-stdAge)
high=round(meanAge+stdAge)
X_train.loc[X_train['Age'].isna(),'Age'] = X_train.loc[X_train['Age'].isna(), 'Age'].apply(lambda x:random.randint(low,high) )
X_test.loc[X_test['Age'].isna(),'Age'] = X_test.loc[X_test['Age'].isna(), 'Age'].apply(lambda x:random.randint(low,high) )
df.loc[df['Age'].isna(),'Age'] = df.loc[df['Age'].isna(), 'Age'].apply(lambda x:random.randint(low,high) )

#Cabin messi in ordinal encoder perchè potrebbe esserci un ordine di vicinanza tra le cabine della stessa lettera e/o numeri vicini
enc2=OrdinalEncoder()
enc2.fit(df[['Cabin']].sort_values(by=['Cabin']))
X_train['Cabin']=enc2.transform(X_train[["Cabin"]])
X_test['Cabin']=enc2.transform(X_test[["Cabin"]])
df['Cabin']=enc2.transform(df[["Cabin"]])

#one hot encoding manuale (odio il modo in cui viene gestito da sklearn)
X_train['Sex'] = X_train['Sex'].map( {'female': 1, 'male': 0}).astype(int)
X_test['Sex'] = X_test['Sex'].map( {'female': 1, 'male': 0}).astype(int)
df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0}).astype(int)
X_train['Alone'] = X_train['Alone'].map( {'Yes': 1, 'No': 0}).astype(int)
X_test['Alone'] = X_test['Alone'].map( {'Yes': 1, 'No': 0}).astype(int)
df['Alone'] = df['Alone'].map( {'Yes': 1, 'No': 0}).astype(int)


#nonostante le 2000 ricerche questo è il modo più funzionale che sono riuscito a ottenere per avere un one hot encoding
#niente ho deciso di non usare one hot perchè si creano troppe feature e non mi dava risultati abbastanza soddisfacenti comparati ad altri encoding
""" X_train=pd.get_dummies(data=X_train, columns=['Embarked', 'Title'])
X_test=pd.get_dummies(data=X_test, columns=['Embarked', 'Title'])
X_train, X_test = X_train.align(X_test,axis=1,fill_value=0) """
X_train['Embarked'] = X_train['Embarked'].map( {'S': 0, 'C': 1, 'Q':2})
X_test['Embarked'] = X_test['Embarked'].map( {'S': 0, 'C': 1, 'Q':2})

#TODO occhio sta parte
enc2=OrdinalEncoder()
enc2.fit(df[['Title']].sort_values(by=['Title']))
X_train['Title']=enc2.transform(X_train[["Title"]])
X_test['Title']=enc2.transform(X_test[["Title"]])
df['Title']=enc2.transform(df[["Title"]])

#salvo i dati puliti in dataset nuovi
with open("trainingclean.csv","w+") as f1:
    X_train.to_csv(f1, index=False)
with open("testclean.csv","w+") as f1:
    X_test.to_csv(f1, index=False)
