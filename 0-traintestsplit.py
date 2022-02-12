import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

df= pd.read_csv('train.csv')
y=df.pop('Survived')


#Elimino dati sensibili per garantire anonimato
#prima ricavo il titolo, potrebbe essere una informazione utile
df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
df.pop('Name')
df.pop('PassengerId')

#divido train da test
X_train, X_test, y_train,y_test= train_test_split(df,y,test_size=0.30)
X_train['Survived']=y_train
X_test['Survived']=y_test
df['Survived']=y
with open("trainingset.csv","w+") as f1:
    X_train.to_csv(f1, index=False)
with open("testset.csv","w+") as f1:
    X_test.to_csv(f1, index=False)

with open("dataset.csv","w+") as f1:
    df.to_csv(f1, index=False)