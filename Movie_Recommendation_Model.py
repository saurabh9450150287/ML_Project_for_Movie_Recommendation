
import pandas as pd
import numpy as np
import random as rd
dataset =pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
a=rd.sample(range(100835),80000)
for i in range(80000):
     b=a[i]
     ratings=ratings.drop(axis=0,index=b)
i=0
dataset['Action']=0
dataset['Adventure']=0
dataset['Animation']=0
dataset['Children']=0
dataset['Comedy']=0
dataset['Crime']=0
dataset['Documentary']=0
dataset['Drama']=0
dataset['Fantasy']=0
dataset['Film-Noir']=0
dataset['Horror']=0
dataset['Musical']=0
dataset['Mystery']=0
dataset['Romance']=0
dataset['Sci-Fi']=0
dataset['Thriller']=0
dataset['War']=0
dataset['Western']=0
for i in range(0, dataset.shape[0]):
    string = str(dataset.iloc[i,2])
    array = string.split('|')
    for j in array :
        dataset.loc[i,j]=1
dataset = dataset.drop(['genres','IMAX','(no genres listed)'] , axis='columns')
dataset = pd.merge(ratings,dataset,how='inner',on='movieId')
y=dataset.iloc[:,2]
X=dataset.iloc[:,0:23]
X=X.drop(['movieId','rating','title'],axis='columns')

    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.values[:, 0] = labelencoder_X.fit_transform(X.values[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test=np.true_divide(y_test,5)
y_test=np.round(y_test).astype(bool)
y_test2=list(y_test)
y_test2=np.asarray(y_test2)
y_test2.values[:,0]=y_test.values[:,0]

print(y_test.values[794,0])
for i in range(y_test.shape[0]):
    np.append(arr=y_test2,values=y_test.values[i,0],axis=0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)
y_pred=np.true_divide(y_pred,5)
y_pred=np.round(y_pred).astype(bool)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test2, y_pred)