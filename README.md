# ML Project for Movie Recommendation
 
## Introduction
 In this project a machine learning model is built which can be used to analyse that whether a person/user will like a movie or not.If he likes it then 
that movie can be recommended to him to watch or that movie would get higher priority in his/her search results.


## Working Model
We do our prediction based on genres of the movie and the reviews which were gathered from various users at various time intervals.
We have used regression initially to get the rating's . The method used for regression is **Random Forest regression** which comes from python's _ScikitLearn
Library_.


 ## Code
 ### Importing Libraries
 ```python
import pandas as pd
import numpy as np
import random as rd
```
### Creating Datasets
```python
dataset =pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
```
### Reducing Dataset Size
Since the dataset size is very huge so for our analysis we pick only a part of it.For elimination purpose we are using randon function.
This is done to maintain hetrogeneity in the dataset.
```python
a=rd.sample(range(100835),80000)
for i in range(80000):
     b=a[i]
     ratings=ratings.drop(axis=0,index=b)
```

### Binarization of Generes

```python
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
```
### Removing Insignificant Columns
```python
        
dataset = dataset.drop(['genres','IMAX','(no genres listed)'] , axis='columns')

```
### Merging Datasets
```python
dataset = pd.merge(ratings,dataset,how='inner',on='movieId')
```

### Creating numpy Arrays from dataset
```python
y=dataset.iloc[:,2]#dependent variable
X=dataset.iloc[:,0:23]#array of independent variables
X=X.drop(['movieId','rating','title'],axis='columns')#removing unwanted features from independent array
```
### Transforming User_Id Into categorical variables
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.values[:, 0] = labelencoder_X.fit_transform(X.values[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
```
### Creating trainset and testset
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_test=np.true_divide(y_test,5)
y_test=np.round(y_test).astype(bool)
#since we have randomly splitted data into trainset and testset so indexing in y_test doen't start from 0 and is not unlike y_pred(predicted values)
#so we create y_test2 which is similar to y_test but whose indexing is seqential and starts from 0
y_test2=list(y_test)
y_test2=np.asarray(y_test2)
```
### Building Model
```python
for i in range(y_test.shape[0]):
    np.append(arr=y_test2,values=y_test.values[i,0],axis=0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X_train, y_train)
```
### Predicting test data
```python
y_pred=regressor.predict(X_test)
```

### Converting predicted result into boolean type
```python
y_pred=np.true_divide(y_pred,5)
y_pred=np.round(y_pred).astype(bool)
```

### Builiding confusion Matrix
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test2, y_pred)
```


## Estimating Model
- Accuracy = ((190+3272)/(190+3272+570+136))*100
