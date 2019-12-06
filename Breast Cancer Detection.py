#!/usr/bin/env python
# coding: utf-8

# In[196]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# In[12]:


from pandas.plotting import scatter_matrix


# In[13]:


#loading the dataset
url='https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'


# In[14]:


names=['id','clump-thickness','uniform_cellsize','uniform_cell_shape','marginal_adhesion','single_pithelial_size','bare_nuclei',
      'bland_chromatin','normal_nucleoli','mitoses','class']


# In[44]:


df=pd.read_csv(url,names=names)


# In[45]:


df.shape


# In[46]:


#preprocessing
df.replace('?',-99999,inplace=True)


# In[47]:


df.shape


# In[48]:


df.columns


# In[49]:


len(df.columns)


# In[50]:


response=['class']


# In[156]:


predictors=df.columns[1:10]


# In[157]:


predictors


# In[158]:


X=df[predictors]
#X=np.array(X)


# In[159]:


y=df[response]
y=np.array(y)
y=y.ravel()


# In[160]:


X.shape


# In[161]:


y.shape


# In[57]:


X.hist(figsize=(10,10))
plt.show()


# In[58]:


scatter_matrix(df,figsize=(18,18))
plt.show()


# In[162]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# models=[]

# In[211]:


models=[]
seed = 0
scoring = 'accuracy'
X_test.shape


# In[212]:
19312661761

models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM', SVC()))
models.append(('Forest', RandomForestClassifier()))
models.append(('Decision', DecisionTreeClassifier()))


# In[213]:


models


# In[214]:


names=[]
results=[]


# In[155]:


X.columns


# In[215]:


for name, model in models:
    kfolds = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv= kfolds, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f %f' % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[216]:


for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))


# In[ ]:




