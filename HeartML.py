#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
from IPython.display import display
plt.rc('font', family='Verdana')


# In[17]:


df = pd.read_csv('/home/shashy/Загрузки/heart-disease-uci/heart.csv')
df.head()


# In[18]:


df.shape


# In[19]:


heatmap = df.drop('target', axis=1)
X_raw = df.drop('target', axis = 1).values
y = df['target'].values


# In[20]:


sns.pairplot(df.head())


# In[21]:


sns.heatmap(heatmap.corr())


# In[22]:


sns.countplot(y)


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, random_state=0)


# In[24]:


print(X_train.shape, y_train.shape)


# In[25]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=100000).fit(X_train, y_train)


# In[26]:


print("Точность предсказания на тестовом наборе на основе на Логистической регрессии c C=100000 составила: {:3f}".format(logreg.score(X_test, y_test)))
print("Точность предсказания на основе тренировочного набора на Логистической регрессии c C=100000 составила: {:3f}".format(logreg.score(X_train, y_train)))


# In[27]:


from sklearn.svm import LinearSVC
svc = LinearSVC(C=100000).fit(X_train, y_train)


# In[28]:


print("Точность предсказания на тестовом наборе на основе линейного метода опорных векторов с C=100000 составила: {:3f}".format(svc.score(X_test, y_test)))
print("Точность предсказания на основе тренировочного набора на основе линейного метода опорных векторов с C=100000 составила: {:3f}".format(svc.score(X_train, y_train)))


# In[ ]:




