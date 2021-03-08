#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[32]:


df = pd.read_csv(r'C:/Users/Success/Desktop/fruits.csv')


# In[33]:


df.head()


# In[34]:


df.describe()


# In[35]:


df.mean()


# In[36]:


df.columns


# In[37]:


predct = dict(zip(df.fruit_label.unique(), df.fruit_name.unique()))   
predct


# In[38]:


fname = df['fruit_name'].unique()
fname


# In[39]:


fsize = df.groupby('fruit_name',sort = False).size()
fsize


# In[40]:


print(type(fsize))
print(fsize.dtype)


# In[41]:


a =  plt.subplot(211)
a.scatter(df['width'],df['height'])
a.set_title('Width vs Height')
a =  plt.subplot(212)
a.scatter(df['width'],df['color_score'])
a.set_title('Width vs Colorscore')
plt.xlabel('Fruits')
plt.ylabel('Quantity')
plt.title('Fruits Category')
plt.show()


# In[44]:


x = df.iloc[:, :-1].values
print(x)
y = df.iloc[:, -1].values
print(y)


# In[51]:


from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
df['fruit_name'] = en.fit_transform(df['fruit_name'])


# In[52]:


v1=x
v2=y


# In[55]:


class KNearestNeighbors(object):
    def __init__(self,k):
        self.k=k
    def euclidean_distance(v1,v2):
        v1,v2 = np.array(v1),np.array(v2)
        dist=0
        for i in range(len(v1)-1):
            dist+=(v1[i]-v2[i]) ** 2
        return np.sqrt(dist)
    def predict(k,train,test):
        dist=[]
        for i in range(len(train)):
            dist=euclidean_distance(train[i][:-1],test)
            distances.append((train[i],dist))
        distances.sort(key=lambda x: x[1])
        neighbors=[]
        for i in range(k):
            neighbors.append(distances[i][0])
        classes={}
        for i in range(len(neighbors)):
            response=neighbors[i][-1]
            if response in classes:
                classes[response]+=1
            else:
                classes[response]=1
        sorted_class=sorted(classes.items(),key=lambda x: x[1],reverse=True)
        return sorted_class[0][0]
    def evaluate(y_true,y_pred):
        nc=0
        for act,pred in zip(y_true,y_pred):
            if act==pred:
                nc+=1
        return nc/len(y_true)
    def train_test_split(df,test_size=0.2):
        ntest=int(len(df)*test_size)
        test_set=df.sample(ntest)
        train_set=[]
        for ind in df.index:
            if ind in test_set.index:
                continue
            train_set.append(df.iloc[ind])
    
        train_set=pd.DataFrame(train_set).astype(float).values.tolist()
        test_set=test_set.astype(float).values.tolist()
    
        return train_set,test_set


# In[56]:


knn=KNearestNeighbors(k=3)
preds=[]

for row in test_set:
    predictors=row[:-1]
    pred=knn.predict(train_set,predictors)
    preds.append(pred)

actual=np.array(test_set)[:,-1]
return knn.evaluate(actual,preds)


# In[57]:


k_eval=[]
for k in range(1,25):
    knn=KNearestNeighbors(k=k)
    preds=[]
    
    for row in test_set:
        predictors=row[:-1]
        pred=knn.predict(train_set,predictors)
        preds.append(pred)
    curr_accuracy=knn.evaluate(actual,preds)
    k_eval.append((k,curr_accuracy))
k_eval


# In[58]:


print("Accuracy",k_eval)


# In[ ]:




