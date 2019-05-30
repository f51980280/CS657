#!/usr/bin/env python
# coding: utf-8

# In[68]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
from sklearn.preprocessing import StandardScaler

train_csv = pd.read_csv('train.csv') #read CSV檔 file
test_csv = pd.read_csv('test.csv')

train_df = pd.DataFrame(train_csv) #to panda dataFrame
test_df = pd.DataFrame(test_csv)



train_df = train_df.drop('Alley',1) ##以下移除的這4筆column很明顯與資料結果無關係
train_df = train_df.drop('PoolQC',1)
train_df = train_df.drop('MiscFeature',1)
train_df = train_df.drop('Fence',1)
train_df["GarageYrBlt"].fillna(0,inplace = True)  
train_df["Utilities"].fillna("None",inplace = True)  #把這幾筆Column的Nan補上None值
train_df["Exterior1st"].fillna("None",inplace = True)
train_df["Exterior2nd"].fillna("None",inplace = True)
train_df["FireplaceQu"].fillna("None",inplace = True)


test_df = test_df.drop('Alley',1) ##以下移除的這4筆column很明顯與資料結果無關係
test_df = test_df.drop('PoolQC',1)
test_df = test_df.drop('MiscFeature',1) 
test_df = test_df.drop('Fence',1)
test_df["GarageYrBlt"].fillna(0,inplace = True)
test_df["Utilities"].fillna("None",inplace = True)  #把這幾筆Column的Nan補上None值
test_df["Exterior1st"].fillna("None",inplace = True)
test_df["Exterior2nd"].fillna("None",inplace = True)
test_df["FireplaceQu"].fillna("None",inplace = True)


test_df = test_df.fillna(value=0)
train_df = train_df.fillna(value=0)

y_train = train_df["SalePrice"].values #轉成np
t = y_train
X_train = train_df = train_df.drop('SalePrice', 1)

for col_name in X_train.columns:
    if(X_train[col_name].dtype == 'object'):
        X_train[col_name]= X_train[col_name].astype('category')
        X_train[col_name] = X_train[col_name].cat.codes
        
for col_name in test_df.columns:
    if(test_df[col_name].dtype == 'object'):
        test_df[col_name]= test_df[col_name].astype('category')
        test_df[col_name] = test_df[col_name].cat.codes
        
        
X_test = test_df.values  #轉成np
X_train = X_train.values        


# In[ ]:





# In[69]:


## 先讓X與y去作標準化 使用StandardScaler()

y_train = y_train.reshape(1460,1)
SC_X = StandardScaler()
SC_y = StandardScaler()
 
X_train = SC_X.fit_transform(X_train)   ##原本是int標準化會有小數點自然會有數字會遺失出現警告
y_train = SC_y.fit_transform(y_train)
X_test = SC_X.transform(X_test) 

for i in range (len(X_train)):    
    X_train[i][0] = 1
for i in range (len(X_test)):
    X_test[i][0] = 1


# ### 我寫的Gradient_desent function  theta為要找出的角度,alpha為learning rate, num_iters為次數

# In[63]:


def gradient_Descent(X, y, theta, alpha, num_iters):
    m = len(y)
    n = len(theta)
    tmp = theta
    theta.reshape(n,1)
    y_pred = np.dot(X,theta)

    for i in range(num_iters):
        for j in range(n):
            tmp[j] = tmp[j] - alpha * (1/m)*sum((y_pred - y)*X[:,j])   ##計算theta最關鍵的就是這一行，也是微分過後的公式帶進去一直算
        y_pred = np.dot(X,theta)
        cost =  (1.0 / (2.0 * m)) * sum((y_pred-y)*(y_pred-y)) #這邊的Y都是標準化過後的Y
        print ("iteration {}, cost {}".format(i,cost))
    
    return theta   


# In[ ]:





# In[64]:


y_train = y_train.flatten()   #我丟進去的y要是一維,原本二維用這個涵式降維
theta = np.random.rand(len(X_train[0]))  #theta我採用隨機的數值
gradient_Descent(X_train,y_train,theta,0.1,500) #呼叫Gradient_descent  #看紀錄大致上500次就差不多了 alpha=0.1的原因是只要讓他可以穩定就好


# In[65]:


ANS = np.dot(X_test,theta)  #找出來的theta值與需要預測的資料內積

ANS_inv = ANS.reshape(1459,1)  ## 由於前面的"Price"有作標準化，在這邊要把預測出的結果給inverse回去
ANS_inv = SC_y.inverse_transform((ANS_inv))
ANS_inv = ANS_inv.flatten()

Id = np.zeros(len(X_test))

for i in range(len(X_test)):  ##寫檔用的Id
    Id[i]=i+1461
Id = Id.astype(np.int32)

f = open("submission.csv", "w")  ##寫檔
f.write("{},{}\n".format("Id", "SalePrice"))
for x in zip(Id, ANS_inv):
    f.write("{},{}\n".format(x[0], x[1]))
f.close()


# In[67]:


print(ANS_inv)


# In[ ]:




