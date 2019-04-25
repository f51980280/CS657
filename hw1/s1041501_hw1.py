#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

train_csv = pd.read_csv('train.csv') #read CSV檔 file
test_csv = pd.read_csv('test.csv')

train_df = pd.DataFrame(train_csv) #to panda dataFrame
test_df = pd.DataFrame(test_csv)


# In[2]:


#補上Age空格部分 直接使用平均值
#fill age column for Nan
train_df_fix = train_df.fillna({'Age': train_df['Age'].mean()})


# In[3]:


#把Nmame部分做整理 增加title2的欄位 把不為Miss Mrs Mr Master的整理成這幾類
train_df_fix["Title"] = train_df_fix["Name"].str.split(",", expand = True)[1]
train_df_fix["Title"] = train_df_fix["Title"].str.split(".",expand = True)[0]
train_df_fix['Title2'] = train_df_fix['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Col','Don','Jonkheer','Capt','Rev','Sir']
                                                    ,['Miss','Mrs','Miss','Mr','Mr','Mrs','Mrs','Miss','Mr','Mr','Mr','Mr','Mr'], regex=True)
train_df_fix['Title2'].unique()


# In[4]:


#Embark為上船地 空格部分補上最常出現的的S
train_df_fix['Embarked'].describe()
train_df_fix['Embarked'] = train_df_fix['Embarked'].fillna('S')


# In[5]:


#把survivied欄位0,-1做轉換
train_df_fix["Survived"] = train_df_fix["Survived"].replace(0,-1)


# In[7]:


#把所有data轉成int的型態
train_df_fix["PassengerId"] = train_df_fix["PassengerId"].astype('category').cat.codes
train_df_fix["Pclass"] = train_df_fix["Pclass"].astype('category').cat.codes
train_df_fix["Name"] = train_df_fix["Name"].astype('category').cat.codes
train_df_fix["Sex"] = train_df_fix["Sex"].astype('category').cat.codes
train_df_fix["Age"] = train_df_fix["Age"].astype('category').cat.codes
train_df_fix["SibSp"] = train_df_fix["SibSp"].astype('category').cat.codes
train_df_fix["Parch"] = train_df_fix["Parch"].astype('category').cat.codes
train_df_fix["Ticket"] = train_df_fix["Ticket"].astype('category').cat.codes
train_df_fix["Fare"] = train_df_fix["Fare"].astype('category').cat.codes
train_df_fix["Cabin"] = train_df_fix["Cabin"].astype('category').cat.codes
train_df_fix["Embarked"] = train_df_fix["Embarked"].astype('category').cat.codes
train_df_fix["Title"] = train_df_fix["Title"].astype('category').cat.codes
train_df_fix["Title2"] = train_df_fix["Title2"].astype('category').cat.codes


# In[8]:


#判斷1 or -1
def sign(z):
    if z > 0:
        return 1
    else:
        return -1


# In[9]:


#把survived取出來並調整維度為Y
train_np = train_df_fix.values
Y = train_np[:,1]
Y = Y.reshape(891,1)
train_np[0]


# In[10]:


#把不必要的特徵給刪除 survived, Name, title, ticket, fare, Cabin 因為看似跟生存結果無關
train_np = np.delete(train_np,1,axis=1) # delete survived data
train_np = np.delete(train_np,2,axis=1) # delete Name
train_np = np.delete(train_np,10,axis=1)# delete title
train_np = np.delete(train_np,6,axis=1)#ticeket
train_np = np.delete(train_np,6,axis=1)#fare
train_np = np.delete(train_np,6,axis=1)#Cabin


# In[33]:


#接著把第一維的PassengerId都設為一
num = len(train_np)
for i in range(num):
    train_np[i,0]=1


# In[67]:


#Pocket_PLA algorithm
m,n = train_np.shape
update = 0
min_err = m
max_update = 25000  #執行次數
alpha = 0.05    #權重調整成長係數 #感覺上0.1以下調整的都差不多
err=0
wt = np.random.rand(8) #使用為隨機向量 
min_w = np.random.rand(8)
random.seed(1024 * 1024)

###Pocket_PLA 基本就是因為大多數的資料都不是線性可分 使用greedy的方式與原本PLA方法一樣再暴力找出最佳權重的方式

while update <= max_update:
    err = 0
    print(update)
    i = random.randint(0, m - 1)                         #很多種用隨機的方式挑 這種方式感覺最快最準
    if np.sign(np.dot(wt,train_np[i])) != np.sign(Y[i]): #隨機挑一筆來看上次的Wt是否正確 若不正確就修改
        wt += alpha * train_np[i].T * Y[i]   
        update +=1   
        for k in range(num):                   #算錯誤數
            if np.sign(np.dot(wt,train_np[k])) != np.sign(Y[k]):
                err+=1
        print(err)
        print(min_err)
        if err <= min_err:
            min_w = wt.copy() #python很多list物件要assign都會assign reference所以要copy的方式
            print("變")  #看權重哪時候改變
            min_err = err
    else:
        continue
    print(wt)
    print(min_w)


# In[68]:


print(min_w)


# In[23]:


#處理 Test.csv 與 train.csv方式相同
#補上Age空格部分 直接使用平均值
test_df_fix = test_df.fillna({'Age': train_df['Age'].mean()})

test_df_fix["Title"] = test_df_fix["Name"].str.split(",", expand = True)[1]
test_df_fix["Title"] = test_df_fix["Title"].str.split(".",expand = True)[0]
test_df_fix['Title2'] = test_df_fix['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Col','Don','Jonkheer','Capt','Rev','Sir']
                                                    ,['Miss','Mrs','Miss','Mr','Mr','Mrs','Mrs','Miss','Mr','Mr','Mr','Mr','Mr'], regex=True)


# In[25]:


##Embark為上船地 空格部分補上最常出現的的S
test_df_fix['Embarked'] = test_df_fix['Embarked'].fillna('S')


# In[26]:


test_df_fix["PassengerId"] = test_df_fix["PassengerId"].astype('category').cat.codes
test_df_fix["Name"] = test_df_fix["Name"].astype('category').cat.codes
test_df_fix["Age"] = test_df_fix["Age"].astype('category').cat.codes
test_df_fix["Sex"] = test_df_fix["Sex"].astype('category').cat.codes
test_df_fix["Pclass"] = test_df_fix["Pclass"].astype('category').cat.codes
test_df_fix["SibSp"] = test_df_fix["SibSp"].astype('category').cat.codes
test_df_fix["Parch"] = test_df_fix["Parch"].astype('category').cat.codes
test_df_fix["Fare"] = test_df_fix["Fare"].astype('category').cat.codes
test_df_fix["Cabin"] = test_df_fix["Cabin"].astype('category').cat.codes
test_df_fix["Embarked"] = test_df_fix["Embarked"].astype('category').cat.codes
test_df_fix["Ticket"] = test_df_fix["Ticket"].astype('category').cat.codes
test_df_fix["Title"] = test_df_fix["Title"].astype('category').cat.codes
test_df_fix["Title2"] = test_df_fix["Title2"].astype('category').cat.codes


# In[27]:


#PassengerId 第一維全設為1
test_len = len(test_df_fix)
test_np = test_df_fix.values
for i in range(test_len):
    test_np[i,0]=1 


# In[28]:


#把不必要的特徵刪除
test_np = np.delete(test_np,2,axis=1) # delete Name
test_np = np.delete(test_np,10,axis=1)# delete test
test_np = np.delete(test_np,6,axis=1)#ticeket
test_np = np.delete(test_np,6,axis=1)#fare
test_np = np.delete(test_np,6,axis=1)#Cabin


# In[70]:


#計算出Test sequence
sub = np.zeros(len(test_np))
PassId = np.zeros(len(test_np))
sur = 0
died = 0
ID=0
for i in range(len(test_np)):
    ID=892
    PassId[i] += ID+i 
    if np.sign(np.dot(min_w,test_np[i])) > 0:
            sub[i]=1
            sur+=1
    else:
            died+=1
            sub[i]=0

sub = sub.astype(np.int32)
PassId = PassId.astype(np.int32)


# In[73]:


#寫成CSV檔案
f = open("submission.csv", "w")
f.write("{},{}\n".format("PassengerId", "Survived"))
for x in zip(PassId, sub):
    f.write("{},{}\n".format(x[0], x[1]))
f.close()


# In[ ]:





# In[ ]:




