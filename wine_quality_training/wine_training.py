#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports for data exploration and analysis
import pandas as pd
import numpy as np
#imports for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
#importing models
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

import pickle

import warnings 
warnings.filterwarnings('ignore')


# In[2]:


#training data
wine_train_df = pd.read_csv('wine_quality_training/wine_data/train_wine_data.csv') 
wine_train_df.head()


# In[3]:


wine_train_df.info()


# In[4]:


wine_train_df.isnull().sum()


# In[5]:


wine_train_df.describe()


# In[6]:


# create box plots
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()
for col, value in wine_train_df.items():
 if col != 'type':
    sns.boxplot(y=col, data=wine_train_df, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[7]:


#checking which values affect wine quality
# Exclude non-numeric columns
numeric_columns = wine_train_df.select_dtypes(include=['float64', 'int64']).columns
wine_test = wine_train_df[numeric_columns].corr()

plt.figure(figsize=(20, 20))
sns.heatmap(wine_test, cmap='Blues', annot=True)



# In[8]:


#checking if the type of wine will affect the quality
sns.barplot(data=wine_train_df, x='type', y='quality')
plt.show()


# In[9]:


#distribution of wine quality
sns.distplot(wine_train_df['quality'])


# In[10]:


wine_train_df['quality'].describe(percentiles=[0.95, 0.98, 0.99])


# In[11]:


from sklearn.preprocessing import OneHotEncoder

#encoding the wine types as 1 and 0 with respect to red to be computated

encoder = OneHotEncoder(sparse=False)

encode_types = encoder.fit_transform(wine_train_df['type'].values.reshape(-1, 1))
encode_types


# In[12]:


encoder.categories_


# In[13]:


label= ['red','white']
wine_types = pd.DataFrame(encode_types, columns= label)


# In[14]:


wine_types.value_counts()


# In[15]:


wine_types['red'].value_counts()


# In[16]:


#confirming the presence of the encoded column
wine_train_df['red_wine'] = wine_types['red']

wine_train_df.head()


# In[17]:


X_train = wine_train_df.drop(columns=['type','quality'])
X_train


# In[18]:


target = 'quality'
y_train = wine_train_df[target].values
y_train


# In[19]:


#trying with random forest regression

rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)

y_pred = rfr.predict(X_train)

rmse = mean_squared_error(y_train,y_pred,squared=False)
accuracy = r2_score(y_train,y_pred)

print('RandomForestRegressor')
print (f'RMSE: ',{rmse})
print(f'Accuracy: ',{accuracy})


# In[20]:


sns.distplot(y_pred, label = 'prediction')
sns.distplot(y_train, label = 'actual')
plt.title(f'RandomForestRegressor')

plt.legend()


# In[21]:


#trying with random forest classifier

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_train)

rmse = mean_squared_error(y_train,y_pred,squared=False)
accuracy = r2_score(y_train,y_pred)

print('RandomForestClassifier')
print (f'RMSE: ',{rmse})
print(f'Accuracy: ',{accuracy})


# In[22]:


sns.distplot(y_pred, label = 'prediction')
sns.distplot(y_train, label = 'actual')
plt.title('RandomForestClassifier')

plt.legend()


# In[23]:


#trying with extra trees classifier

etc = ExtraTreesClassifier(n_estimators=100)
etc.fit(X_train, y_train)

y_pred = etc.predict(X_train)

rmse = mean_squared_error(y_train,y_pred,squared=False)
accuracy = r2_score(y_train,y_pred)

print('ExtraTreesClassifier')
print (f'RMSE: ',{rmse})
print(f'Accuracy: ',{accuracy})


# In[24]:


sns.distplot(y_pred, label = 'prediction')
sns.distplot(y_train, label = 'actual')
plt.title('ExtraTreesClassifier')

plt.legend()


# In[25]:


#trying with extra trees classifier

dtc =  DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = etc.predict(X_train)

rmse = mean_squared_error(y_train,y_pred,squared=False)
accuracy = r2_score(y_train,y_pred)

print('DecisionTreeClassifier')
print (f'RMSE: ',{rmse})
print(f'Accuracy: ',{accuracy})


# In[26]:


sns.distplot(y_pred, label = 'prediction')
sns.distplot(y_train, label = 'actual')
plt.title('DecisionTreeClassifier')

plt.legend()


# In[27]:


#trying with kNeighbors Regression

knr = KNeighborsRegressor(n_neighbors=5) 
knr.fit(X_train, y_train)

y_pred = knr.predict(X_train)

rmse = mean_squared_error(y_train,y_pred,squared=False)
accuracy = r2_score(y_train,y_pred)


print (f'RMSE: ',{rmse})
print(f'Accuracy: ',{accuracy})


# In[28]:


sns.distplot(y_pred, label = 'prediction')
sns.distplot(y_train, label = 'actual')

plt.title('KNeighborsRegressor')
plt.legend()


# In[29]:


#reading the data with a function to avoid too much repetition
def read_dafaframe(filename):
    df = pd.read_csv(filename)

    encoder = OneHotEncoder(sparse=False)

    encoded_types = encoder.fit_transform(df['type'].values.reshape(-1, 1))

    label= ['red','white']
    wine_types = pd.DataFrame(encoded_types, columns= label)

    df['red_wine'] = wine_types['red']

    return df


# In[30]:


train_df = read_dafaframe('wine_quality_training/wine_data/train_wine_data.csv')
#adding a validation dataset
validation_df = read_dafaframe('wine_quality_training/wine_data/test_wine_data.csv')


# In[31]:


train_df.head()


# In[32]:


validation_df.head()


# In[33]:


validation_df.info()


# In[34]:


#filling in the mean values

for col, value in validation_df.items():
 if col != 'type':
    validation_df[col] = validation_df[col].fillna(validation_df[col].mean())
validation_df.isnull().sum()


# In[35]:


trained_dict =  train_df.drop(columns=['type','quality'])
val_dict =  validation_df.drop(columns=['type','quality'])


# In[36]:


X_train = trained_dict
X_val = val_dict


# In[37]:


target = 'quality'
y_train = train_df[target].values

y_val = validation_df[target].values


# In[39]:


rfr.fit(X_train, y_train)
y_pred_rfr = rfr.predict(X_val)

with open('models/random_forest_reg.bin','wb') as f_out:
    pickle.dump((rfr),f_out)

rmse = mean_squared_error(y_val,y_pred_rfr,squared=False)
accuracy = r2_score(y_val, y_pred_rfr)

print('RandomForestRegressor')
print(f'RMSE: {rmse}')
print(f'Accuracy: {accuracy}')


# In[40]:


rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_val)

with open('models/rfc.bin','wb') as f_out:
    pickle.dump((rfc),f_out)

rmse = mean_squared_error(y_val,y_pred_rfc,squared=False)
accuracy = r2_score(y_val, y_pred_rfc)

print('RandomForestClassifier')
print(f'RMSE: {rmse}')
print(f'Accuracy: {accuracy}')


# In[41]:


etc.fit(X_train, y_train)
y_pred_etc = etc.predict(X_val)

with open('models/etc.bin','wb') as f_out:
    pickle.dump((etc),f_out)

rmse = mean_squared_error(y_val,y_pred_etc,squared=False)
accuracy = r2_score(y_val,y_pred_etc)

print('ExtraTreesClassifier')
print (f'RMSE: ',{rmse})
print(f'Accuracy: ',{accuracy})


# In[42]:


knr = KNeighborsRegressor(n_neighbors=5)
knr.fit(X_train, y_train)
y_pred_knr = knr.predict(X_val)


with open('models/kn_reg.bin','wb') as f_out:
    pickle.dump((knr),f_out)

rmse = mean_squared_error(y_val,y_pred_knr,squared=False)
accuracy = r2_score(y_val, y_pred_knr)

print('KNeighborsRegressor')
print(f'RMSE: {rmse}')
print(f'Accuracy: {accuracy}')


# In[ ]:




