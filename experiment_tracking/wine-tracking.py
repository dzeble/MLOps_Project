#!/usr/bin/env python
# coding: utf-8

# In[2]:


#imports for data exploration and analysis
import pandas as pd
#importing models
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import OneHotEncoder

import pickle


# In[3]:


import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wine-quality-experiment")


# In[4]:


from sklearn.preprocessing import LabelEncoder

def read_dafaframe(filename):
    df = pd.read_csv(filename)

    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])

    encoder = OneHotEncoder()
    encoded_types = encoder.fit_transform(df['type'].values.reshape(-1, 1))

    label = le.classes_.tolist()
    wine_types = pd.DataFrame(encoded_types.toarray(), columns=label)

    df = pd.concat([df, wine_types], axis=1)

    for col, value in df.items():
        if col != 'type':
            df[col] = df[col].fillna(df[col].mean())

    return df


# In[5]:


train_df = read_dafaframe('wine_quality_training/wine_data/train_wine_data.csv')
validation_df = read_dafaframe('wine_quality_training/wine_data/test_wine_data.csv')


trained_dict =  train_df.drop(columns=['type','quality'])
val_dict =  validation_df.drop(columns=['type','quality'])


X_train = trained_dict
X_val = val_dict


target = 'quality'
y_train = train_df[target].values

y_val = validation_df[target].values


# In[6]:


rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_val)

rmse = mean_squared_error(y_val,y_pred_rfc,squared=False)
accuracy = r2_score(y_val, y_pred_rfc)

print('RandomForestClassifier')
print(f'RMSE: {rmse}')
print(f'Accuracy: {accuracy}')


# In[7]:


etc = ExtraTreesClassifier(n_estimators=100)

etc.fit(X_train, y_train)
y_pred_etc = etc.predict(X_val)

rmse = mean_squared_error(y_val,y_pred_etc,squared=False)
accuracy = r2_score(y_val,y_pred_etc)

print('ExtraTreesClassifier')
print (f'RMSE: ',{rmse})
print(f'Accuracy: ',{accuracy})


# In[8]:


with open('models/etc.bin','wb') as f_out:
    pickle.dump((etc),f_out)


# In[40]:


with mlflow.start_run():

    mlflow.set_tag("developer","Sven")

    mlflow.log_param("train-data-path", "wine_quality_training/wine_data/train_wine_data.csv")
    mlflow.log_param("valid-data-path", "wine_quality_training/wine_data/test_wine_data.csv")

    n_estimators = 100
    mlflow.log_param('n_estimators',n_estimators)

    etc = ExtraTreesClassifier(n_estimators=n_estimators)

    etc.fit(X_train, y_train)
    y_pred_etc = etc.predict(X_val)

    rmse = mean_squared_error(y_val,y_pred_etc,squared=False)
    accuracy = r2_score(y_val,y_pred_etc)
    
    print('ExtraTreesClassifier')
    print (f'RMSE: ',{rmse})
    print(f'Accuracy: ',{accuracy})

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.set_tag("model", "ExtraTreesClassifier")

    mlflow.log_artifact("models/etc.bin",artifact_path="sklearn_model")
    
    mlflow.sklearn.log_model(etc,"model")


# In[10]:


import xgboost as xgb
from hyperopt import  fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope


# In[11]:


train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)


# In[12]:


def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        accuracy = r2_score(y_val,y_pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("accuracy", accuracy)

    return {'loss': rmse, 'accuracy':accuracy, 'status': STATUS_OK}


# In[13]:


search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}

best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
)


# In[25]:


params={
    'learning_rate':	0.19810073258268465,
    'max_depth':	8,
    'min_child_weight':	3.6166554203005914,
    'objective':	'reg:linear',
    'reg_alpha':	0.1778602103708601,
    'reg_lambda':	0.026355351557057448,
    'seed':	42
}

mlflow.xgboost.autolog(disable=True)

booster = xgb.train(
    params=params,
    dtrain=train,
    num_boost_round=1000,
    evals=[(valid, 'validation')],
    early_stopping_rounds=50
)



# In[16]:


with mlflow.start_run():

    mlflow.set_tag("developer","Sven")

    mlflow.log_param("train-data-path", "wine_quality_training/wine_data/train_wine_data.csv")
    mlflow.log_param("valid-data-path", "wine_quality_training/wine_data/train_wine_data.csv")

    n_estimators = 100
    mlflow.log_param('n_estimators',n_estimators)

    etc = ExtraTreesClassifier(n_estimators=n_estimators)

    etc.fit(X_train, y_train)
    y_pred_etc = etc.predict(X_val)

    rmse = mean_squared_error(y_val,y_pred_etc,squared=False)
    accuracy = r2_score(y_val,y_pred_etc)
    
    print('ExtraTreesClassifier')
    print (f'RMSE: ',{rmse})
    print(f'Accuracy: ',{accuracy})

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("accuracy", accuracy)
    
    mlflow.log_artifact(local_path="models/etc.bin", artifact_path="models_pickle")


# In[20]:


booster.save_model('models/xgb.bin')


# In[21]:


params={
    'learning_rate':	0.19810073258268465,
    'max_depth':	8,
    'min_child_weight':	3.6166554203005914,
    'objective':	'reg:linear',
    'reg_alpha':	0.1778602103708601,
    'reg_lambda':	0.026355351557057448,
    'seed':	42
}


booster = xgb.train(
    params=params,
    dtrain=train,
    num_boost_round=1000,
    evals=[(valid, 'validation')],
    early_stopping_rounds=50
)

y_pred = booster.predict(valid)
rmse = mean_squared_error(y_val, y_pred, squared=False)
accuracy = r2_score(y_val,y_pred)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("accuracy", accuracy)
mlflow.set_tag("model", "xgboost")
mlflow.log_params(params)

mlflow.log_artifact(local_path="models/xgb.bin", artifact_path="models_pickle")


# In[26]:


if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run():
    
    best_params={
    'learning_rate':	0.19810073258268465,
    'max_depth':	8,
    'min_child_weight':	3.6166554203005914,
    'objective':	'reg:linear',
    'reg_alpha':	0.1778602103708601,
    'reg_lambda':	0.026355351557057448,
    'seed':	42
}

    mlflow.log_params(best_params)

    booster = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=1000,
        evals=[(valid, 'validation')],
        early_stopping_rounds=50
    )

    y_pred = booster.predict(valid)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    accuracy = r2_score(y_val,y_pred)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.set_tag("model", "xgboost")


    with open("models/xgb.bin","wb") as f_out:
        pickle.dump(booster,f_out)
    
    mlflow.log_artifact("models/xgb.bin",artifact_path="xgboost_model")

    mlflow.xgboost.log_model(booster,artifact_path="models_mlflow")


# In[27]:


# logged_model = 'runs:/972efc8246b347e8beb77b09ad439e4d/models_mlflow'

# # Load model as a PyFuncModel.
# loaded_model = mlflow.pyfunc.load_model(logged_model)


# # In[28]:


# loaded_model


# # In[29]:


# xgboost_model = mlflow.xgboost.load_model(logged_model)


# # In[30]:


# xgboost_model


# # In[31]:


# new_y_pred = xgboost_model.predict(valid)


# # In[32]:


# new_y_pred[:10]


# # In[41]:


with mlflow.start_run():

    mlflow.set_tag("developer","Sven")

    mlflow.log_param("train-data-path", "wine_quality_training/wine_data/train_wine_data.csv")
    mlflow.log_param("valid-data-path", "wine_quality_training/wine_data/train_wine_data.csv")

    n_estimators = 100
    mlflow.log_param('n_estimators',n_estimators)

    rfc = RandomForestClassifier(n_estimators=100)

    rfc.fit(X_train, y_train)
    y_pred_rfc = rfc.predict(X_val)

    rmse = mean_squared_error(y_val,y_pred_rfc,squared=False)
    accuracy = r2_score(y_val,y_pred_rfc)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.set_tag("model", "RandomForestClassifier")

    mlflow.log_artifact("models/rfc.bin",artifact_path="sklearn_model")

  
    
    mlflow.sklearn.log_model(rfc,"model")


# In[35]:


with open('models/rfc.bin','wb') as f_out:
    pickle.dump((rfc),f_out)


# In[ ]:




