import pathlib
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from prefect import flow, task
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from prefect.artifacts import create_markdown_artifact
from datetime import date




#function for reading the data
@task(retries=4)
def read_dataframe(filename):
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

#function to split the data
@task
def transform(train_df, validation_df):
    trained_dict =  train_df.drop(columns=['type','quality'])
    val_dict =  validation_df.drop(columns=['type','quality'])


    X_train = trained_dict
    X_val = val_dict


    target = 'quality'
    y_train = train_df[target].values

    y_val = validation_df[target].values

    return X_train, y_train, X_val, y_val
#function to train the data
@task(log_prints=True)
def training(X_train, y_train, X_val, y_val):

    with mlflow.start_run():

        mlflow.set_tag("developer","Sven")

        mlflow.log_param("train-data-path", "../wine_data/train_wine_data.csv")
        mlflow.log_param("valid-data-path", "../wine_data/test_wine_data.csv")

        n_estimators = 100
        mlflow.log_param('n_estimators',n_estimators)

        etc = ExtraTreesClassifier(n_estimators=n_estimators)

        etc.fit(X_train, y_train)
        y_pred_etc = etc.predict(X_val)

        rmse = mean_squared_error(y_val,y_pred_etc,squared=False)
        accuracy = r2_score(y_val,y_pred_etc)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/etc.bin", "wb") as f_out:
            pickle.dump(etc, f_out)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tag("model", "ExtraTreesClassifier")

        mlflow.log_artifact("models/etc.bin",artifact_path="sklearn_model")
        
        mlflow.sklearn.log_model(etc,"model")

        markdown__metric_report = f"""# Metric Report

        ## Summary

        Wine Quality Prediction 

        ## RMSE ExtraTreesClassifier Model

        | Region    | RMSE | Accuracy|
        |:----------|-------:|-------:|
        | {date.today()} | {rmse:.2f} | {accuracy:.2f} |
"""
        
        create_markdown_artifact(
        key="wine-quality-model-report", markdown=markdown__metric_report
        )

    return None

#main flow
@flow
def main_flow(
    train_path: str = './wine_data/train_wine_data.csv',
    val_path: str = './wine_data/test_wine_data.csv',
) -> None:
    """The main training pipeline"""

    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("wine-quality-experiment")

    # Load
    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)

    # Transform
    X_train,y_train,X_val,y_val = transform(df_train, df_val)

    # Train
    training(X_train,y_train,X_val,y_val)

if __name__ == "__main__":
    main_flow()