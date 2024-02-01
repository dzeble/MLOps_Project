import unittest
import pandas as pd
import sys
import mlflow
sys.path.append('../')
from src.utils.functions import transform, read_dataframe, training


test_path = '/mnt/c/Users/Dev/desktop/mlops_project/tests/wine_data/test_wine_data.csv'
train_path = '/mnt/c/Users/Dev/desktop/mlops_project/tests/wine_data/train_wine_data.csv'


class TestTrain(unittest.TestCase):

    def setUp(self):
        print('test started')

    def test_training(self):
        train_df = read_dataframe(train_path)
        validation_df = read_dataframe(test_path)
        X_train,y_train,X_val,y_val = transform(train_df, validation_df)
        training(X_train,y_train,X_val,y_val)
        last_run = mlflow.get_run(mlflow.last_active_run().info.run_id)
        metrics = last_run.data.metrics

        self.assertGreater(metrics['accuracy'], 0.6)
        self.assertLess(metrics['rmse'], 1.2) 

    def tearDown(self):
        print('test ended')

if __name__ == '__main__':
    unittest.main()