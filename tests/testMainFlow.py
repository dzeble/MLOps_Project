import unittest
import pandas as pd
import sys
import mlflow
import json
import warnings 
warnings.filterwarnings('ignore')
sys.path.append('../')
from src.utils.functions import main_flow


test_path = '/mnt/c/Users/Dev/desktop/mlops_project/tests/wine_data/test_wine_data.csv'
train_path = '/mnt/c/Users/Dev/desktop/mlops_project/tests/wine_data/train_wine_data.csv'


class TestMainFlow(unittest.TestCase):

    def setUp(self):
        print('test started')

    def test_mainflow(self):
        training_path = train_path
        val_path = test_path

        main_flow(training_path,val_path)

        runs = mlflow.search_runs()

        # Convert DataFrame to list of Run objects
        run_objs = runs[["run_id", "artifact_uri"]].to_dict(orient="records")  

        # Access run by index 
        run = mlflow.get_run(run_objs[1]["run_id"])

        history = json.loads(run.data.tags['mlflow.log-model.history'])
  
        self.assertIn('Sven', run.data.tags['developer']) 
        self.assertIn('ExtraTreesClassifier', run.data.tags['model']) 
        self.assertIn('model', run.data.tags['mlflow.log-model.history'])
        self.assertIn('model', history[0]['artifact_path'])


    def test_invalid_filepath(self):
        with self.assertRaises(FileNotFoundError):
            main_flow(train_path='invalidpath',val_path='scoobydoo')


    def tearDown(self):
        print('test ended')

if __name__ == '__main__':
    unittest.main()