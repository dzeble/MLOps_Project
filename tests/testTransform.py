import unittest
import pandas as pd
import sys
sys.path.append('../')
from src.utils.functions import transform, read_dataframe


test_path = '/mnt/c/Users/Dev/desktop/mlops_project/tests/wine_data/test_wine_data.csv'
train_path = '/mnt/c/Users/Dev/desktop/mlops_project/tests/wine_data/train_wine_data.csv'


class TestTransform(unittest.TestCase):

    def setUp(self):
        print('test started')

    def test_transform(self):
        train_df = read_dataframe(train_path)
        validation_df = read_dataframe(test_path)
        X_train,y_train,X_val,y_val = transform(train_df, validation_df)
        #compare number of columns
        self.assertEqual(len(X_train.columns),len(X_val.columns))
        self.assertEqual(y_train.ndim,y_val.ndim)
        #assert that train is more than test
        self.assertGreater(len(X_train),len(X_val))
        self.assertGreater(len(y_train),len(y_val))

    def tearDown(self):
        print('test ended')

if __name__ == '__main__':
    unittest.main()