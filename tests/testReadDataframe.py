import unittest
import pandas as pd
import sys
sys.path.append('../')
from src.utils.functions import read_dataframe


test_path = '/mnt/c/Users/Dev/desktop/mlops_project/tests/wine_data/test_wine_data.csv'
train_path = '/mnt/c/Users/Dev/desktop/mlops_project/tests/wine_data/train_wine_data.csv'

class TestReadDataFrame(unittest.TestCase):

    def setUp(self):
        print('test started')

    def test_read_csv(self):
        df = read_dataframe(train_path)
        self.assertEqual(df.shape, (32485, 13))

        expected_cols = ['fixed_acidity', 'volatile_acidity', 
                        'citric_acid', 'residual_sugar', 
                        'chlorides', 'free_sulfur_dioxide',  
                        'total_sulfur_dioxide', 'density', 'pH', 
                        'sulphates', 'alcohol', 'quality'] 
        
        self.assertCountEqual(df[expected_cols].columns.tolist(), expected_cols)

    def test_label_encoding(self):
        df = read_dataframe(train_path)
        actual_df = pd.read_csv(train_path)
        # Count instances of each type 
        white_count = (actual_df['type'] == 'white').sum()  
        red_count = (actual_df['type'] == 'red').sum()
        # Check encoded columns match counts
        encoded_white = (df['type'] == 1).sum() 
        encoded_red = (df['type'] == 0).sum()
        
        # Compare counts between original and encoded
        self.assertEqual(white_count, encoded_white)
        self.assertEqual(red_count, encoded_red)

    
    def test_missing_values(self):
         df = read_dataframe(train_path)  
         # Check missing values
         self.assertFalse(df.isnull().any().any())
            


    def tearDown(self):
        print('test ended')

if __name__ == '__main__':
    unittest.main()