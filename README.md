# MLOps_Project
Final Project for MLOPS learning
---
This project serves as a capstone to the tutorial sessions taken concerning the work flow of Machine Learning Operations. 

In machine learning, once the dataset is acquired, MLOps Engineers typically go through the following steps
- Model Training
- Experiment Tracking
- Orchestration 
- Deployment 
- Monitoring

So let's go through my process at tackling this problem.

### The Dataset
After spending a weekend thinking of what dataset to use, finding my Eureka moment then later seeiing the potential flaws of using the chosen dataset, I decided to work on a simple and very common dataset that I had worked on in the past. I decided to go with a Wine Quality Dataset, where some characteristics of wine are gathered and matched to what professional wine taster deem to rate the selected wine.

Instead of splitting one dataset into train and test, I decided to get two different wine quality datasets to use for each. I wanted the training dataset to be as big as possible to improve the model.

I got both my datasets from kaggle, this is the link;

- Training Set: https://www.kaggle.com/datasets/subhajournal/wine-quality-data-combined

- Validation Set: https://www.kaggle.com/datasets/rajyellow46/wine-quality 


### 1. Model Training

Before we even start the project, if you're starting from scratch then you might want to get Python installed and create a virtual environment for this.

#### - Packages
For the packages and libraies used for this particular section, we would have to ```pip install``` them. Primarily;

- ```pandas``` for data manipulation
- ```matplotlib``` and ```seaborn``` for data visualization
- ```sklearn``` for model training

#### - Inspecting the Data

Following the code in the repository, I loaded the training data and examined it. A few interesting things and be gathered from the output below which was achieved by the following code 
```python 
wine_train_df.isnull().sum()
```
The output:
```
type                    0
fixed_acidity           0
volatile_acidity        0
citric_acid             0
residual_sugar          0
chlorides               0
free_sulfur_dioxide     0
total_sulfur_dioxide    0
density                 0
pH                      0
sulphates               0
alcohol                 0
quality                 0
dtype: int64
```

Frome the out put, there are no null values (a good thing). As much as this data looks almost perfect, there was a problem.


On a closer look at the data, we can see that almost all the columns are numerical except ```type```. Numerical data is crucial when it comes to working with Machine Learning Models.

To fix this problem, I changed the values of the ```type``` column from **categorical** to **numerical** using *One Hot Encoding*


```python
from sklearn.preprocessing import OneHotEncoder

#encoding the wine types as 1 and 0 with respect to red to be computated

encoder = OneHotEncoder(sparse=False)

encode_types = encoder.fit_transform(wine_train_df['type'].values.reshape(-1, 1))
encode_types
```

the code above changes the values in the ```type``` column from white and red to 1 and 0. Now red wine will be represented as 1 and white wine as 0