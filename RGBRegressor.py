# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:54:47 2022

@author: Bun
"""
#%% Loading dataset
import pandas as pd

X = pd.read_csv('C:\\Users\dinos\\Desktop\\Python Project\\Earthquake Dataset\\train_values.csv')
y = pd.read_csv("C:\\Users\\dinos\\Desktop\\Python Project\\Earthquake Dataset\\train_labels.csv")
X_test = pd.read_csv("C:\\Users\\dinos\Desktop\\Python Project\\Earthquake Dataset\\test_values.csv")
submission = pd.read_csv("C:\\Users\\dinos\\Desktop\\Python Project\\Earthquake Dataset\\submission_format.csv")

#%% Data Cleaning
## Check for NA
print(X.isnull().sum())

## Get the target variables
X.drop(['building_id'], axis = 1, inplace = True)
y.drop(['building_id'], axis = 1, inplace = True)
X_test.drop(['building_id'], axis = 1, inplace = True)

#%% train and valid set split
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8,
                                                      test_size = 0.2,
                                                      random_state = 0)

#%% Building Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

model = Pipeline(steps = [
    ('onehot', OneHotEncoder(handle_unknown = 'ignore')),
    ('xgb',XGBRegressor( n_estimators = 1000, learning_rate = 0.05))
    ])

#%% Model fit

model.fit(X_train, y_train)

#%% Model Predict 

preds = model.predict(X_valid)

#%% Evaluation method
from sklearn.metrics import mean_absolute_error

score = mean_absolute_error(y_valid, preds)
print(score)

#%% Predict test data
y_test = model.predict(X_test)

submission['damage_grade'] = y_test.astype(int)

#%% Export output
import pandas as pd

submission.to_csv('submission.csv', index = False)
# Score 0.5406 - ranking 1600