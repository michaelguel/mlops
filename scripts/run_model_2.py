import pandas as pd
from joblib import load
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import os

script_dir = os.path.dirname(os.path.abspath(__file__))

os.chdir(script_dir)

print(os.getcwd())

model = load('./models/model_v2.pkl')

data = pd.read_csv("./sampregdata.csv")

print(pd.DataFrame(data[['x4','x3']]))

predictions = model.predict(pd.DataFrame(data[['x4','x3']]))

print(predictions)

mse = mean_squared_error(data['y'], predictions)

mae = mean_absolute_error(data['y'], predictions)

print("MSE",mse)

print("MAE", mae)