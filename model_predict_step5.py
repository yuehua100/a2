import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt

import itertools
import re
from io import StringIO
import seaborn as sns
import joblib

folder_path = '/data/zyh/feature'
path_to_feature = os.path.join(folder_path, 'features_exp.csv')

data = pd.read_csv(path_to_feature)

Start = datetime.now()

path_to_model = os.path.join('/data/zyh',"Model","model.sav")

model = joblib.load(path_to_model)

y_predict = model.predict(data)

print(y_predict)

unique_motion = pd.Series(y_predict.tolist()).drop_duplicates().tolist()

label_to_type = {
	1: "ATTM",
	2: "BW",
	3: "CTRW",
	4: "FBM",
	5: "LW",
	6: "SBM",
}

coverted_prediction = [label_to_type[pred] for pred in unique_motion]

series = pd.Series(y_predict)
frequency = series.value_counts()
proportions = frequency / len(series)

print(f"total diffusion type 类型种类: {len(unique_motion)}")
print(f"total diffusion type 类型种类: {unique_motion}")
print(f"total diffusion type 类型种类: {coverted_prediction}")
print(f"each diffusion type freq 频率:\n{frequency}") 
print(f"each diffusion type proportion 比例:\n{proportions}") 

output_model=pd.DataFrame(y_predict)

output_feature_file = os.path.join(folder_path, 'model.csv')
output_model.to_csv(output_feature_file,index=False,header=False)

output_frequency=frequency.reset_index()

output_feature_file = os.path.join(folder_path, 'model_name.csv')
output_frequency.to_csv(output_feature_file,index=False,header=False) 
