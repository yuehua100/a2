import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

folder_path = os.getcwd()
feature_file_path = os.path.join(folder_path, 'feature', 'features_zyh.csv')
target_file_path = os.path.join(folder_path, 'feature', 'target_zyh.csv')

characteristics_data = pd.read_csv(feature_file_path)
target_data = pd.read_csv(target_file_path)
X = characteristics_data.iloc[:,:-1]
m = characteristics_data['Mode']
y = target_data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42, stratify=m)

Start = datetime.now()
#project_directory = os.path.dirname(os.getcwd())
project_directory = os.getcwd()
path_to_hyperparameters = os.path.join(project_directory,"Model")

if not os.path.exists(path_to_hyperparameters):
    os.makedirs(path_to_hyperparameters)

random_params = {'n_estimators': [int(x) for x in range(200, 401, 10)],
                 'criterion': ["gini", "entropy"],
                 'max_features': ["log2", "sqrt", "None"],
                 'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
                 'min_samples_split': [2, 5, 6, 8],
                 'min_samples_leaf': [1, 2, 4],
                 'bootstrap': [True, False]
                }

model = RandomForestClassifier()

box_random = RandomizedSearchCV(model, param_distributions=random_params, n_iter=200, cv=10,
                                verbose=2, random_state=43, n_jobs=30)

box_random.fit(X_train, y_train)

print(box_random.best_params_)
print(box_random.best_score_)

with open(os.path.join(path_to_hyperparameters, "hyperparameters.json"), 'a') as fp:
    json.dump(box_random.best_params_, fp)
    
End = datetime.now()
ExecutedTime = End - Start
df = pd.DataFrame({'ExecutedTime': [ExecutedTime]})
df.to_csv(os.path.join(path_to_hyperparameters, "time_for_searching.csv"),mode='a')
print(ExecutedTime)

