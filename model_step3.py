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

characteristics_data = pd.read_csv('features_zyh.csv')
target_data = pd.read_csv('target_zyh.csv')
X = characteristics_data.iloc[:,:-1]
m = characteristics_data['Mode']
y = target_data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42, stratify=m)

print(X_test.shape)
print(len(y_train))
print(len(y_test))

Start = datetime.now()
project_directory = os.path.dirname(os.getcwd())
path_to_model = os.path.join(project_directory,"Model")



model = RandomForestClassifier(n_estimators=720, criterion='entropy', bootstrap=True, max_depth=30, max_features='sqrt', min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=20)


model.fit(X_train, y_train)

joblib.dump(model, os.path.join(path_to_model, 'model.sav'))

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

y_predict = model.predict(X_test)

print(len(y_predict))

print(accuracy_score(y_test,y_predict))


cm = confusion_matrix(y_test,y_predict)

print(cm)

plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,cmap='Blues',fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('Truth labels')
plt.title('Confusion Matrix')
plt.show()

cm1 = np.random.randint(0,3000,(20,20))
cm1 = cm1 + cm1.T
np.fill_diagonal(cm1,range(1,20))

print(cm1)

plt.figure(figsize=(15,12))
sns.heatmap(cm1,annot=True,cmap='crest')
plt.xlabel('Predicted labels')
plt.ylabel('Truth labels')
plt.title('Confusion Matrix')
plt.show()


importances = model.feature_importances_
column_names = characteristics_data.drop(['Mode'], axis=1).columns.values
indices = np.argsort(importances)[::-1]


plt.figure(figsize=(10,7))
plt.title('Feature Importances')
plt.bar(range(len(importances)),importances[indices],color="b",align="center")
plt.xticks(range(len(importances)),[column_names[i] for i in indices],rotation=90)
plt.xlim([-1,len(importances)])
plt.show()

    
End = datetime.now()
ExecutedTime = End - Start
print(ExecutedTime)

