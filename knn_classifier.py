# Importing essential libraries 

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Reading the dataset and importing to the project

data = pd.read_csv("cancer.csv")


# Specifying feature variables from dataset
# We assigned the most important features which cuase cancer as variables for the model 

features = data[["radius_mean", "radius_mean", "area_mean", "area_worst", "symmetry_mean"]]
target = data.diagnosis


# Reshaping the features and converting the data to numpy arrays as we have 5 feature to diagonse the cancer

x = np.array(features).reshape(-1, 5)
y = np.array(target)


# Preprocessing the x, y for better model performance and to recieve higher accuracy

x = preprocessing.MinMaxScaler().fit_transform(x)


# Splitting the dataset into test set and train set so the model can train on dataset and can we can test the model with the unseen data to see the result 

feature_train, feature_test, target_train, target_test = train_test_split(x, y, test_size=0.25)


# _________________________ Building KNN model _________________________

model = KNeighborsClassifier(n_neighbors=5)
model.fit(feature_train, target_train)
result = model.predict(feature_test)


# printing score of the model to evaluate our model
print(accuracy_score(target_test, result))
