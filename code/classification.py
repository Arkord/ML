import numpy as np
import pandas as pd

import sklearn

from sklearn import datasets
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import k_means
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.size'] = 12

iris = datasets.load_iris()

#print(iris)

data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target = pd.DataFrame(data=iris.target, columns=['species'])
data = pd.concat([data, target], axis=1)
data = data.sample(frac=1, random_state=1234)

#print(data.head());

#Fracción de entrenamiento 80%
Ntrain = int(data.shape[0] * 0.8)

#Datos de entrenamiento
train = data.iloc[:Ntrain, :]

#Datos de prueba
test = data.iloc[Ntrain, :]

plength = data["petal length (cm)"]
pwidth = data["petal width (cm)"]

#grafica para comparar Petal length (eje x) contra Petal width (eje y)
f = plt.figure(figsize=(5, 5))
ax = f.add_subplot(1,1,1)
ax.scatter(plength, pwidth)
ax.set_xlabel("Petal length")
ax.set_ylabel("Petal width")
ax.set_title("Petal with vs length")
f.tight_layout()

plt.show()