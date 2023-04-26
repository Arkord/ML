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

data.head();