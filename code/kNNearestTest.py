import pandas as pd
import numpy as np

df = pd.read_csv('../datasets/fraud_prediction.csv')

features = df.drop('isFraud', axis=1).values
target = df['isFraud'].values

#print(target)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.3,
    random_state=42,
    stratify=target
)

# Implementation and evaluation of the model

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors=4)
knn_classifier.fit(x_train, y_train)
print(knn_classifier.score(x_test, y_test))

# ##Check best parameters Neighbors

# from sklearn.model_selection import GridSearchCV

# grid = { 'n_neighbors': np.arange(1, 25) }
# knn_classifier = KNeighborsClassifier()
# knn = GridSearchCV(knn_classifier, grid, cv=10)
# knn.fit(x_train, y_train)

# Extracting the optimal number of neighbors
# print(knn.best_params_)
# print(knn.best_score_)

###

### Standarization
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline_order = [
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=1))
]

pipeline = Pipeline(pipeline_order)
knn_classifier_scaled = pipeline.fit(x_train, y_train)

# Extracting the score
print('Standarized scale, score')
print(knn_classifier_scaled.score(x_test, y_test))