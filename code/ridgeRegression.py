import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

df = pd.read_csv('../datasets/fraud_prediction.csv')

features = df.drop('isFraud', axis = 1).values
target = df['isFraud'].values

x_train, x_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.3,
    random_state=42,
    stratify=target
)

ridge_reg = Ridge(alpha=0)
ridge_reg.fit(x_train, y_train)

print(ridge_reg.score(x_test, y_test))

ridge_regression = Ridge()

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    ridge_regression,
    {
        'alpha': [
            0.0001,
            0.001,
            0.01,
            0.1,
            10
        ]
    }
)
grid.fit(x_train, y_train)

print("The most optimal value of alpha is :", grid.best_params_)

ridge_regression = Ridge(alpha=0.01)
ridge_regression.fit(x_train, y_train)

print('Score with optimal alpha = 0.01.', ridge_regression.score(x_test, y_test))