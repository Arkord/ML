import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import warnings

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

lasso_reg = Lasso(alpha=0)
lasso_reg.fit(x_train, y_train)

warnings.filterwarnings('ignore')

print(lasso_reg.score(x_test, y_test))

from sklearn.model_selection import GridSearchCV

lasso_regression = Lasso();

grid = GridSearchCV(
    lasso_regression,
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

print("The most optimal value of alpha is:", grid.best_params_)

lasso_regression = Lasso(alpha=0.0001)
lasso_regression.fit(x_train, y_train)

print('Lasso regression with the best value of alpha', lasso_regression.score(x_test, y_test))