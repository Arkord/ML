import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../datasets/fraud_prediction.csv')
df = df.drop(['Unnamed: 0'], axis=1)

features = df.drop('isFraud', axis=1).values
target = df['isFraud'].values

x_train, x_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.3,
    random_state=42,
    stratify=target
)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=1, random_state=42)

ada_boost = AdaBoostClassifier(estimator=tree, n_estimators=100)
ada_boost.fit(x_train, y_train)

print(ada_boost.score(x_test, y_test))

from sklearn.model_selection import GridSearchCV

grid_params = {
    'n_estimators': [100, 200, 300]
}

grid_object = GridSearchCV(estimator=ada_boost, param_grid=grid_params, scoring='accuracy', cv=3, n_jobs=-1)

grid_object.fit(x_train, y_train)

print(grid_object.best_params_)
ada_best = grid_object.best_estimator_

print(ada_best)

