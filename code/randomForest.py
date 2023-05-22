import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../datasets/fraud_prediction.csv')

df = df.drop(
    ['Unnamed: 0'], axis=1
)

features = df.drop('isFraud', axis=1).values
target = df['isFraud'].values

x_train, x_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.3,
    random_state=42,
    stratify=target
)

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(random_state=50)
rf_classifier.fit(x_train, y_train)
rf_classifier.score(x_test, y_test)

from sklearn.model_selection import GridSearchCV

grid_params = {
    'n_estimators': [100, 200, 300, 400, 5000],
    'max_depth': [1, 2, 4, 6, 8],
    'min_samples_leaf': [0.05, 0.1, 0.2]
}

grid_object = GridSearchCV(estimator=rf_classifier, param_grid=grid_params, scoring='accuracy', cv=3, n_jobs=-1)
grid_object.fit(x_train, y_train)

grid_object.best_params_
rf_best = grid_object.best_estimator_
