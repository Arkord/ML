import pandas as pd
from sklearn.model_selection import train_test_split

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

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini', random_state=50)
dt.fit(x_train, y_train)

print(dt.score(x_test, y_test))

from sklearn.model_selection import GridSearchCV

grid_params = {
    'max_depth': [1, 2, 3, 4, 5, 6],
    'min_samples_leaf': [ 0.02, 0.04, 0.06, 0.08]
}

grid_object = GridSearchCV(estimator=dt, param_grid= grid_params, scoring='accuracy', cv=10, n_jobs=-1)
grid_object.fit(x_train, y_train)

print('Best parameters for decision Tree:', grid_object.best_params_)

from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import tree

dt.fit(features, target)

feature_names = df.drop('isFraud', axis = 1)

data = tree.export_graphviz(dt, out_file=None, feature_names=feature_names.columns.values, proportion=True)

graph = pydotplus.graph_from_dot_data(data)

graph.write_pdf("tree.pdf")