import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../datasets/4 PostCovid v2.csv')

df.gender = pd.factorize(df.gender)[0]
df.age = pd.factorize(df.age)[0]
df.country = pd.factorize(df.country)[0]

# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LinearRegression

features = df.drop(
    [
        'transtornos_mentales',
        'miedo_generalizado',
        'ideacion_suicida',
        'aislamiento',
        'estres',
        'perdida_memoria',
        'niebla_cerebral',
        'depresion',
        'ansiedad'
    ]
    , axis = 1).values
target = df['ansiedad'].values

#print(df)

x_train, x_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.3,
    random_state=42,
    stratify=target
)

from sklearn.naive_bayes import GaussianNB

nb_classifier = GaussianNB()

nb_classifier.fit(x_train, y_train)

# Extract the score accuracy
print(nb_classifier.score(x_test, y_test))

# from sklearn.tree import DecisionTreeClassifier

# dt = DecisionTreeClassifier(criterion='gini', random_state=50, max_depth=10, min_samples_leaf=0.01)
# dt.fit(x_train, y_train)

# print(dt.score(x_test, y_test))

# from sklearn.model_selection import GridSearchCV

# grid_params = {
#     'max_depth': [1, 2, 3, 4, 5, 6],
#     'min_samples_leaf': [0.01, 0.02, 0.04, 0.06, 0.08]
# }

# grid_object = GridSearchCV(estimator=dt, param_grid= grid_params, scoring='accuracy', cv=10, n_jobs=-1)
# grid_object.fit(x_train, y_train)

# print('Best parameters for decision Tree:', grid_object.best_params_)

# from six import StringIO
# from IPython.display import Image
# from sklearn.tree import export_graphviz
# import pydotplus
# from sklearn import tree

# dt.fit(features, target)

# feature_names = df.drop(
#      [
#         'transtornos_mentales',
#         'miedo_generalizado',
#         'ideacion_suicida',
#         'aislamiento',
#         'estres',
#         'perdida_memoria',
#         'niebla_cerebral',
#         'depresion',
#         'ansiedad'
#     ]
#     , axis = 1)

# data = tree.export_graphviz(dt, out_file=None, feature_names=feature_names.columns.values, proportion=True)

# graph = pydotplus.graph_from_dot_data(data)

# graph.write_pdf("postCovidTree-1.pdf")