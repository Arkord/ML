import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler


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

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini', random_state=50, max_depth=10, min_samples_leaf=0.01)
dt.fit(x_train, y_train)

print(dt.score(x_test, y_test))

y_pred = dt.predict(x_test)

#print(y_pred)


std_slc = StandardScaler()
pca = decomposition.PCA()
dtreeReg = tree.DecisionTreeRegressor()

pipe = Pipeline(steps=[("std_slc", std_slc),
                           ("pca", pca),
                           ("dtreeReg", dtreeReg)])

n_components = list(range(1,x_train.shape[1]+1,1))

criterion = ["friedman_mse", "mse"]
max_depth = [4,6,8,10]

parameters = dict(pca__n_components=n_components,
                      dtreeReg__criterion=criterion,
                      dtreeReg__max_depth=max_depth)

clf = GridSearchCV(pipe, parameters)
clf.fit(x_train, y_train)

print("Best Number Of Components:", clf.best_estimator_.get_params()["pca__n_components"])
print(); print(clf.best_estimator_.get_params()["dtreeReg"])

CV_Result = cross_val_score(clf, x_train, y_train, cv=3, n_jobs=-1, scoring="r2")
print(); print(CV_Result)
print(); print(CV_Result.mean())
print(); print(CV_Result.std())


