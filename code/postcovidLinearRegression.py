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

from sklearn import linear_model

# dt = DecisionTreeClassifier(criterion='gini', random_state=50, max_depth=10, min_samples_leaf=0.01)
# dt.fit(x_train, y_train)

linear_reg = linear_model.LinearRegression()
linear_reg.fit(x_train, y_train)

#print(linear_reg.score(x_test, y_test))
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline_order = [
    ('scaler', StandardScaler()),
    ('linear_reg', linear_model.LinearRegression())
]

pipeline = Pipeline(pipeline_order)

linear_reg_scaled = pipeline.fit(x_train, y_train)

print('Standarized', linear_reg_scaled.score(x_test, y_test))

