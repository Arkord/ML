import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../datasets/4 PostCovid v3.csv')

df.gender = pd.factorize(df.gender)[0]
df.age = pd.factorize(df.age)[0]

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
linear_reg = linear_model.LinearRegression()
linear_reg.fit(x_train, y_train)

print(linear_reg.score(x_test, y_test))
