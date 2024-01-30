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

