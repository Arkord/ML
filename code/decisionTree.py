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