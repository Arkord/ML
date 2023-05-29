import pandas as  pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../datasets/fraud_prediction.csv')
df = df.drop([ 'Unnamed: 0'], axis = 1)

# Creating features
features = df.drop('isFraud', axis=1).values
target = df['isFraud'].values

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)

from sklearn.naive_bayes import GaussianNB

nb_classifier = GaussianNB()

nb_classifier.fit(x_train, y_train)

# Extract the score accuracy
print(nb_classifier.score(x_test, y_test))