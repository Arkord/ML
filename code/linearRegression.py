import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../datasets/fraud_prediction.csv')

# Define feature & target arrays

feature = df['oldbalanceOrg'].values
target = df['amount'].values

### Plot
# plt.scatter(feature, target)
# plt.xlabel('Old Balance of Account Older')
# plt.ylabel('Amount of Transaction')
# plt.title('Amount vs Old balance')

# plt.show()

# Implement Linear Regression Algorithm
from sklearn import linear_model

linear_reg = linear_model.LinearRegression()

feature = feature.reshape(-1, 1)
target = target.reshape(-1, 1)

linear_reg.fit(feature, target)
x_lim = np.linspace(min(feature), max(feature)).reshape(-1, 1)

### scatter plot
# plt.scatter(feature, target)
# plt.xlabel('Old Balance of account older')
# plt.ylabel('Amount of transaction')
# plt.title('Amount vs old balance')

# # creating prediction line

# plt.plot(x_lim, linear_reg.predict(x_lim), color='red')
# plt.show();

# Splitting the data into training and test sets
from sklearn.model_selection import train_test_split

features = df.drop('isFraud', axis=1)
target = df['isFraud'].values

#print(features)
x_train, x_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.3,
    random_state=42,
    stratify=target
)

linear_reg = linear_model.LinearRegression()
linear_reg.fit(x_train, y_train)

print(linear_reg.score(x_test, y_test))

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline_order = [
    ('scaler', StandardScaler()),
    ('linear_reg', linear_model.LinearRegression())
]

pipeline = Pipeline(pipeline_order)

linear_reg_scaled = pipeline.fit(x_train, y_train)

print('Standarized', linear_reg_scaled.score(x_test, y_test))