import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../datasets/fraud_prediction.csv')

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

# Implement Logistic Regression Algorithm
from sklearn import linear_model

logistic_regression = linear_model.LogisticRegression(C=10, penalty='l1', solver='liblinear')
logistic_regression.fit(x_train, y_train)

print(logistic_regression.score(x_test, y_test))
#print(y_train)
#print(y_test)

# Fine tuning hyperparameters
# from sklearn.model_selection import GridSearchCV
# logistic_regression = linear_model.LogisticRegression(penalty='l1', solver='liblinear')

# grid = GridSearchCV(
#     logistic_regression,
#     {
#         'C': [
#             0.0001,
#             0.001,
#             0.01,
#             0.1,
#             10
#         ]
#     }
# )
# grid.fit(x_train, y_train)

# print("the most optimal inverse regularization strenght is:", grid.best_params_)

### Check GridSearchCV Plot
train_errors = []
test_errors = []

C_list = [
    0.0001,
    0.001,
    0.01,
    0.1,
    10,
    100,
    1000
]

for value in C_list:
    logistic_regression = linear_model.LogisticRegression(C=value, penalty='l1', solver='liblinear')
    logistic_regression.fit(x_train, y_train)

    train_errors.append(logistic_regression.score(x_train, y_train))
    test_errors.append(logistic_regression.score(x_test, y_test))
plt.semilogx(C_list, train_errors, C_list, test_errors)
plt.legend(
    ("train"),
    ("test")
)
plt.ylabel('Accuracy Score')
plt.xlabel('C (Inverse regularization strength)')
#plt.show()

# Standarize & Scale
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline_order = [
    ('scaler', StandardScaler()),
    ('logistic_reg', linear_model.LogisticRegression(C=10, penalty='l1', solver='liblinear'))
]

pipeline = Pipeline(pipeline_order)
logistic_regression_scaled = pipeline.fit(x_train, y_train)

print('Scaled score', logistic_regression_scaled.score(x_test, y_test))

# printing out the coefficients of each variable

print(logistic_regression.coef_)

# printing the intercept of the model

print(logistic_regression.intercept_)