import pandas as  pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

df = pd.read_csv('../datasets/fraud_prediction.csv')
df = df.drop([ 'Unnamed: 0'], axis = 1)

# Creating features
features = df.drop('isFraud', axis=1).values
target = df['isFraud'].values

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)

from sklearn.svm import LinearSVC

svm = LinearSVC(random_state=50, dual=False)
svm.fit(x_train, y_train)

print(svm.score(x_test, y_test))


### Graphical method to check best hyperparameter
# training_scores = []
# testing_scores = []

# param_list = [
#     0.0001,
#     0.001,
#     0.01,
#     0.1,
#     10,
#     100,
#     1000
# ]

# for param in param_list:
#     svm = LinearSVC(C=param, random_state=42)
#     svm.fit(x_train, y_train)

#     training_scores.append(svm.score(x_train, y_train))
#     testing_scores.append(svm.score(x_test, y_test))

# plt.semilogx(
#     param_list,
#     training_scores,
#     param_list,
#     testing_scores
# )

# plt.legend('train', 'test')
# plt.ylabel('Accuracy scores')
# plt.xlabel('C (Inverse regularizartion strength)')

# #plt.show()

from sklearn.model_selection import GridSearchCV
svm = LinearSVC(random_state=50, dual=False)

grid = GridSearchCV(svm, {
    'C': [
        0.00001,
        0.0001,
        0.001,
        0.001,
        0.01,
        0.1,
        10
    ]
})

grid.fit(x_train, y_train)

print("The best value of the inverse regularization strength is: ", grid.best_params_)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#Setting up the scaling pipeline
order = [('scaler', StandardScaler()), ('SVM', LinearSVC(C = 0.1,
random_state = 50, dual=False))]
pipeline = Pipeline(order)
#Fitting the classfier to the scaled dataset
svm_scaled = pipeline.fit(x_train, y_train)
#Extracting the score

print('scaled score', svm_scaled.score(x_test, y_test))