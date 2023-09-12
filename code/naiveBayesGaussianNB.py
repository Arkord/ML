import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

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
# print(nb_classifier.score(x_test, y_test))

param_grid = {
    # You can specify different hyperparameters to search here
    # For example, 'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=nb_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the grid search to your data
grid_search.fit(x_train, y_train)  # Replace X and y with your data

# Print the best hyperparameters and corresponding accuracy
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)