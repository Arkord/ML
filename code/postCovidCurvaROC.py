import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.utils.multiclass import unique_labels

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Prepare the data
# Assume you have a CSV file 'data.csv' with the features in columns and the target variable in the last column.
data = pd.read_csv('C:/Users\octavio.mejia/Documents/proyectos/ml/datasets/4 PostCovid v52.csv')

# Convert categorical variables to one-hot encoded representation
# data = pd.get_dummies(data)

# Atributos a excluir
exclude = [
        'genero',
        'edad',
        'enrojecimiento_ojos',
        'transtornos_mentales'
    ]

X = data.drop(exclude, axis=1) # Features
categoricalY = data['transtornos_mentales'] # Target variable

# onehot_encoder = OneHotEncoder(sparse=False)
# y = onehot_encoder.fit_transform(categoricalY.values.reshape(-1, 1))

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(categoricalY)

print(categoricalY)
print(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, )

# Step 2.1: Standardize the numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Create and train the decision tree classifier
# clf = DecisionTreeClassifier(criterion='gini', random_state=50, max_depth=20, splitter='best')
# clf.fit(X_train, y_train)

# Step 3: Create the pipeline
pipeline_tree = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(criterion='gini', random_state=50, max_depth=20, splitter='best'))
])

pipeline_logistic = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier',  linear_model.LogisticRegression())
])

pipeline_bayes = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GaussianNB())
])

pipes = [
    #{ "name": "TREE", "method": pipeline_tree }, 
    { "name": "LOGISTIC", "method": pipeline_logistic }, 
    #{ "name": "BAYES", "method": pipeline_bayes }
]

for pipe in pipes:
    print("-------------------> ", pipe["name"])

    pipe["method"].fit(X_train, y_train)
    y_pred = pipe["method"].predict(X_test)

    y_pred = pipe["method"].predict(X_test)
    predicted_probabilities = pipe["method"].predict_proba(X_test)

    predicted_labels = np.argmax(predicted_probabilities, axis=1)

    y_true_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from itertools import cycle

   # Binarize the labels
    y_train_bin = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5])
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])


    # Get predicted probabilities for each class
    y_scores = pipe["method"].predict_proba(X_test)

    # Compute ROC curve and ROC area under the curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(6):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves for each class
    plt.figure(figsize=(8, 6))
    colors = cycle(['blue', 'red', 'green', 'yellow', 'gray', 'purple'])
    for i, color in zip(range(6), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Rango Falsos Positivos')
    plt.ylabel('Rango Verdaderos Positivos')
    plt.title('Característica Operativa del Receptor (ROC) Curva - Multiclase')
    plt.legend(loc='lower right')
    plt.show()