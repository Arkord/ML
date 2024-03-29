import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.utils.multiclass import unique_labels

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Prepare the data
# Assume you have a CSV file 'data.csv' with the features in columns and the target variable in the last column.
data = pd.read_csv('C:/Users/octavio.mejia/Documents/proyectos/ml/datasets/4 PostCovid v54.csv')

# Convert categorical variables to one-hot encoded representation
# data = pd.get_dummies(data)

# Atributos a excluir
exclude = [
        'genero',
        'edad',
        'enrojecimiento_ojos',
        'transtornos_mentales',
        'enfermedad_transmitible',
        'enfermedad_no_transmitible',	
        'enfermedad_organos_cuerpo',	
        'enfermedad_respiratoria',
        'goteo_nasal',
        'vomito',
        'palpitaciones',
        'erupcion_cutanea',
        'malestar_postesfuerzo',
        'tinnitus',
        'dolor_ardor_nervioso',
        'dolor_agudo_costillas',
        'dolor_garganta',
        'estornudos',
        # 'ideacion_suicida',
        # 'aislamiento',
        # 'estres',
        # 'perdida_memoria',
        # 'niebla_cerebral',	
        # 'depresion',
        # 'ansiedad',
        # 'enfermedad_transmitible',
        # 'enfermedad_no_transmitible',	
        # 'enfermedad_organos_cuerpo',	
        # 'enfermedad_respiratoria',
        # 'enrojecimiento_ojos'

    ]

X_raw = data.drop(exclude, axis=1) # Features
XHeat = data.drop(exclude, axis=1) # Features for PCA Heat

categoricalY = data['transtornos_mentales'] # Target variable

# onehot_encoder = OneHotEncoder(sparse=False)
# y = onehot_encoder.fit_transform(categoricalY.values.reshape(-1, 1))

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(categoricalY)

print(categoricalY)
print(y)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, )

# Step 2.1: Standardize the numeric features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Step 3: Create and train the decision tree classifier
# clf = DecisionTreeClassifier(criterion='gini', random_state=50, max_depth=20, splitter='best')
# clf.fit(X_train, y_train)

# Step 3: Create the pipeline
pipeline_tree = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(criterion='gini', random_state=50, max_depth=20, splitter='best'))
])

pipeline_logistic = Pipeline([
    ('classifier',  linear_model.LogisticRegression())
])

pipeline_bayes = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GaussianNB())
])

pipeline_adaboost = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', AdaBoostClassifier(n_estimators=50, random_state=42))
])

pipeline_randomforest = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipes = [
    { "name": "TREE", "method": pipeline_tree }, 
    { "name": "LOGISTIC", "method": pipeline_logistic }, 
    { "name": "BAYES", "method": pipeline_bayes },
    { "name": "ADABOOST", "method": pipeline_adaboost }, 
    { "name": "RANDOM FOREST", "method": pipeline_randomforest }
]

for pipe in pipes:
    print("-------------------> ", pipe["name"])

    pipe["method"].fit(X_train, y_train)

    y_pred = pipe["method"].predict(X_test)
    predicted_probabilities = pipe["method"].predict_proba(X_test)

    y_true_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    print("Score", pipe["method"].score(X_test, y_test))

    print(classification_report(y_true_labels, y_pred_labels))

