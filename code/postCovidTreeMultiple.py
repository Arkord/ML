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
data = pd.read_csv('../datasets/4 PostCovid v52.csv')

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
     { "name": "TREE", "method": pipeline_tree }, 
    #  { "name": "LOGISTIC", "method": pipeline_logistic }, 
    #  { "name": "BAYES", "method": pipeline_bayes }
]

for pipe in pipes:

    print("-------------------> ", pipe["name"])

    #pipe["method"].fit(X_train,y_train)

    # Step 4: Train the pipeline
    pipe["method"].fit(X_train, y_train)

    # Step 5: Make predictions on the test set
    y_pred = pipe["method"].predict(X_test)
    predicted_probabilities = pipe["method"].predict_proba(X_test)

    predicted_labels = np.argmax(predicted_probabilities, axis=1)

    # # Step 4: Make predictions on the test set
    # y_pred = clf.predict(X_test)

    # Step 5: Evaluate the model's accuracy
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)

    # unique_classes = unique_labels(y_test, y_pred)

    # print(unique_classes)

    y_true_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    print("Score", pipe["method"].score(X_test, y_test))
    print(classification_report(y_true_labels, y_pred_labels))

    # Step 6: Compute the confusion matrix
    labels = ["Anxiety", "Depression", "Isolation", "Memory Loss", "None of the above", "Stress"]



    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)

    #Step 7: Visualize the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    plt.title('Confusion Matrix - Trastornos mentales')
    #plt.show()

    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # tick_marks = np.arange(len(labels))
    # plt.xticks(tick_marks, labels)
    # plt.yticks(tick_marks, labels)
    # plt.title('Confusion Matrix - Trastornos mentales')
    # plt.show()

    # Step 6: Compute evaluation metrics
    # metrics = classification_report(y_test, y_pred, output_dict=True)

    # # Step 7: Prepare the data for plotting
    # class_names = list(metrics.keys())[:2]
    # precision = [metrics[class_name]['precision'] for class_name in class_names]
    # recall = [metrics[class_name]['recall'] for class_name in class_names]
    # f1_score = [metrics[class_name]['f1-score'] for class_name in class_names]

    # # Step 8: Plot the metrics
    # x = range(len(class_names))
    # width = 0.2

    # plt.bar(x, precision, width, label='Precision')
    # plt.bar(x, recall, width, label='Recall', bottom=precision)
    # plt.bar(x, f1_score, width, label='F1-score', bottom=[p + r for p, r in zip(precision, recall)])

    # plt.xlabel('Class')
    # plt.ylabel('Score')
    # plt.title('Evaluation Metrics')
    # plt.xticks(x, class_names)
    # plt.legend()
    # plt.show()

    for i in range(len(X_test)):
        print(f"Predicted Class: {y_pred_labels[i]}, Probabilities: {predicted_probabilities[i]}")
        #print(y_pred_labels)

# 6 - Estress
# 5 - Ninguna
# 4 - Pérdida de memoria
# 3 - Aislamiento
# 2 - Depresión
# 1 - Ansidedad
