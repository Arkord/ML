import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.utils.multiclass import unique_labels

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Prepare the data
# Assume you have a CSV file 'data.csv' with the features in columns and the target variable in the last column.
data = pd.read_csv('../datasets/4 PostCovid v51.csv')

# Convert categorical variables to one-hot encoded representation
# data = pd.get_dummies(data)

# Atributos a excluir
exclude = [
        'gender',
        'age',
        'miedo_generalizado',
        'ideacion_suicida',
        'aislamiento',
        'estres',
        'perdida_memoria',
        'niebla_cerebral',
        'depresion',
        'enrojecimiento_ojos',
        'ansiedad',
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
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(criterion='gini', random_state=50, max_depth=20, splitter='best'))
])

# Step 4: Train the pipeline
pipeline.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = pipeline.predict(X_test)

# # Step 4: Make predictions on the test set
# y_pred = clf.predict(X_test)

# Step 5: Evaluate the model's accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# unique_classes = unique_labels(y_test, y_pred)

# print(unique_classes)

y_true_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("Score", pipeline.score(X_test, y_test))
print(classification_report(y_true_labels, y_pred_labels))

# Step 6: Compute the confusion matrix
labels = ["Anxiety", "Depression", "Stress"]



cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)

# Step 7: Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)

plt.title('Confusion Matrix - Trastornos mentales')
plt.show()

