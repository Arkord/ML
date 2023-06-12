import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Prepare the data
# Assume you have a CSV file 'data.csv' with the features in columns and the target variable in the last column.
data = pd.read_csv('../datasets/4 PostCovid v3.csv')

# Convert categorical variables to one-hot encoded representation
data = pd.get_dummies(data)

X = data.drop('ansiedad', axis=1)  # Features
y = data['ansiedad']                # Target variable

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2.1: Standardize the numeric features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

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

print("Score", pipeline.score(X_test, y_test))

# Step 6: Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Step 7: Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Ansiedad')
plt.show()

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

