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
XHeat = data.drop(exclude, axis=1) # Features for PCA Heat

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

from sklearn.decomposition import PCA

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.decomposition import PCA
 
# Reduce from 4 to 2 features with PCA
pca = PCA(n_components=2)
 
# Fit and transform data
pca_features = pca.fit_transform(X)
 
# Create dataframe
pca_df = pd.DataFrame(
    data=pca_features, 
    columns=['PC1', 'PC2'])
 
# map target names to PCA features   
target_names = {
    0:'Ansiedad',
    1:'Depresión',
    2:'Aislamiento',
    3:'Pérdida de memoria',
    4:'Ninguna',
    5:'Estres',
}
 
pca_df['Secuela'] = y
pca_df['Secuela'] = pca_df['Secuela'].map(target_names)

pca_df.head()
sns.set()
 
sns.lmplot(
    x='PC1', 
    y='PC2', 
    data=pca_df, 
    hue='Secuela', 
    fit_reg=False, 
    legend=True,
    palette={'Ansiedad': '#F72585', 'Depresión': '#7209B7', 'Aislamiento': '#3A0CA3', 'Pérdida de memoria': '#4361EE', 'Ninguna': '#4CC9F0', 'Estres': '#b7094c'}
    )
 

explained_var_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio for PC1:", explained_var_ratio[0])
print("Explained Variance Ratio for PC2:", explained_var_ratio[1])

# plt.title('Gráfico PCA')
# plt.xlim(-4,6)
# plt.ylim(-4,6)
# plt.show()

print(pca.components_)
#data_top = XHeat.head() 

# print(XHeat.columns)
df_comp = pd.DataFrame(pca.components_, columns=XHeat.columns)
plt.figure(figsize=(16, 8))

y_axis_labels = ["PC1", "PC2"]

s = sns.heatmap(df_comp, yticklabels=y_axis_labels, cmap='viridis')
plt.xticks(fontsize=10)

print(df_comp)
plt.show()