import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Preparar los datos, carga del archivo csv con los datos
data = pd.read_csv('../datasets/DATOS_COVID.csv')
print(data.shape)

# Atributos a excluir elegidos de manera manual, se tomó como criterio sólo síntomas y no enfermedades preexistentes
# Las enfermedades preexistentes serían más útiles al tratar de estudiar la gravedad de la enfermedad
# u otras consecuencias, como la probabilidad de decesos
exclude = [
        'Odinofagia', 
        'Sexo', 
        'Edad_', 
        'Ataque_al_estado_general', 
        'EPOC', 
        'Asma', 
        'Inmunosupresion',
        'Enfermedad_Cardiovascular', 
        'Obesidad', 
        'Insuficiencia_renal_cronica', 
        'Tabaquismo'
    ]

X = data.drop(exclude, axis=1) # Características
y = data['Covid'].values # Variable objetivo


# Dividir los datos en los conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, )

# Crear el pipeline (Estandarización y el clasificador Árbol de decisión)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(criterion='gini', random_state=50, max_depth=20, splitter='best'))
])

# Entrenar el pipeline
pipeline.fit(X_train, y_train)

# Hacer predicciones con el conjunto de pruebas
y_pred = pipeline.predict(X_test)

# Evaluación del modelo
print("Score", pipeline.score(X_test, y_test))
print(classification_report(y_test, y_pred))

# Cálculo de la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Visualizar matriz de confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Valor verdadero')

plt.title('Matriz de confusión - Tiene o no Covid')
plt.show()