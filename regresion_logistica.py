# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar el dataset
train_data = pd.read_csv('train.csv')


# preprocesamiento de dats
# Imputar valores faltantes en Age 
train_data['Age'] = train_data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
test_data['Age'] = test_data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# Imputar valores faltantes en Embarked en el conjunto de entrenamiento
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Imputar valores faltantes en Fare en el conjunto de prueba
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# 2. Target Encoding para la variable 'Sex'
mean_survival_by_sex = train_data.groupby('Sex')['Survived'].mean()
train_data['Sex'] = train_data['Sex'].map(mean_survival_by_sex)
test_data['Sex'] = test_data['Sex'].map(mean_survival_by_sex)

# 3. Conversión de 'Embarked' en variables dummies (columnas binarias)
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

# 4. Agrupación de Fare en intervalos
train_data['FareBin'] = pd.qcut(train_data['Fare'], 4, labels=False)
test_data['FareBin'] = pd.qcut(test_data['Fare'], 4, labels=False)

# 5. Crear la variable 'FamilySize' combinando SibSp y Parch
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# Seleccionar las características finales
features = ['Pclass', 'Sex', 'Age', 'FareBin', 'FamilySize', 'Embarked_Q', 'Embarked_S']
X_train = train_data[features]
y_train = train_data['Survived']
X_test = test_data[features]

# Escalado de características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dividir los datos en entrenamiento y validación
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión logística
model = LogisticRegression(max_iter=200, solver='liblinear')
model.fit(X_train_split, y_train_split)

# Evaluar el modelo
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
class_report = classification_report(y_val, y_pred)

print("Accuracy:", accuracy)

print("Classification Report:\n", class_report)

# Realizar predicciones en los datos de prueba
test_predictions = model.predict(X_test)

# Guardar predicciones en un archivo CSV
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_predictions})
output.to_csv('clasificacion.csv', index=False)
print("Predicciones guardadas en clasificacion.csv")
