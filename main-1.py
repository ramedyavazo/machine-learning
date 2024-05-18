# Étape 1: Charger les données

# Importer les librairies nécessaires
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Charger le jeu des données iris dans pandas
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=np.c_[
                       iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Résumer le jeu des données
print("Dimensions du jeu des données : ", iris_df.shape)
print("\nAperçu des données : \n", iris_df.head())
print("\nRésumé statistique de toutes les caractéristiques : \n", iris_df.describe())
print("\nRépartition des données par rapport à la variable de classe : \n",
      iris_df['target'].value_counts())

# Visualiser les données en utilisant les histogrammes et les « plots »
iris_df.hist()
plt.show()

# Étape 2: Evaluer certains algorithmes et estimer leurs précisions sur des données non vues

# Créer un jeu de test
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Mettre en place le dispositif de test pour utiliser une validation croisée 10 folds

# Construire 4 modèles différents pour prédire l’espère à partir des mesures des fleurs

models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

# Évaluer les modèles
results = []
names = []
for name, model in models:
    cv_results = cross_val_score(
        model, X_train, y_train, cv=10, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

# Sélectionner le meilleur modèle
best_model = models[np.argmax([result.mean() for result in results])]
print("\nMeilleur modèle : ", best_model[0])

# Étape 3: Faire des prédictions

# Entraîner le meilleur modèle sur les données d'entraînement
best_model[1].fit(X_train, y_train)

# Faire des prédictions sur les données du jeu de test
predictions = best_model[1].predict(X_test)

# Evaluer la précision du meilleur modèle sur les données du jeu de test
print("\nPrécision du meilleur modèle sur les données du jeu de test : ",
      accuracy_score(y_test, predictions))
