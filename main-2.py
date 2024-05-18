# Import des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
# Chargement des données
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
df = pd.read_csv(url, header=None)

# Analyse des données
print("Shape:", df.shape)
print("Types des données:", df.dtypes)
print("Aperçu des données:", df.head())
print("Résumé statistique:\n", df.describe())
print("Distribution des classes:", df[60].value_counts())

# Visualisation des données
# À vous de compléter cette partie en utilisant des histogrammes, plots, et visualisations de corrélations
# Visualisation des histogrammes pour chaque attribut
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Visualisation des corrélations entre les attributs
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation entre les attributs")
plt.show()

# Division des données en features et target
X = df.drop(60, axis=1)
y = df[60]

# Création du jeu de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Définition des modèles
models = [
    ("Logistic Regression", LogisticRegression()),
    ("K Nearest Neighbors", KNeighborsClassifier()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Support Vector Machine", SVC())
]

# Évaluation des modèles
for name, model in models:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=10)
    print(f"{name}: Accuracy moyenne = {
          np.mean(scores)}, Écart-type = {np.std(scores)}")

# Sélection du meilleur modèle
best_model = SVC()

# Entraînement du meilleur modèle
best_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", best_model)
])
best_pipeline.fit(X_train, y_train)

# Prédictions sur le jeu de test
y_pred = best_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy sur le jeu de test:", accuracy)
