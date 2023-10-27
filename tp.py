import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.datasets import load_boston

# Charger la base de données Boston Housing 

boston = pd.read_csv('./BostonHousing.csv');
data = pd.DataFrame(boston)
data['MEDV'] = boston.medv
# Ajouter la variable cible MEDV

# Résumé statistique des variables
summary = data.describe()
print("Résumé statistique des variables")
print(summary)

# Matrice de corrélation pour étudier les relations entre les variables
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.show()

# Diagrammes de dispersion pour visualiser les relations entre MEDV et d'autres variables
plt.figure(figsize=(16, 8))
sns.pairplot(data, x_vars=data.columns[:-1], y_vars=['MEDV'], kind='scatter')
plt.title('Diagrammes de dispersion pour visualiser les relations entre MEDV et d\'autres variables')
plt.show()

# Tracer des histogrammes pour chaque variable
plt.figure(figsize=(16, 10))
for i, column in enumerate(data.columns):
    plt.subplot(3, 5, i + 1)
    sns.histplot(data[column], kde=True)
    plt.title(f'Histogramme de {column}')
plt.tight_layout()
plt.show()

# Tracer des boîtes à moustaches (box plots) pour chaque variable
plt.figure(figsize=(16, 8))
sns.boxplot(data=data, orient="v", palette="Set2")
plt.title("Box plots des variables")
plt.xticks(rotation=45)
plt.show() 
