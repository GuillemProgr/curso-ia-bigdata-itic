import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.feature_selection import SelectKBest

noms_features = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]

#Carrego el csv.
apple = "apple_quality.csv"

X = np.genfromtxt(apple, delimiter=",", skip_header=1, usecols=(1,2,3,4,5,6,7)) # Carregem les caracteristiques que guardem en X
y = np.genfromtxt(apple, delimiter=",", skip_header=1, usecols=(8,), dtype = str) # Carregem les solucions 


#Elimino l'ultima fila de X i de y, que es brossa del csv i no es una poma
X = X[:-1]
y = y[:-1]


#Divideixo en train i test en aquest cas me quedo en un 30% de dades per fer "d'examen"
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)


#Entreno el model amb desicion tree 
arbre = DecisionTreeClassifier(max_depth=8,random_state=42)
arbre.fit(X_train,y_train)

#Mirem la puntacio del model amb el score
print(f"7 caract., train: {arbre.score(X_train, y_train)}")
print(f"7 caract., test:  {arbre.score(X_test, y_test)}")

# Ens quedem amb les millors caracteristiques per predir si la poma es bona o no 

select_features = SelectKBest(k=4)
select_features.fit(X_train, y_train)

X_train_4 = select_features.transform(X_train)     # train només amb les 4 columnes triades
X_test_4 = select_features.transform(X_test)  

noms_4 = list(np.array(noms_features)[select_features.get_support()])
print("Millors caracteristiques:", noms_4)

#Tornem a entenar pero amb les 4 features millors
arbre_4 = DecisionTreeClassifier(max_depth=8,random_state=42)
arbre_4.fit(X_train_4,y_train)

# Mirem la puntació del model amb el score pero en aquest cas amb les 4 millors caracteristiques
print(f"4 caract., train: {arbre_4.score(X_train_4, y_train)}")
print(f"4 caract., test:  {arbre_4.score(X_test_4, y_test)}")