#0.0.1 Import bibliotek
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.metrics import confusion_matrix,accuracy_score

#0.1.2 Wczytanie zbioru oraz podział na część uczącą i testową
X,y = load_digits(return_X_y=True)

"""
fig, ax = plt.subplots(1,10,figsize=(10,100))
for i in range (10):
    ax[i].imshow(X[i,:].reshape(8,8),cmap=plt.get_cmap('gray'))
    ax[i].axis('off')
fig.tight_layout()
"""

X.shape #wymiary danych (1797 obserwacji wektórów o długości 64)

    #podział na uczący i testowy w proporcji 8:2
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=2022,stratify=y)#stratify odpowiada zachowaniu proporcji w zbiorach


#0.1.3 Analiza składowych głównych (PCA)
pca_transform=PCA()
pca_transform.fit(X_train)
variances = pca_transform.explained_variance_ratio_
cumulated_variances=variances.cumsum()
plt.scatter (np.arange(variances.shape[0]),cumulated_variances)
plt.show()

    #wyznaczenie ilości głównych składowych, żeby pokrywały min 95% wariancji
PC_num = (cumulated_variances<0.95).sum()+1 #przekroczenie 95% wariancji -> 28
print("Aby wyjasnic 95% wariancji, potrzeba "+str(PC_num)+' składowych głównych')

    #analiza PCA pokrywająca 95% wariancji (szybciej i prościej)
pca95=PCA(n_components=0.95)
X_train_pca=pca95.fit_transform(X_train)

pca95.n_components_

X_test_pca=pca95.transform(X_test)


#0.1.4 Zastosowanie PCA w klasyfikacji
    #skalowanie danych (tzw.standaryzacja)
scaler=StandardScaler()
X_train_pca_scaled=scaler.fit_transform(X_train_pca)
X_test_pca_scaled=scaler.transform(X_test_pca)

    #klasyfikacja kNN
model=kNN(n_neighbors=5, weights='distance')
model.fit(X_train_pca_scaled,y_train)
y_predict=model.predict(X_test_pca_scaled)

print("Macierz pomyłek:\n",confusion_matrix(y_test, y_predict))
print("Accuracy score:",accuracy_score(y_test,y_predict))

#0.1.5 Zastosowanie klasy Pipeline
from sklearn.pipeline import Pipeline
pipe=Pipeline([
    ['transformer',PCA(0.95)],
    ['scaler',StandardScaler()],
    ['classifier',kNN(weights='distance')]
])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)

print("Macierz pomyłek:\n",confusion_matrix(y_test,y_pred))
print("Accuracy score:",accuracy_score(y_test,y_pred))

#0.1.6 Analiza składowych niezależnych
pipe=Pipeline([
    ['transformer',FastICA(20,random_state=2022)],
    ['scaler',StandardScaler()],
    ['classifier',kNN(weights='distance')]
])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)

print("Macierz pomyłek:\n",confusion_matrix(y_test,y_pred))
print("Accuracy score:",accuracy_score(y_test,y_pred))

