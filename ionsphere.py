#zaczytanie danych
import pandas as pd
ionosphere=pd.read_csv('ionosphere_data.csv',header=None)
ionosphere.columns=['C'+str(i) for i in range(36)]

    #usunięcie kolumny z indeksami
ionosphere.drop("C0", axis=1, inplace=True)

#ionosphere.head()
#ionosphere.shape
#ionosphere.iloc[:,-1].value_counts()


#podział zbioru na część uczącą i testową
from sklearn.model_selection import train_test_split
X, y = ionosphere.iloc[:, :-1], ionosphere.iloc[:, -1]
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=2022,stratify=y)


#wypisać ile to 95% wariancji
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
pca_transform = PCA()
pca_transform.fit(X_train)
variances = pca_transform.explained_variance_ratio_
cumulated_variances=variances.cumsum()

PC_num=(cumulated_variances<0.95).sum()+1
print("Aby wyjasnic 95% wariancji, potrzeba "+str(PC_num)+' składowych głównych')
#wykres wyjaśnianej wariancji
plt.scatter (np.arange(variances.shape[0]),cumulated_variances)
plt.show()

#testowanie zestawów klasyfikacji
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA,FastICA
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler

    #//////////////////P-C-A////////////////////////////////
    # zestaw_1 - (PCA,StandardScaler,KNN)
pipe1=Pipeline([
    ['transformer',PCA(0.95)],
    ['scaler',StandardScaler()],
    ['classifier',kNN(weights='distance')]
])
pipe1.fit(X_train,y_train)
y1_pred=pipe1.predict(X_test)

    # zestaw_2 - (PCA,MinMaxScaler,KNN)
pipe2=Pipeline([
    ['transformer',PCA(0.95)],
    ['scaler',MinMaxScaler()],
    ['classifier',kNN(weights='distance')]
])
pipe2.fit(X_train,y_train)
y2_pred=pipe2.predict(X_test)

    # zestaw_3 - (PCA,RobustScaler,KNN)
pipe3=Pipeline([
    ['transformer',PCA(0.95)],
    ['scaler',RobustScaler()],
    ['classifier',kNN(weights='distance')]
])
pipe3.fit(X_train,y_train)
y3_pred=pipe3.predict(X_test)

    # zestaw_4 - (PCA,brak skalowania,KNN)
pipe4=Pipeline([
    ['transformer',PCA(0.95)],
    ['classifier',kNN(weights='distance')]
])
pipe4.fit(X_train,y_train)
y4_pred=pipe4.predict(X_test)

    # zestaw_5 - (PCA,StandardScaler,SVC)
pipe5=Pipeline([
    ['transformer',PCA(0.95)],
    ['scaler',StandardScaler()],
    ['classifier',SVC()]
])
pipe5.fit(X_train,y_train)
y5_pred=pipe5.predict(X_test)

    # zestaw_6 - (PCA,MinMaxScaler,SVC)
pipe6=Pipeline([
    ['transformer',PCA(0.95)],
    ['scaler',MinMaxScaler()],
    ['classifier',SVC()]
])
pipe6.fit(X_train,y_train)
y6_pred=pipe6.predict(X_test)

    # zestaw_7 - (PCA,RobustScaler,SVC)
pipe7=Pipeline([
    ['transformer',PCA(0.95)],
    ['scaler',RobustScaler()],
    ['classifier',SVC()]
])
pipe7.fit(X_train,y_train)
y7_pred=pipe7.predict(X_test)

    # zestaw_8 - (PCA,brak skalowania,SVC)
pipe8=Pipeline([
    ['transformer',PCA(0.95)],
    ['classifier',SVC()]
])
pipe8.fit(X_train,y_train)
y8_pred=pipe8.predict(X_test)

    # zestaw_9 - (PCA,brak skalowania,DecisionTree)
pipe9=Pipeline([
    ['transformer',PCA(0.95)],
    ['classifier',DT(max_depth=5)]
])
pipe9.fit(X_train,y_train)
y9_pred=pipe9.predict(X_test)

    # zestaw_10 - (PCA,brak skalowania,RandomForest)
pipe10=Pipeline([
    ['transformer',PCA(0.95)],
    ['classifier',RF()]
])
pipe10.fit(X_train,y_train)
y10_pred=pipe10.predict(X_test)

    #//////////////////FastICA////////////////////////////////
    # zestaw_11 - (FastICA,StandardScaler,KNN)
    # zestaw_12 - (FastICA,MinMaxScaler,KNN)
    # zestaw_13 - (FastICA,RobustScaler,KNN)
    # zestaw_14 - (FastICA,brak skalowania,KNN)

    # zestaw_15 - (FastICA,StandardScaler,SVC)
    # zestaw_16 - (FastICA,MinMaxScaler,SVC)
    # zestaw_17 - (FastICA,RobustScaler,SVC)
    # zestaw_18 - (FastICA,brak skalowania,SVC)

    # zestaw_19 - (FastICA,brak skalowania,DecisionTree)
    # zestaw_20 - (FastICA,brak skalowania,RandomForest)

#//////////////////FastICA////////////////////////////////
# zestaw_11 - (FastICA,StandardScaler,KNN)
pipe11=Pipeline([
    ['transformer',FastICA(23,random_state=2022)],
    ['scaler',StandardScaler()],
    ['classifier',kNN(weights='distance')]
])
pipe11.fit(X_train,y_train)
y11_pred=pipe11.predict(X_test)

# zestaw_12 - (FastICA,MinMaxScaler,KNN)
pipe12=Pipeline([
    ['transformer',FastICA(23,random_state=2022)],
    ['scaler',MinMaxScaler()],
    ['classifier',kNN(weights='distance')]
])
pipe12.fit(X_train,y_train)
y12_pred=pipe12.predict(X_test)

# zestaw_13 - (FastICA,RobustScaler,KNN)
pipe13=Pipeline([
    ['transformer',FastICA(23,random_state=2022)],
    ['scaler',RobustScaler()],
    ['classifier',kNN(weights='distance')]
])
pipe13.fit(X_train,y_train)
y13_pred=pipe13.predict(X_test)

# zestaw_14 - (FastICA,brak skalowania,KNN)
pipe14=Pipeline([
    ['transformer',FastICA(23,random_state=2022)],
    ['classifier',kNN(weights='distance')]
])
pipe14.fit(X_train,y_train)
y14_pred=pipe14.predict(X_test)

# zestaw_15 - (FastICA,StandardScaler,SVC)
pipe15=Pipeline([
    ['transformer',FastICA(23,random_state=2022)],
    ['scaler',StandardScaler()],
    ['classifier',SVC()]
])
pipe15.fit(X_train,y_train)
y15_pred=pipe15.predict(X_test)

# zestaw_16 - (FastICA,MinMaxScaler,SVC)
pipe16=Pipeline([
    ['transformer',FastICA(23,random_state=2022)],
    ['scaler',MinMaxScaler()],
    ['classifier',SVC()]
])
pipe16.fit(X_train,y_train)
y16_pred=pipe16.predict(X_test)

# zestaw_17 - (FastICA,RobustScaler,SVC)
pipe17=Pipeline([
    ['transformer',FastICA(23,random_state=2022)],
    ['scaler',RobustScaler()],
    ['classifier',SVC()]
])
pipe17.fit(X_train,y_train)
y17_pred=pipe17.predict(X_test)

# zestaw_18 - (FastICA,brak skalowania,SVC)
pipe18=Pipeline([
    ['transformer',FastICA(23,random_state=2022)],
    ['classifier',SVC()]
])
pipe18.fit(X_train,y_train)
y18_pred=pipe18.predict(X_test)

# zestaw_19 - (FastICA,brak skalowania,DecisionTree)
pipe19=Pipeline([
    ['transformer',FastICA(23,random_state=2022)],
    ['classifier',DT(max_depth=5)]
])
pipe19.fit(X_train,y_train)
y19_pred=pipe19.predict(X_test)

# zestaw_20 - (FastICA,brak skalowania,RandomForest)
pipe20=Pipeline([
    ['transformer',FastICA(23,random_state=2022)],
    ['classifier',RF()]
])
pipe20.fit(X_train,y_train)
y20_pred=pipe20.predict(X_test)

#macierze pomyłek
from sklearn.metrics import confusion_matrix,accuracy_score
print("Macierz pomyłek [1]\n",confusion_matrix(y_test,y1_pred))
print("Macierz pomyłek [2]\n",confusion_matrix(y_test,y2_pred))
print("Macierz pomyłek [3]\n",confusion_matrix(y_test,y3_pred))
print("Macierz pomyłek [4]\n",confusion_matrix(y_test,y4_pred))
print("Macierz pomyłek [5]\n",confusion_matrix(y_test,y5_pred))
print("Macierz pomyłek [6]\n",confusion_matrix(y_test,y6_pred))
print("Macierz pomyłek [7]\n",confusion_matrix(y_test,y7_pred))
print("Macierz pomyłek [8]\n",confusion_matrix(y_test,y8_pred))
print("Macierz pomyłek [9]\n",confusion_matrix(y_test,y9_pred))
print("Macierz pomyłek [10]\n",confusion_matrix(y_test,y10_pred))
print("Macierz pomyłek [11]\n",confusion_matrix(y_test,y11_pred))
print("Macierz pomyłek [12]\n",confusion_matrix(y_test,y12_pred))
print("Macierz pomyłek [13]\n",confusion_matrix(y_test,y13_pred))
print("Macierz pomyłek [14]\n",confusion_matrix(y_test,y14_pred))
print("Macierz pomyłek [15]\n",confusion_matrix(y_test,y15_pred))
print("Macierz pomyłek [16]\n",confusion_matrix(y_test,y16_pred))
print("Macierz pomyłek [17]\n",confusion_matrix(y_test,y17_pred))
print("Macierz pomyłek [18]\n",confusion_matrix(y_test,y18_pred))
print("Macierz pomyłek [19]\n",confusion_matrix(y_test,y19_pred))
print("Macierz pomyłek [20]\n",confusion_matrix(y_test,y20_pred))
#Punkty dokładności
print("Accuracy score (1):",accuracy_score(y_test,y1_pred))
print("Accuracy score (2):",accuracy_score(y_test,y2_pred))
print("Accuracy score (3):",accuracy_score(y_test,y3_pred))
print("Accuracy score (4):",accuracy_score(y_test,y4_pred))
print("Accuracy score (5):",accuracy_score(y_test,y5_pred))
print("Accuracy score (6):",accuracy_score(y_test,y6_pred))
print("Accuracy score (7):",accuracy_score(y_test,y7_pred))
print("Accuracy score (8):",accuracy_score(y_test,y8_pred))
print("Accuracy score (9):",accuracy_score(y_test,y9_pred))
print("Accuracy score (10):",accuracy_score(y_test,y10_pred))
print("Accuracy score (11):",accuracy_score(y_test,y11_pred))
print("Accuracy score (12):",accuracy_score(y_test,y12_pred))
print("Accuracy score (13):",accuracy_score(y_test,y13_pred))
print("Accuracy score (14):",accuracy_score(y_test,y14_pred))
print("Accuracy score (15):",accuracy_score(y_test,y15_pred))
print("Accuracy score (16):",accuracy_score(y_test,y16_pred))
print("Accuracy score (17):",accuracy_score(y_test,y17_pred))
print("Accuracy score (18):",accuracy_score(y_test,y18_pred))
print("Accuracy score (19):",accuracy_score(y_test,y19_pred))
print("Accuracy score (20):",accuracy_score(y_test,y20_pred))


#Najlepsze wyniki klasyfikacji dla zestawów
"""
Accuracy score (6): 0.9577464788732394 # zestaw_6 - (PCA,MinMaxScaler,SVC)
Accuracy score (8): 0.9577464788732394 # zestaw_8 - (PCA,brak skalowania,SVC)
ccuracy score (10): 0.9577464788732394 # zestaw_10 - (PCA,brak skalowania,RandomForest)
ccuracy score (16): 0.9577464788732394 # zestaw_16 - (FastICA,MinMaxScaler,SVC)

"""