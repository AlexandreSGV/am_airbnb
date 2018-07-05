import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
# from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# leitura dos dados de treinamento
# dataset = pd.read_csv("../input/train_users_reduzido.csv")
dataset = pd.read_csv("../input/users_clean_train_reduzido.csv")
# dataset = dataset.sort_values(by='country_destination', ascending=False)
rotulos = dataset['country_destination'].copy()
classes = rotulos.unique()

dataset = dataset.drop(['id', 'country_destination'],axis=1)







X = dataset
print(X)


# pd.DataFrame(X).to_csv('dados_completos_transformados.csv')
y = rotulos
print('Split conjunto de treinamento ... ')
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)


print('Iniciando treinamento ... ')
model = XGBClassifier(max_depth=[3,4,5], eta=[0.1, 0.3], n_estimators = [25,50] )
model.fit(X_treino, y_treino)

# fit model no training data
# clf = svm.SVC()
# clf.fit(X_treino, y_treino)  

# model = XGBClassifier(max_depth=5, eta=0.3, n_estimators = 50 )
# model.fit(X_treino, y_treino)

print('Iniciando predição ... ')
# y_pred = model.predict(X_teste)
y_pred = model.predict(X_teste)
print(y_pred)


rotulos = y_teste.values

acertos = 0

for i in range(len(y_pred)):
    if (y_pred[i] == rotulos[i]):
        acertos +=1
print('acurácia >> ', acertos/len(y_pred))

print(classes)
conf_matrix = confusion_matrix(rotulos, y_pred,labels=classes)
print(conf_matrix)