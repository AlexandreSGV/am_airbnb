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
dataset = pd.read_csv("../input/train_users_2.csv")
dataset = dataset.sort_values(by='country_destination', ascending=False)
rotulos = dataset['country_destination'].copy()
classes = rotulos.unique()
colunas = dataset.columns.values;
# raise Exception()

print('Colunas', colunas)
print(dataset.shape)
print((dataset.age.value_counts()))

age_counts = dataset.age.value_counts()
# print((dataset.age.value_counts() *100)/dataset.age.size)

idades_limpas = dataset.loc[ (dataset['age'] >=15.0) & (dataset['age'] <=95.0)]

print('Idades Limpas shape',idades_limpas.shape)
print(idades_limpas.age.value_counts())


# for i ,j in age_counts.items():
	# print(i,j)
idades_erradas = dataset.loc[ ((dataset['age'] <15.0) | (dataset['age'] >100.0)) & (dataset['age'] <=1900.0)]
print('Idades Erradas shape',idades_erradas.shape)
print(idades_erradas.age.value_counts())


idades_calculaveis = dataset.loc[ (dataset['age'] <2000.0) & (dataset['age'] >1900.0)]
print('Idades Calculaveis shape',idades_calculaveis.shape)
print(idades_calculaveis.age.value_counts())

# print(idades_calculaveis)
for i in range(idades_calculaveis.shape[0]):
	print(idades_calculaveis.iloc[i])
	# idades_calculaveis.iat[i] = 2015.0 - idades_calculaveis.iloc[i]

print(idades_calculaveis.age.value_counts())
