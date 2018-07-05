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
classes_minoritarias = {'FR': 1, 'IT':1, 'GB':1, 'ES':1, 'CA':1, 'DE':1, 'NL':1, 'AU':1, 'PT':1}
# classes_minoritarias['bla bla'] = 1
print (classes_minoritarias)
# raise Exception()

print(dataset.country_destination.size)
print((dataset.country_destination.value_counts() *100)/dataset.country_destination.size)



# print(dataset.loc[dataset['country_destination'] == 'PT'])

for pais, fator in classes_minoritarias.items():
    print(pais , fator)
    incremento = dataset.loc[dataset['country_destination'] == pais]
    dataset = dataset.append([incremento]*3)

print((dataset.country_destination.value_counts() *100)/dataset.country_destination.size)
dataset.to_csv('../input/train_users_completo_reforco_3.csv')