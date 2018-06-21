import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
# from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# leitura dos dados de treinamento
dataset = pd.read_csv("../input/train_users_reduzido_01.csv")
dataset = dataset.sort_values(by='country_destination', ascending=False)
rotulos = dataset['country_destination'].copy()
classes = rotulos.unique()


# definição dos atributos
atributos_numericos = ['age']
# atributos_categoricos = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
atributos_categoricos = ['gender']
atributo_classe = ['country_destination']

def pre_processamento_dados(dados, a_numericos, a_categoricos, scaler = None):
    dados_numericos = dados[a_numericos]
    
    imputer = Imputer(strategy="median")
    print(dados_numericos)
    imputer.fit(dados_numericos)
    dados_numericos_completo = imputer.transform(dados_numericos)
    print(dados_numericos_completo)
    # dados_numericos_completo = dados_numericos
    
    if scaler == None:
        # Normalização
        scaler = StandardScaler()
        scaler.fit(dados_numericos_completo)
    
    numericos_dados_normalizados = scaler.transform(dados_numericos_completo)
    
    # atributos categóricos
    dados_categoricos = dados[a_categoricos]
    dados_caregoricos_encoded = pd.get_dummies(dados_categoricos).values
    
    dados_completo = np.concatenate((dados_caregoricos_encoded, numericos_dados_normalizados),1)
    return dados_completo


print('Iniciando pre processamento de dados ... ')
X = pre_processamento_dados(dataset, atributos_numericos, atributos_categoricos)
pd.DataFrame(X).to_csv('dados_completos_transformados.csv')
y = rotulos
print('Split conjunto de treinamento ... ')
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)



print('Iniciando treinamento ... ')
# fit model no training data
model = XGBClassifier()
model.fit(X_treino, y_treino)

print('Iniciando predição ... ')
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