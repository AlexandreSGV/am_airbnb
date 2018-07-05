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
dataset = dataset.sort_values(by='country_destination', ascending=False)
rotulos = dataset['country_destination'].copy()
classes = rotulos.unique()


# definição dos atributos
atributos_numericos = ['age']
atributos_categoricos = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
# atributos_categoricos = ['gender']
atributo_classe = ['country_destination']


def pre_processamento_data(datas):
    # converte o mês da date_account_created para um atributo categórico indicando a estação
    def escolhe_estacao(mes):
        if (mes <= 5) and (mes >= 3):
            return 'SPRING'
        elif (mes <= 8) and (mes >= 6):
            return 'SUMMER'
        elif (mes <= 11) and (mes >= 9):
            return 'FALL'
        else:
            return 'WINTER'
    estacoes = pd.DataFrame(columns=['estacoes'])
    for i in range(len(datas)):
        mes_data = datas[i][5:7]
        # print(escolhe_estacao(int(mes_data)))
        estacoes.loc[i] = [ escolhe_estacao(int(mes_data)) ]
    return estacoes

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
    # datas = dados['date_account_created']
    dados_estacao = pre_processamento_data(dados['date_account_created'])
    
    dados_categoricos = np.concatenate((dados[a_categoricos], dados_estacao), 1)
    
    dados_categoricos = pd.DataFrame(dados_categoricos)
    dados_caregoricos_encoded = pd.get_dummies(dados_categoricos).values
    
    dados_completo = np.concatenate((dados_caregoricos_encoded, numericos_dados_normalizados),1)
    print('Shape completos',dados_completo.shape)
    return dados_completo


print('Iniciando pre processamento de dados ... ')
# X = pre_processamento_dados(dataset, atributos_numericos, atributos_categoricos)

X = dataset
print(X)


# pd.DataFrame(X).to_csv('dados_completos_transformados.csv')
y = rotulos
print('Split conjunto de treinamento ... ')
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)



print('Iniciando treinamento ... ')
# fit model no training data
clf = svm.SVC()
clf.fit(X_treino, y_treino)  

# model = XGBClassifier(max_depth=5, eta=0.3, n_estimators = 50 )
# model.fit(X_treino, y_treino)

print('Iniciando predição ... ')
# y_pred = model.predict(X_teste)
y_pred = clf.predict(X_teste)
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