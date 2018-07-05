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
dataset = pd.read_csv("../input/train_users_reduzido.csv")
dataset = dataset.sort_values(by='country_destination', ascending=False)
rotulos = dataset['country_destination'].copy()
classes = rotulos.unique()


# definição dos atributos
atributos_numericos = ['age']
atributos_categoricos = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
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

def pre_processamento_dados(dados, a_categoricos):
    dados_1 = dados[a_categoricos]
    dados_2 = pre_processamento_data(dados['date_account_created'])
    dados_categoricos = np.concatenate((dados_1, dados_2), 1)
    dados_categoricos = pd.DataFrame(dados_categoricos)


    dados_categoricos_encoded = pd.get_dummies(dados_categoricos)
    print(dados_1.shape)
    print(dados_2.shape)
    print(dados_categoricos.shape)
    print(dados_categoricos_encoded.shape)
    return dados_categoricos_encoded


print(pre_processamento_dados(dataset, atributos_categoricos))

