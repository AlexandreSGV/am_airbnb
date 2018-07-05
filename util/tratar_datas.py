import pandas as pd



df = pd.read_csv('../input/train_users_reduzido.csv', decimal='.')

datas = df['date_account_created']
# print(datas)



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
    estacoes = pd.DataFrame(['estacoes'])
    for i in range(len(datas)):
        mes_data = datas[i][5:7]
        # print(escolhe_estacao(int(mes_data)))
        estacoes.loc[i] = [ escolhe_estacao(int(mes_data)) ]
    return estacoes


print(pre_processamento_data( datas))