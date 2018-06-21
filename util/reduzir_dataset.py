import pandas as pd
# escolha a proporção de redução
proporcao = 0.4
df_teste_completo = pd.read_csv('../input/test_users.csv')
df_teste_reduzido = df_teste_completo.sample(frac=proporcao)
df_teste_reduzido.to_csv('../input/test_users_reduzido.csv')
df_train_completo = pd.read_csv('../input/train_users_2.csv')
df_train_reduzido = df_train_completo.sample(frac=proporcao)
df_train_reduzido.to_csv('../input/train_users_reduzido.csv')

