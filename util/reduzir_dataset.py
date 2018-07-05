import pandas as pd
# escolha a proporção de redução
proporcao = 0.5

# teste
# df_teste_completo = pd.read_csv('../input/test_users.csv')
# df_teste_reduzido = df_teste_completo.sample(frac=proporcao)
# df_teste_reduzido.to_csv('../input/test_users_reduzido.csv')

# #train 
# df_train_completo = pd.read_csv('../input/train_users_2.csv')
# df_train_reduzido = df_train_completo.sample(frac=proporcao)
# df_train_reduzido.to_csv('../input/train_users_reduzido.csv')

# sessions
# df_sessions_completo = pd.read_csv('../input/sessions.csv')
# df_sessions_reduzido = df_teste_completo.sample(frac=proporcao)
# df_sessions_reduzido.to_csv('../input/sessions_reduzido.csv')

# reforço
# df_sessions_completo_reforco = pd.read_csv('../input/train_users_completo_reforco_3.csv')
# df_sessions_reduzido_reforco = df_sessions_completo_reforco.sample(frac=proporcao)
# df_sessions_reduzido_reforco.to_csv('../input/train_users_reduzido_reforco_3.csv')

# #users clean
df_users_clean = pd.read_csv('../input/users_clean_train.csv')
df_users_clean_reduzido = df_users_clean.sample(frac=proporcao)
df_users_clean_reduzido.to_csv('../input/users_clean_train_reduzido.csv')