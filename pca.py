import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from versao1.versao_1_21_06_2018 import pre_processamento_dados
import matplotlib.pyplot as plt

num_features = [
    'age',
]

cat_features = [
    'gender', 'signup_method', 'language', 'affiliate_channel', 
    'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 
    'first_device_type', 'first_browser'
]

class_feature = ['country_destination']

# dataframe train
df_train = pd.read_csv('input/train_users_2.csv')
target = df_train['country_destination']
    
# dataframe test
df_test = pd.read_csv('input/test_users.csv')
id_test = df_test['id']

# preprocessing
X = pre_processamento_dados(df_train, num_features, cat_features)

# generate PCA dimensions
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
principal_df = pd.DataFrame(data=principal_components, columns = ['pc1', 'pc2'])
final_df = pd.concat([principal_df, df_train[['country_destination']]], axis = 1)

# save to csv
final_df.to_csv('PCA/pca-2-dim.csv')
print('saving csv file in PCA/pca-2-dim.csv')

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA to 2 Components', fontsize = 20)

targets = [
    'US', 'FR', 'CA', 'GB', 
    'ES', 'IT', 'PT', 'NL',
    'DE', 'AU', 'NDF', 'other'
]
colors = [
    'r', 'g', 'b', 'k',
    'lime', 'skyblue', 'yellow', 'y',
    'purple', 'gray', 'tomato', 'c'
]
for target, color in zip(targets, colors):
    indicesToKeep = final_df['country_destination'] == target
    ax.scatter(
        final_df.loc[indicesToKeep, 'pc1'],
        final_df.loc[indicesToKeep, 'pc2'],
        c = color,
        s = 5
    )

ax.legend(targets)
ax.grid()
plt.show()