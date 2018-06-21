from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=",")

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
print(X.shape)
print(Y.shape)

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

print(model)


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
















# import xgboost as xgb
# import numpy as np

# # dtrain = xgb.DMatrix('test_users.csv?format=csv&label_column=15')
# # dtest = xgb.DMatrix('train_users_2.csv?format=csv&label_column=15')

# teste = np.random.rand(10, 10)  # 5 entities, each contains 10 features
# data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
# label = np.random.randint(2, size=5)  # binary target
# dtrain = xgb.DMatrix(data, label=label)
# dtest = xgb.DMatrix(teste)



# param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
# param['nthread'] = 4
# param['eval_metric'] = 'auc'

# evallist = [(dtest, 'eval'), (dtrain, 'train')]

# num_round = 10
# bst = xgb.train(param.items(), dtrain, num_round)

# bst.save_model('0001.model')


# data = np.random.rand(7, 10)
# dtest = xgb.DMatrix(data)
# ypred = bst.predict(dtest)

# xgb.plot_importance(bst)