import os
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


current_directory = 'C:\\Users\\Atit Acharya\\OneDrive - The University of Texas at Tyler\\Desktop\\Research\\Classification\\XGBOOST'
# print(current_directory)

acc_XGB = []
acc_DMat = []
for filename in os.listdir(current_directory):
    #     print(filename)
    if os.path.isfile(os.path.join(current_directory, filename)) and '.csv' in filename:
        #         # Do something with the file
        print("\n-------------------------------------------------------------------\n")
        data = pd.read_csv(filename)
        train, test = train_test_split(data, test_size=0.1, random_state=42)
        train, val = train_test_split(train, test_size=0.1, random_state=42)

        X_train = train.drop(['LastStatementMinimumPaymentDueAmount', 'CardExternalStatus','Target', 'LastStatementBalanceAmount'] , axis=1)
        # X_train = train.drop('Target', axis=1)
        y_train = train['Target']

        X_val = val.drop(['LastStatementMinimumPaymentDueAmount' , 'CardExternalStatus', 'Target', 'LastStatementBalanceAmount'] , axis=1)
        # X_val = val.drop('Target', axis=1)
        y_val = val['Target']

        X_test = test.drop(['LastStatementMinimumPaymentDueAmount', 'CardExternalStatus', 'Target', 'LastStatementBalanceAmount'] , axis=1)
        # X_test = test.drop('Target', axis=1)
        y_test = test['Target']

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'max_depth': 10,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lamdba': 0.1
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(params, dtrain, evals=[(dval, 'validation')], early_stopping_rounds=10)

        dtest = xgb.DMatrix(X_test)

        y_pred = model.predict(dtest)
        y_pred = np.round(y_pred)

        Dmatrix_accuracy = accuracy_score(y_test, y_pred)

        print('Accuracy using DMatrix: {:.2f}%'.format(Dmatrix_accuracy * 100))

        # initialize the XGBClassifier model with pruning parameters
        clf = XGBClassifier(
            max_depth=3,  # Maximum tree depth
            learning_rate=0.1,  # Learning rate
            #             n_estimators=100,     # Number of trees in the forest
            gamma=0.1,  # Minimum loss reduction to make a further partition
            reg_alpha=0.1,  # L1 regularization on leaf weights
            #             reg_lambda=0.1,       # L2 regularization on leaf weights
            subsample=0.8,  # Subsample ratio of the training set
            colsample_bytree=0.8  # Subsample ratio of columns when constructing each tree
        )

        # train the XGBClassifier model on the training set
        clf.fit(X_train, y_train)

        # evaluate the accuracy of the model on the testing set
        XGBClassifier_accuracy = clf.score(X_test, y_test)
        print("Accuracy using XGBClassifier: %.2f%%" % (XGBClassifier_accuracy * 100.0))

        acc_XGB.append(XGBClassifier_accuracy)
        acc_DMat.append(Dmatrix_accuracy)

Average_DMatrix = np.mean(acc_DMat)
print(Average_DMatrix)

Average_XGBClassifier = np.mean(acc_XGB)
print(Average_XGBClassifier)