import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


acc_XGBDART = []
current_directory = 'C:\\Users\\Atit Acharya\\OneDrive - The University of Texas at Tyler\\Desktop\\Research\\Classification\\XGBOOST'
# print(current_directory)
for filename in os.listdir(current_directory):
    #     print(filename)
    if os.path.isfile(os.path.join(current_directory, filename)) and '.csv' in filename:
       # Do something with the file
            print("\n---------------------------------------------------------------------------------------\n")
            # Load the dataset
            data = pd.read_csv('dataset_1.csv')

            # Split data into features (X) and target (y)
            X = data.drop(['LastStatementMinimumPaymentDueAmount', 'CardExternalStatus', 'Target', 'LastStatementBalanceAmount'], axis=1)
            # X = data.drop('Target', axis=1)
            y = data['Target']

            # Split data into training, validation, and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            # Create an XGBClassifier model with XGBDART
            model = xgb.XGBClassifier(booster='dart', tree_method='hist', max_depth=6, learning_rate=0.1, n_estimators=100, gamma=0.1, subsample=0.5, colsample_bytree=0.5, reg_lambda=1, reg_alpha=0.5, sample_type='weighted', normalize_type='forest', rate_drop=0.1, skip_drop=0.5, random_state=42)

            # Train the model on the training set
            model.fit(X_train, y_train)

            # Predict the classes of the test set
            y_pred = model.predict(X_test)

            # Evaluate the classifier on the validation set
            y_val_pred = model.predict(X_val)
            print('Validation Classification Report:')
            print(classification_report(y_val, y_val_pred))

            # Evaluate the classifier on the test set
            y_test_pred = model.predict(X_test)
            print('Test Classification Report:')
            print(classification_report(y_test, y_test_pred))

            # Evaluate the accuracy of the model
            XGBDARTClassifier_accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy using Dart Booster:", XGBDARTClassifier_accuracy)

            acc_XGBDART.append(XGBDARTClassifier_accuracy)

Average_XGBDART = np.mean(acc_XGBDART)
print(Average_XGBDART)