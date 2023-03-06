# importing Libraries
import pandas as pd
import xgboost as xgb
import shap

#importing sklearn library and its methods

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score

model = xgb.XGBClassifier(
    max_depth=6, # default 3,
    n_estimators=100, # default 100,
    subsample=0.6,
    learning_rate=0.03, # default 0.1 ,
    min_child_weight=1,
    colsample_bytree= 0.3
)

datasets = [ 'dataset_1.csv', 'dataset_10.csv', 'dataset_100.csv', 'dataset_11.csv']

for dataset_file in datasets:
    df = pd.read_csv(dataset_file)
    numSamples, numFeatures = df.shape
    D = df.drop(['Unnamed: 0', 'ChargeOffAmount', 'LastStatementPurchaseReturnAmount'], axis=1)

    X = D.loc[:, ~D.columns.isin(['name', 'Target'])]
    y = D.loc[:, "Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
F1_score = f1_score(y_test, y_pred)

#     accuracy = model.score(X_test, y_test)
print(f"Accuracy on {dataset_file}: {accuracy}")
print(f"Accuracy on {dataset_file}: {precision}")
print(f"Accuracy on {dataset_file}: {F1_score}")


# Perforing SHAP operations and plots

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
shap.summary_plot(shap_values, X_test)