# importing Libraries
# import pandas as pd
# import numpy as np
# import xgboost as xgb

#importing sklearn library and its methods
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
# from scipy.stats import uniform, randint

# Loading a dataset
df = pd.read_csv("dataset.csv")
# print(df)

# Separating the important features from the original dataset
imp_Cardtypefeatures = df.loc[:, [
                                     'CardType',
                                     'CreditLine',
                                     'LastStatementBalanceAmount',
                                     'LastStatementPurchaseAmount',
                                     'LastStatementPurchaseReturnAmount',
                                     'LastStatementMinimumPaymentDueAmount',
                                     'LastStatementPaymentTotalAmount',
                                     'CardExternalStatus',
                                     'CardInternalStatus',
                                     'ActivityHistory1_12',
                                     'DelinquentDaysCount',
                                     'DelinquentTotalAmount',
                                     'LifetimeDelinquent1CycleCount',
                                     'LifetimeDelinquent2CycleCount',
                                     'LifetimeDelinquent3CycleCount',
                                     'LifetimeDelinquent4CycleCount',
                                     'LifetimeDelinquent5CycleCount',
                                     'LifetimeDelinquent6CycleCount',
                                     'LifetimeDelinquent7CycleCount',
                                     'ChargeOffAmount',
                                     'Target'
                                 ]
                       ]
# print(imp_Cardtypefeatures)

# Adding 12 features mentioning from 1st statement cycle to 12th statement cycle
for i in range(12):
    # print(i)
    save_str = ("ActivityHistory_%2d" % (i + 1)).replace(' ', '')
    imp_Cardtypefeatures[save_str] = ''
# print(imp_Cardtypefeatures)

# Dictionary to assign values related to the ActivityHistory1_12 feature
dict = {
    "0": "10",
    "1": "-150",
    "2": "-160",
    "3": "-170",
    "4": "-180",
    "5": "-190",
    "6": "200",
    "7": "210",
    "A": "20",
    "B": "-220",
    "C": "-230",
    "D": "-240",
    "E": "-250",
    "F": "-260",
    "G": "-270",
    "H": "-280",
    "I": "50",
    "J": "-10",
    "K": "-20",
    "L": "-30",
    "M": "-40",
    "N": "-50",
    "O": "-60",
    "P": "-70",
    "Q": "40",
    "R": "-80",
    "S": "-90",
    "T": "-100",
    "U": "-110",
    "V": "-120",
    "W": "-130",
    "X": "-140",
    "Z": "0",
    "%": "10",
    "#": "20",
    "+": "30",
    "-": "40"
}

# Adding individual values of ActivityHistory1_12 to their respective features
# for index, row in imp_Cardtypefeatures.iterrows():
#     item_len = len(row['ActivityHistory1_12'])
#     item = row['ActivityHistory1_12']
#     for i in range(item_len):
#         target_row_name = 'ActivityHistory_' + str(i+1)
#         imp_Cardtypefeatures.at[index,target_row_name] = dict[item[i]]
#         row[target_row_name] = dict[item[i]]

# imp_Cardtypefeatures

# imp_Cardtypefeatures.to_csv("New_Dataset.csv")

# data = pd.read_csv("New_Dataset.csv")
# data

#dropping multiple columns
# df1 = data.drop(['Unnamed: 0', 'ActivityHistory1_12'], axis=1)
# df1

#dropping multiple columns
# df1 = data.drop(['Unnamed: 0', 'ActivityHistory1_12'], axis=1)
# df1

dict_CardType = {
    "VISA PLATINUM": "1",
    "VISA PREMIER CASH": "2",
    "VISA PREMIER REWARDS": "3",
    "MC PLATINUM": "4",
    "MC REWARDS": "5",
    "MC CASH": "6",
    "VISA PLATINUM CASH": "7",
    "MC CLASSIC": "8",
    "VISA PLATINUM REWARDS": "9",
    "VISA GOLD": "10",
    "VISA BUSINESS": "11"
}

#replacing the categorical variable "CardType"into numerical values using dictionary
# df1["CardType"] = df1.CardType.map(dict_CardType)
# df1["CardType"].unique()



dict_CardExternalStatus = {
    "Active": "1",
    "Authorization Prohibited": "2",
    "Closed": "3",
    "Charged-Off": "4",
    "Revoked": "5",
    "Lost": "6",
    "Bankrupt": "7",
    "Interest Prohibited": "8",
    "Stolen": "9",
    "Frozen": "10"

}

# df1["CardExternalStatus"] = df1.CardExternalStatus.map(dict_CardExternalStatus)
# df1["CardExternalStatus"].unique()



dict_CardInternalStatus = {
    "Active": "1",
    "Credit Balance": "2",
    "Delinquent": "3",
    "Overlimit": "4",
    "Delinquent and Overlimit": "5"
}

# df1["CardInternalStatus"] = df1.CardInternalStatus.map(dict_CardInternalStatus)
# df1["CardInternalStatus"].unique()

#Exporting the dataset with target values = 1
# target_value_1 = df1[df1.loc[:, "Target"] == 1]
# target_value_1

#Exporting the dataset with target values = 0
# target_value_0 = df1[df1.loc[:, "Target"] == 0]
# target_value_0

#Exporting 100 dataset by concatenating "target_value_1" with its fixed 1120 values and random 1120 values from "target_value_0"
# result = []
# for i in range (100):
    # print(i)
    # dataset = target_value_0.sample(1120)

    #merging dataframes
#     frames = [dataset, target_value_1]
#     merged_dataset = pd.concat(frames)
#     result.append(merged_dataset)
#     save_str = ("dataset_%2d.csv" % (i+1)). replace(' ', '')
#     merged_dataset.to_csv(save_str)
# print(len(result))


# Sample 1 to check if the XGBoost works properly
# D = pd.read_csv("dataset_1.csv")
# D.drop(D.columns[0], axis=1)

# inputs = D.loc[:, ~D.columns.isin(['name', 'Target'])]
# inputs = inputs.drop(inputs.columns[0], axis=1)
# targets = D.loc[:, "Target"]

# targets

# Dividing training and testing dataset
# X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3, random_state=42)
# X_test

# train = xgb.DMatrix(X_train, label=y_train)
# test = xgb.DMatrix(X_test, label=y_test)

# params = {
#     "colsample_bytree": uniform(0.7, 0.3),
#     "gamma": uniform(0, 0.5),
#     "learning_rate": uniform(0.03, 0.3), # default 0.1
#     "max_depth": randint(2, 6), # default 3
#     "n_estimators": randint(100, 150), # default 100
#     "subsample": uniform(0.6, 0.4)
# }
# epochs = 10

# model = xgb.train(param, train, epochs)

# predictions = model.predict(test)
# predictions

# accuracy_score(y_test, predictions)

# print(classification_report(y_test, predictions))
