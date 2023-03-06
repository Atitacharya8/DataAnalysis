import pandas as pd
import numpy as np

# importing csv files
df = pd.read_csv("dataset_1.csv")
# print(df)

# Separating the important features from the original dataset
df1 = df.loc[:, [
                             'CardType',
                             'CreditLine',
                             'LastStatementBalanceAmount',
                             'LastStatementPurchaseAmount',
                             'LastStatementPurchaseReturnAmount',
                             'LastStatementMinimumPaymentDueAmount',
                             'LastStatementPaymentTotalAmount',
                             # 'CardExternalStatus',
                             # 'CardInternalStatus',
                             # 'ActivityHistory1_12',
#                              'DelinquentDaysCount',
#                              'DelinquentTotalAmount',
#                              'LifetimeDelinquent1CycleCount',
#                              'LifetimeDelinquent2CycleCount',
#                              'LifetimeDelinquent3CycleCount',
#                              'LifetimeDelinquent4CycleCount',
#                              'LifetimeDelinquent5CycleCount',
#                              'LifetimeDelinquent6CycleCount',
#                              'LifetimeDelinquent7CycleCount',
#                              'ChargeOffAmount',
                              'Target'
]
                            ]

# print(df1)

#Dictionary to assign number to cardtype feature
# dict_CardExternalStatus ={
#      "Active":"1",
#      "Authorization Prohibited":"2",
#      "Closed":"3",
#      "Charged-Off":"4",
#      "Revoked":"5",
#      "Lost":"6",
#      "Bankrupt":"7",
#      "Interest Prohibited":"8",
#      "Stolen":"9",
#      "Frozen":"10"
# }

#replacing the categorical variable "CardType"into numerical values using dictionary
# df1["CardExternalStatus"] = df1.CardExternalStatus.map(dict_CardExternalStatus)
# print(df1["CardExternalStatus"].unique())

# Dictionary to assign number to cardtype feature
dict_CardType = {
    "VISA PLATINUM": "1",
     "VISA PREMIER CASH":"2",
     "VISA PREMIER REWARDS":"3",
     "MC PLATINUM": "4",
     "MC REWARDS":"5",
     "MC CASH":"6",
     "VISA PLATINUM CASH":"7",
     "MC CLASSIC":"8",
     "VISA PLATINUM REWARDS":"9",
     "VISA GOLD":"10",
     "VISA BUSINESS":"11"
}

#replacing the categorical variable "CardType"into numerical values using dictionary
df1["CardType"] = df1.CardType.map(dict_CardType)
# print(df1["CardType"].unique())


#Exporting the dataset with target values = 1
target_value_1 = df1[df1.loc[:, "Target"] == 1]
# print(target_value_1)

#Exporting the dataset with target values = 0
target_value_0 = df1[df1.loc[:, "Target"] == 0]
# print(target_value_0)

#Exporting 100 dataset by concatenating "target_value_1" with its fixed 1120 values and random 1120 values from "target_value_0"
result = []
for i in range (100):
    # print(i)
    dataset = target_value_0.sample(1120)

    #merging dataframes
    frames = [dataset, target_value_1]
    merged_dataset = pd.concat(frames)
    merged_dataset.sort_values(by="Target", ignore_index=True)
    # merged_dataset["Target"]
    result.append(merged_dataset)
    save_str = ("dataset_%2d.csv" % (i+1)). replace(' ', '')
    merged_dataset.to_csv(save_str)

print(df1.columns)
value_count = df1["LastStatementPaymentTotalAmount"].value_counts()
print(value_count)


