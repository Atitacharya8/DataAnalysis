import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os

Tot_corr1 = []
Tot_corr2 = []
current_directory = 'C:\\Users\\Atit Acharya\\OneDrive - The University of Texas at Tyler\\Desktop\\Research\\Classification\\XGBOOST'
# print(current_directory)
for filename in os.listdir(current_directory):
    #     print(filename)
    if os.path.isfile(os.path.join(current_directory, filename)) and '.csv' in filename:
        # Do something with the file
        print("\n-----------------------------------------------------------------\n")
        # Load the dataset
        data = pd.read_csv(filename)
        data = data.drop(['Unnamed: 0'], axis=1)

        # calculate Pearson correlation coefficient
        corr1, _ = pearsonr(data['CardExternalStatus'], pd.Categorical(data['Target']).codes)
        print('Pearson correlation coefficient:', corr1)

        # calculate Spearman rank correlation coefficient
        corr2, _ = spearmanr(data['CardExternalStatus'], pd.Categorical(data['Target']).codes)
        print('Spearman rank correlation coefficient:', corr2)

        Tot_corr1.append(corr1)
        Tot_corr2.append(corr2)

Average_Pearson = np.mean(Tot_corr1)
print("Average Pearson correlation: %.2f" % Average_Pearson)

Average_Spearman = np.mean(Tot_corr2)
print("Average Spearman correlation: %.2f" % Average_Spearman)