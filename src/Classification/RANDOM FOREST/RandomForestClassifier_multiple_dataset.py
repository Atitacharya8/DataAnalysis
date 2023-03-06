import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

acc_RANDOM = []
current_directory = 'C:\\Users\\Atit Acharya\\OneDrive - The University of Texas at Tyler\\Desktop\\Research\\Classification\\XGBOOST'
# print(current_directory)
for filename in os.listdir(current_directory):
    #     print(filename)
    if os.path.isfile(os.path.join(current_directory, filename)) and '.csv' in filename:
        #         # Do something with the file
        print("\n-------------------------------------------------------------------\n")
        data = pd.read_csv(filename)


        # print('\n----------------------------------------------------------------------------\n')
        # # Listing the name of the features in the dataset
        # data = data.values
        #
        # print('\n----------------------------------------------------------------------------\n')

        # Split data into features (X) and target (y)
        X = data.drop(['LastStatementMinimumPaymentDueAmount', 'CardExternalStatus', 'Target'] , axis=1)
        # X = data.drop('Target', axis=1)
        # X = data[['CreditLine']]
        y = data['Target']

        # Split data into training, validation, and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        # Train random forest classifier
        rfc = RandomForestClassifier(random_state=42)

        # Define grid search parameters
        # param_grid = {
        #                  'n_estimators': [100, 200, 300],
        #                  'max_depth': [5, 10, 20],
        #                  'min_samples_split': [2, 5, 10],
        #                  'min_samples_leaf': [1, 2, 4],
        #                     }
        param_grid = {
            'n_estimators': [300],
            'max_depth': [20],
            'min_samples_split': [10],
            'min_samples_leaf': [4],
        }

        # Perform grid search on the training set
        grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Get best hyperparameters
        best_params = grid_search.best_params_

        # Train the classifier on the entire training set using the best hyperparameters
        rfc = RandomForestClassifier(random_state=42, **best_params)
        rfc.fit(X_train, y_train)

        # Evaluate the classifier on the validation set
        y_val_pred = rfc.predict(X_val)
        print('Validation Classification Report:')
        print(classification_report(y_val, y_val_pred))

        # Evaluate the classifier on the test set
        y_test_pred = rfc.predict(X_test)
        print('Test Classification Report:')
        print(classification_report(y_test, y_test_pred))

        RandomForestClassifier_accuracy = accuracy_score(y_test, y_test_pred)
        print('Accuracy using RandomForestClassifier: {:.2f}%'.format(RandomForestClassifier_accuracy * 100))

        acc_RANDOM.append(RandomForestClassifier_accuracy)


Average_RandomClassifier = np.mean(acc_RANDOM)
print(Average_RandomClassifier)

        # # generate a random dataset
        # X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=42)
        #
        # # fit a random forest classifier
        # rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        # rfc.fit(X, y)
        #
        # # extract the first tree from the forest
        # tree = rfc.estimators_[0]
        #
        # # export the tree to a Graphviz dot file
        # dot_data = export_graphviz(tree, out_file=None,
        #                            feature_names=['feature_{}'.format(i) for i in range(X.shape[1])],
        #                            class_names=['class_0', 'class_1'],
        #                            filled=True, rounded=True, special_characters=True)
        # # create a Graphviz graph from the dot file
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # graph
        #
        # # display the graph as an image
        # Image(graph.create_png())