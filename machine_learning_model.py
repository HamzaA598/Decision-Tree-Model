import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


data = pd.read_csv('./data/drug.csv')

X = data.drop('Drug', axis=1)  # Features
y = data['Drug']  # Target variable


def experiment2():
    testSize = 0.7
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
        # Create a decision tree classifier
        clf = DecisionTreeClassifier()

        # Train the classifier on the training data
        clf.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = clf.predict(X_test)

        # Evaluate the performance of the classifier
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)




