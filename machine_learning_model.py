# imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

NUMBER_OF_REPETITIONS = 5

# data preprocessing
X = []
y = []

# first experiment
for i in range(NUMBER_OF_REPETITIONS):
    # Split the data into training and testing sets randomly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    print(f"Iteration {i + 1} - Training Set Size: {len(X_train)}, Testing Set Size: {len(X_test)}")
