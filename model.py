# imports
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

NUMBER_OF_REPETITIONS = 5


# data preprocessing
df = pd.read_csv('./data/drug.csv')
missing_values_count = df.isna().sum()

# replace missing numerical data with the mean
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Na_to_K']] = imputer.fit_transform(df[['Age', 'Na_to_K']])

# drop missing categorial data
df.dropna(inplace=True)

# encoding categorial values
label_encoder = LabelEncoder()

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = df['Drug']


# first experiment
best_classifier = None
best_accuracy = 0
for i in range(NUMBER_OF_REPETITIONS):
    # Split the data into training and testing sets randomly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

    # Apply encoding on training data
    for column in ['Sex', 'BP', 'Cholesterol']:
        label_encoder.fit(X_train[column])
        X_train[column] = label_encoder.transform(X_train[column])
        X_test[column] = label_encoder.transform(X_test[column])

    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train, y_train)
    tree_size = clf.tree_.node_count
    # uses predict to get y_predict and compares it with y_test to calculate the score
    accuracy = clf.score(X_test, y_test)
    if accuracy > best_accuracy:
        best_classifier = clf
        best_accuracy = accuracy
        
print("best classifier has an accuracy = ", best_accuracy, "the number of nodes in the tree = ", clf.tree_.node_count, end = "")


# second experiment


results = {'Train_Size': [], 'Mean_Accuracy': [], 'Max_Accuracy': [], 'Min_Accuracy': [],
               'Mean_Tree_Size': [], 'Max_Tree_Size': [], 'Min_Tree_Size': []}


def createReport():
    report = pd.DataFrame(results)
    print(report)

    # Create plots

    # Plot accuracy against training set size
    plt.figure(figsize=(10, 5))
    plt.plot(report['Train_Size'], report['Mean_Accuracy'], label='Mean Accuracy')
    plt.scatter(report['Train_Size'], report['Max_Accuracy'], color='red', label='Max Accuracy', marker='o')
    plt.scatter(report['Train_Size'], report['Min_Accuracy'], color='green', label='Min Accuracy', marker='o')
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Training Set Size')
    plt.legend()
    plt.show()

    # Plot tree size against training set size
    plt.figure(figsize=(10, 5))
    plt.plot(report['Train_Size'], report['Mean_Tree_Size'], label='Mean Tree Size')
    plt.scatter(report['Train_Size'], report['Max_Tree_Size'], color='red', label='Max Tree Size', marker='o')
    plt.scatter(report['Train_Size'], report['Min_Tree_Size'], color='green', label='Min Tree Size', marker='o')
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('Tree Size')
    plt.title('Tree Size vs. Training Set Size')
    plt.legend()
    plt.show()


def Calculate(train_size, Accuracies, TreeSizes):
    results['Train_Size'].append(train_size)
    results['Mean_Accuracy'].append(sum(Accuracies) / NUMBER_OF_REPETITIONS)
    results['Max_Accuracy'].append(max(Accuracies))
    results['Min_Accuracy'].append(min(Accuracies))
    results['Mean_Tree_Size'].append(sum(TreeSizes) / NUMBER_OF_REPETITIONS)
    results['Max_Tree_Size'].append(max(TreeSizes))
    results['Min_Tree_Size'].append(min(TreeSizes))


def Experiment2():
    testSize = 0.7
    while testSize >= 0.3:
        Accuracies = []
        TreeSizes = []
        for i in range(NUMBER_OF_REPETITIONS):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)

            # Apply encoding on training data
            for column in ['Sex', 'BP', 'Cholesterol']:
                label_encoder.fit(X_train[column])
                X_train[column] = label_encoder.transform(X_train[column])
                X_test[column] = label_encoder.transform(X_test[column])
        
            # Create a decision tree classifier
            clf = DecisionTreeClassifier(criterion="entropy")

            # Train the classifier on the training data
            clf.fit(X_train, y_train)

            # Make predictions on the testing data
            y_pred = clf.predict(X_test)

            # Evaluate the performance of the classifier
            Accuracies.append(accuracy_score(y_test, y_pred))
            TreeSizes.append(clf.tree_.node_count)
        Calculate(1 - testSize, Accuracies, TreeSizes)
        testSize -= 0.1

    createReport()

Experiment2()