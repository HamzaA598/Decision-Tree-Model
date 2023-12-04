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

df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['BP'] = label_encoder.fit_transform(df['BP'])
df['Cholesterol'] = label_encoder.fit_transform(df['Cholesterol'])
df['Drug'] = label_encoder.fit_transform(df['Drug'])


X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = df['Drug']

Accuracies = []


TreeSizes = []

results = {'Mean_Accuracy': [], 'Max_Accuracy': [], 'Min_Accuracy': [],
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


def Calculate():
    results['Mean_Accuracy'].append(sum(Accuracies) / NUMBER_OF_REPETITIONS)
    results['Max_Accuracy'].append(max(Accuracies))
    results['Min_Accuracy'].append(min(Accuracies))
    results['Mean_Tree_Size'].append(sum(TreeSizes) / NUMBER_OF_REPETITIONS)
    results['Max_Tree_Size'].append(max(TreeSizes))
    results['Min_Tree_Size'].append(min(TreeSizes))


def Experiment2():
    testSize = 0.7
    for i in range(NUMBER_OF_REPETITIONS):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
        # Create a decision tree classifier
        clf = DecisionTreeClassifier()

        # Train the classifier on the training data
        clf.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred = clf.predict(X_test)

        # Evaluate the performance of the classifier
        Accuracies.append(metrics.accuracy_score(y_test, y_pred))
        TreeSizes.append(clf.tree_.node_count)

        testSize -= 0.1

    Calculate()
    createReport()


# first experiment
for i in range(NUMBER_OF_REPETITIONS):
    # Split the data into training and testing sets randomly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    print(f"Iteration {i + 1} - Training Set Size: {len(X_train)}, Testing Set Size: {len(X_test)}")


# second experiment
Experiment2()

