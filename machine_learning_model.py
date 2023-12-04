# imports
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

