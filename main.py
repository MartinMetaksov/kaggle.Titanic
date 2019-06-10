# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier


# %%
# read data and drop meaningless columns
def preprocess_data(X):
    y = None
    if hasattr(X, 'Survived'):
        y = X.Survived
        X = X.drop('Survived', axis=1)
    X = X.drop('PassengerId', axis=1)
    X = X.drop('Name', axis=1)
    X = X.drop('Ticket', axis=1)
    # TODO: perhaps the Cabin information can
    # be related and used for something...
    X = X.drop('Cabin', axis=1)

    # fix NaN age values
    imputer_age = SimpleImputer(strategy="median")  # TODO: try with mean
    imputer_age = imputer_age.fit(X.Age.values.reshape(-1, 1))
    X.Age = imputer_age.transform(X.Age.values.reshape(-1, 1))

    # fix NaN fare values
    imputer_fare = SimpleImputer(strategy="mean")  # TODO: try with median
    imputer_fare = imputer_fare.fit(X.Fare.values.reshape(-1, 1))
    X.Fare = imputer_fare.transform(X.Fare.values.reshape(-1, 1))

    # rename Sex to isMale - 0 for female, 1 for male
    X = X.rename(columns={'Sex': 'isMale'})
    le_sex = LabelEncoder()
    le_sex = le_sex.fit(X.isMale)
    X.isMale = le_sex.transform(X.isMale)

    # split Embarked into 2 categories
    X.Embarked[X.Embarked.isnull()] = 'C'  # TODO: try with S or Q
    le_embarked = LabelEncoder()
    le_embarked = le_embarked.fit(X.Embarked)
    X.Embarked = le_embarked.transform(X.Embarked)
    ohe = OneHotEncoder()
    ohe = ohe.fit(X.Embarked.values.reshape(-1, 1))
    _ = ohe.transform(X.Embarked.values.reshape(-1, 1)).toarray()
    X['EmbarkedInQueenstown'] = _[:, 1]
    X['EmbarkedInSouthampton'] = _[:, 2]
    X = X.drop('Embarked', axis=1)

    # feature scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # return
    return (X, y)


# %%
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
X_train, y_train = preprocess_data(train)
X_test, _ = preprocess_data(test)

# %%
# grid search
parameters = [
    # {
    #     'C': [1, 10, 100, 1000],
    #     'kernel': ['rbf'],
    #     'gamma': [0.5, 0.1, 0.01, 0.001]
    # },
    {
        'min_child_weight': [1, 5, 10],
        'gamma': [5, 2, 1.5, 1, 0.5, 0.1, 0.01, 0.001],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'n_estimators': [10, 50, 100, 300],
        'max_depth': [3, 4, 5],
        'booster': ['gbtree', 'gblinear', 'dart'],
    }]
grid_search = GridSearchCV(
    estimator=XGBClassifier(), param_grid=parameters,  # estimator = SVC()
    scoring='accuracy', cv=10, n_jobs=-1, verbose=1)
classifier = grid_search.fit(X_train, y_train).best_estimator_

# %%
# predict and compare results
y_pred_train = classifier.predict(X_train)
cm = confusion_matrix(y_train, y_pred_train)
accuracy = accuracy_score(y_train, y_pred_train)
print(cm)
print('Correct: ' + str(cm[0, 0] + cm[1, 1]))
print('Incorrect: ' + str(cm[0, 1] + cm[1, 0]))
print('Accuracy: ' + str(accuracy))

# %%
y_pred = classifier.predict(X_test)
results = list(zip(test.PassengerId, y_pred))

# %%
np.savetxt("data/test_pred_opt.csv", results, delimiter=",",
           header='PassengerId,Survived', fmt='%i')


#%%
