# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re as re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


# %%
def process_data(data):
    # Name
    data['Title'] = list(map(lambda row: re.search(
        ' ([A-Za-z]+)\.', row).group(1), data.Name))
    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                           'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data['Title'] = LabelEncoder().fit_transform(data.Title)

    # Sex
    data['isMale'] = LabelEncoder().fit_transform(data.Sex)

    # Age
    data['Age'] = SimpleImputer(strategy='mean').fit_transform(
        data.Age.values.reshape(-1, 1))
    data['CatAge'] = LabelEncoder().fit_transform(pd.cut(data.Age, 5))

    # SibSp + Parch
    data['isAlone'] = list(
        map(lambda row: 1 if row == 1 else 0, data.SibSp + data.Parch + 1))

    # Fare
    data['Fare'] = SimpleImputer(strategy='median').fit_transform(
        data.Fare.values.reshape(-1, 1))
    data['CatFare'] = LabelEncoder().fit_transform(pd.qcut(data.Fare, 4))

    # Embarked
    data['Embarked'] = SimpleImputer(
        strategy='constant', fill_value='C').fit_transform(data.Embarked.values.reshape(-1, 1))
    data['Embarked'] = LabelEncoder().fit_transform(data.Embarked)

    # Cleanup unused columns
    data = data.drop(['PassengerId', 'Name',
                      'Sex', 'Age', 'SibSp',
                      'Parch', 'Ticket',
                      'Cabin', 'Fare'], axis=1)
    return data


# %%
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
y = train.Survived
train = train.drop('Survived', axis=1)
X = process_data(train)
X_to_pred = process_data(test)


# %%
classifiers = [
    XGBClassifier(),
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

# %%
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1)
accuracies = {}

for i_train, i_test in sss.split(X, y):
    X_train, X_test = X.values[i_train], X.values[i_test]
    y_train, y_test = y.values[i_train], y.values[i_test]

    for classifier in classifiers:
        name = classifier.__class__.__name__
        classifier.fit(X_train, y_train)
        preds = classifier.predict(X_test)
        acc = accuracy_score(y_test, preds)
        if name in accuracies:
            accuracies[name] += acc
        else:
            accuracies[name] = acc

for i in accuracies:
    accuracies[i] = accuracies[i] / 10.0

# %%
plt.xlabel('Accuracy')
plt.ylabel('Classifier')
plt.title('Classifier Accuracy')
sns.barplot(x=list(accuracies.values()), y=list(accuracies.keys()), color="g")
plt.savefig('img/classifiers.png')
print('Best classifier: ' + str(max(accuracies, key=accuracies.get)))

# %%
classifier = SVC()
classifier.fit(X, y)
y_pred = classifier.predict(X_to_pred)
results = list(zip(test.PassengerId, y_pred))
np.savetxt("data/preds-" + classifier.__class__.__name__ + ".csv", results, delimiter=",",
           header='PassengerId,Survived', comments='', fmt='%i')


# %%
