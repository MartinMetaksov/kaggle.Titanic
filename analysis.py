# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import matplotlib.pylab as plt

# %%
X = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
y = X.Survived

# %%
X.head()

# %%
X.isnull().any()

# %%
test.isnull().any()

# %%
sns.barplot(x=X.Pclass, y=y)
plt.savefig('img/pclass_rel.png')

# %%
sns.barplot(x=X.Sex, y=y)
plt.savefig('img/gender_rel.png')

# %%
sns.barplot(x=X.Embarked, y=y)
plt.savefig('img/embarked_rel.png')

# %%
X.Age.describe()

# %%
age_transformer = ColumnTransformer(
    transformers=[('Age', SimpleImputer(strategy='mean'), ['Age'])])
X.Age = age_transformer.fit_transform(X)
X.CatAge = pd.cut(X.Age, 5)
catAge = sns.barplot(x=X.CatAge, y=y)
plt.xticks(rotation=30)
plt.savefig('img/age_rel.png', bbox_inches='tight')

# %%
X.Fare.describe()

# %%
fare_transformer = ColumnTransformer(
    transformers=[('Fare', SimpleImputer(strategy='median'), ['Fare'])])
X.Fare = fare_transformer.fit_transform(X)
X.CatFare = pd.qcut(X.Fare, 4)
sns.barplot(x=X.CatFare, y=y)
plt.savefig('img/fare_rel.png')

# %%
# Take care of:
X.FamilySize = X.SibSp + X.Parch + 1
sns.barplot(x=X.FamilySize, y=y)
plt.savefig('img/family_size_rel.png')

# %%
