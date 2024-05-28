# %% read data
import pandas as pd

train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% visualize the dataset, starting with the Survied distribution
import seaborn as sns

sns.countplot(x="Survived", data=train)


# %% Survived w.r.t Pclass / Sex / Embarked ?
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.countplot(x="Pclass", hue="Survived", data=train, ax=axes[0])
axes[0].set_title("Survival Count by Pclass")
sns.countplot(x="Sex", hue="Survived", data=train, ax=axes[1])
axes[1].set_title("Survival Count by Sex")
sns.countplot(x="Embarked", hue="Survived", data=train, ax=axes[2])
axes[2].set_title("Survival Count by Embarked")
plt.tight_layout()
plt.show()

# %% Age distribution ?
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(train["Age"], bins=20, kde=True, ax=ax)
ax.set_title("Age Distribution of Passengers")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
plt.show()

# %% Survived w.r.t Age distribution ?
plt.figure(figsize=(10, 6))
sns.histplot(data=train, x='Age', hue='Survived', element='step', stat='density', common_norm=False)
sns.kdeplot(data=train, x='Age', hue='Survived', common_norm=False, fill=False, palette='coolwarm', alpha=0.5)
plt.title('Age Distribution with Respect to Survival')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend(title='Survived', labels=['Died', 'Survived'])
plt.show()

# %% SibSp / Parch distribution ?
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

sns.countplot(x="SibSp", data=train, ax=axes[0])
axes[0].set_title("SibSp Distribution")

sns.countplot(x="Parch", data=train, ax=axes[1])
axes[1].set_title("Parch Distribution")

plt.tight_layout()
plt.show()

# %% Survived w.r.t SibSp / Parch  ?
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

sns.countplot(x="SibSp", hue="Survived", data=train, ax=axes[0])
axes[0].set_title("Survival Count by SibSp")

sns.countplot(x="Parch", hue="Survived", data=train, ax=axes[1])
axes[1].set_title("Survival Count by Parch")

plt.tight_layout()
plt.show()

# %% Dummy Classifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


def evaluate(clf, x, y):
    pred = clf.predict(x)
    result = f1_score(y, pred)
    return f"F1 score: {result:.3f}"


dummy_clf = DummyClassifier(random_state=2020)

dummy_selected_columns = ["Pclass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["Survived"]

dummy_clf.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_clf, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_titanic.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["Survived"]

print("Test Set Performance")
print(evaluate(dummy_clf, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy classifier?")


# %% Your solution to this classification problem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rf_clf = RandomForestClassifier(random_state=2020)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='f1')
grid_search.fit(dummy_train_x, dummy_train_y)
best_rf_clf = grid_search.best_estimator_
best_rf_score = grid_search.best_score_
print("Best Hyperparameters:", grid_search.best_params_)
print("Best F1 score on training set:", best_rf_score)
print("Test Set Performance")
print(evaluate(best_rf_clf, dummy_test_x,dummy_test_y))

# %%
