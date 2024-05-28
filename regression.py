# %% read data
import pandas as pd

train = pd.read_csv(
    "house-prices-advanced-regression-techniques/train.csv"
)
test = pd.read_csv(
    "house-prices-advanced-regression-techniques/test.csv"
)


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% SalePrice distribution
import seaborn as sns

sns.distplot(train["SalePrice"])


# %% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc
# SalePrice distribution w.r.t CentralAir
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.boxplot(x="CentralAir", y="SalePrice", data=train)
plt.title("SalePrice Distribution w.r.t CentralAir")
plt.xlabel("CentralAir")
plt.ylabel("SalePrice")
plt.show()

#  %% SalePrice distribution w.r.t OverallQual
plt.figure(figsize=(12, 6))
sns.boxplot(x="OverallQual", y="SalePrice", data=train)
plt.title("SalePrice Distribution w.r.t OverallQual")
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")
plt.show()

# %% SalePrice distribution w.r.t BldgType
plt.figure(figsize=(12, 6))
sns.boxplot(x="BldgType", y="SalePrice", data=train)
plt.title("SalePrice Distribution w.r.t BldgType")
plt.xlabel("BldgType")
plt.ylabel("SalePrice")
plt.show()

# %% SalePrice distribution w.r.t YearBuilt / Neighborhood
# SalePrice distribution w.r.t YearBuilt
plt.figure(figsize=(35, 10))
sns.boxplot(x="YearBuilt", y="SalePrice", data=train)
plt.title("SalePrice Distribution w.r.t YearBuilt")
plt.xlabel("YearBuilt")
plt.ylabel("SalePrice")
plt.xticks(rotation=45)
plt.show()

# %% SalePrice distribution w.r.t Neighborhood
plt.figure(figsize=(18, 6))
sns.boxplot(x="Neighborhood", y="SalePrice", data=train)
plt.title("SalePrice Distribution w.r.t Neighborhood")
plt.xlabel("Neighborhood")
plt.ylabel("SalePrice")
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better visibility
plt.show()

# %%
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np


def evaluate(reg, x, y):
    pred = reg.predict(x)
    result = np.sqrt(mean_squared_log_error(y, pred))
    return f"RMSLE score: {result:.3f}"


dummy_reg = DummyRegressor()

dummy_selected_columns = ["MSSubClass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["SalePrice"]

dummy_reg.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_reg, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_house_prices.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["SalePrice"]

print("Test Set Performance")
print(evaluate(dummy_reg, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy regressor?")



# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
# Load the dataset


# Selecting features and target variable
features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'YearRemodAdd']
X = train[features]
y = train['SalePrice']

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling missing values by filling them with the median of their respective columns
for column in ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']:
    X_train[column].fillna(X_train[column].median(), inplace=True)
    X_val[column].fillna(X_val[column].median(), inplace=True)

# Initializing the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Predicting on the validation set
predictions = model.predict(X_val)

# Calculating RMSLE for the validation set
rmsle = np.sqrt(mean_squared_log_error(y_val, predictions))
print('New RMSLE Score on the test set:', rmsle)
# %%

