# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns



dataset = pd.read_csv("housing.csv")
dataset.info()


# %%
dataset.dropna(inplace=True)
dataset.info()

# %%
X = dataset.drop(["median_house_value"],axis = 1)
y = dataset["median_house_value"] 
X.hist()
X["total_rooms"] = np.log(X["total_rooms"]+1)
X["total_bedrooms"] = np.log(X["total_rooms"]+1)
X["households"] = np.log(X["total_rooms"]+1)
X["population"] = np.log(X["total_rooms"]+1)
X.hist()
X

# %%
X = X.join(pd.get_dummies(X.ocean_proximity)).drop(["ocean_proximity"],axis = 1 )   
X


# %%
plt.figure(figsize=(15,8))
dataset.corr()
sns.heatmap(dataset.corr(),annot = True)


# %%
X["bedroom_ratio"] = X["total_bedrooms"] / X["total_rooms"] 
X["household_rooms"] = X["total_rooms"] /X["households"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
train_data = X_train.join(y_train)

test_data = X_test.join(y_test)

plt.figure(figsize=(15,8))
train_data.corr()
sns.heatmap(train_data.corr(),annot = True)

# %%
X_train, y_train = train_data.drop(["median_house_value"],axis = 1),  train_data["median_house_value"]

reg_model = LinearRegression()

reg_model.fit(X_train,y_train)



# %%
reg_model.score(X_test,y_test)




# %%
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor()
forest_model.fit(X_train,y_train)

forest_model.score(X_test,y_test)


