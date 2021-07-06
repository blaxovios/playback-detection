# %% Import dataset
csv_stats_file = 'C:\\Users\\tsepe\\Downloads\\Video\\Analysis File\\Copy of Timbaland - Apologize ft OneRepublic DELAYED.mp4.stats.csv'

# %% Extract data and put them in a dataframe
import pandas as pd

df = pd.read_csv(csv_stats_file, sep=';')
df.info()
df.head()
df.describe()

# %% Make a plot from longitude and latitude and show pricing and population density as described below
import matplotlib.pyplot as plt

# #### Visualizing geographical data
#
# Since there is geographical information (latitude and longitude), it is a good idea to create a scatterplot of all districts to visualize the data.

df.plot(kind="scatter", x="Longitude", y="Latitude")
plt.show()

# This looks like California, but other than that it is hard to see any particular pattern. Setting the alpha option to 0.1 makes it much easier to visualize the places where there is a high density of data points.

df.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.1)
plt.show()

# We can now clearly see the high-density areas, namely the Bay Area and around Los Angeles, San Diego etc. Now let us look at the housing prices. The radius of each circle represents the district’s population (option `s`), and the color represents the price (option `c`). We will use a predefined color map (option `cmap`) called jet, which ranges from blue (low values) to red (high prices).

df.plot(kind='scatter', x='Longitude', y='Latitude', alpha=0.4,
        s=df['Population'] / 100, label='Population', figsize=(15, 10),
        c='MedHouseVal', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
plt.show()

# %% Show plots for the variables in the DataFrame
import seaborn as sns

sns.set(font_scale=2)
sns.set_style('whitegrid')

# %% scatter plots of variable plots
n_rows = 3
n_cols = 3

# Create the subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(30, 30))
for i, column in enumerate(df):
    sns.histplot(data=df, x=column, kde=True, ax=axes[i // n_cols, i % n_cols])

# %% plot the correlation coefficient between variables scaled in -1,1
plt.figure(figsize=(16, 16))
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)

# %% Make pairplots of the variables in the Dataframe
pp = sns.pairplot(data=df, vars=df.columns)

# %% standardise the variables
from sklearn.preprocessing import StandardScaler

scaled_columns = StandardScaler().fit_transform(df.values)
scaled_df = pd.DataFrame(scaled_columns, index=df.index, columns=df.columns)

# %% Create training and test sets
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(scaled_df, test_size=.4, random_state=42)

predictors_train = train_set.drop('MedHouseVal', axis=1)

labels_train = df.loc[predictors_train.index, 'MedHouseVal']

predictors_test = test_set.drop('MedHouseVal', axis=1)

labels_test = df.loc[predictors_test.index, 'MedHouseVal']

# %% Define metrics to estimate goodness of fit
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np


def metrics(labels, predictions):
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2_value = r2_score(labels, predictions)
    print(f'RMSE for Train Set: {rmse}')
    print(f'r_squared (Coefficient of Determination): {r2_value}')
    # return rmse,r2_value


# %% Perform linear regression to estimate target variable (MedHouseVal)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
# Train model
lin_reg.fit(predictors_train, labels_train)
# Use model to estimate target variable
predictions_train = lin_reg.predict(predictors_train)
# how well the trained model fits the training data?
print("Linear Regression Model (Training Set)")
metrics(labels_train, predictions_train)
# let's use the model to estimate the test data
predictions_test = lin_reg.predict(predictors_test)
print("Linear Regression Model (Test Set)")
metrics(labels_test, predictions_test)

# %%Use decision tree regressor

from sklearn.tree import DecisionTreeRegressor

DT_reg = DecisionTreeRegressor()
DT_reg.fit(predictors_train, labels_train)

predictions_train = DT_reg.predict(predictors_train)
print("Decision Tree Regressor (Training Set)")
metrics(labels_train, predictions_train)

predictions_test = DT_reg.predict(predictors_test)
print("Decision Tree Regressor (Test Set)")
metrics(labels_test, predictions_test)

# %% Use xgboost regressor (not optimal parameters)
import xgboost as xgb

XG_reg = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree', learning_rate=0.3, n_estimators=1000)
XG_reg.fit(predictors_train, labels_train)

predictions_train = XG_reg.predict(predictors_train)
print("XGBoost Regression (Training Set)")
metrics(labels_train, predictions_train)
print("XGBoost Regression (Test Set)")
predictions_test = XG_reg.predict(predictors_test)
metrics(labels_test, predictions_test)

# %% Use Random Forest regressor (non optimal parameters)
from sklearn.ensemble import RandomForestRegressor

RF_reg = RandomForestRegressor(n_estimators=50, random_state=42)

RF_reg.fit(predictors_train, labels_train)

predictions_train = RF_reg.predict(predictors_train)
print("Random Forest Regression (Training Set)")
metrics(labels_train, predictions_train)
print("Random Forest Regression (Test Set)")
predictions_test = RF_reg.predict(predictors_test)
metrics(labels_test, predictions_test)

# %% Search for optimal parameters of Random Forest using randomized search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'n_estimators': randint(low=1, high=500)
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=12, cv=4,
                                scoring='neg_mean_squared_error',
                                random_state=42)

rnd_search.fit(predictors_train, labels_train)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

feature_importances = rnd_search.best_estimator_.feature_importances_
for (feature_name, feature_importance) in sorted(zip(feature_importances, predictors_train.columns), reverse=True):
    print("{} -> {}".format(feature_name, feature_importance))

# %% Use Random Forest regressor (optimal parameter)
from sklearn.ensemble import RandomForestRegressor

RF_reg = RandomForestRegressor(n_estimators=rnd_search.best_params_['n_estimators'], random_state=42)

RF_reg.fit(predictors_train, labels_train)

predictions_train = RF_reg.predict(predictors_train)
print("Random Forest Regression (Training Set)-Optimized")
metrics(labels_train, predictions_train)
print("Random Forest Regression (Test Set) - Optimized")
predictions_test = RF_reg.predict(predictors_test)
metrics(labels_test, predictions_test)

# %% Search for optimal parameters of XGBoost using randomized search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4],
    'n_estimators': randint(low=1, high=500)
}

xgb_reg = xgb.XGBRFRegressor(objective='reg:squarederror', booster='gbtree', random_state=42)
rnd_search = RandomizedSearchCV(xgb_reg, param_distributions=param_distribs,
                                n_iter=12, cv=4,
                                scoring='neg_mean_squared_error',
                                random_state=42)

rnd_search.fit(predictors_train, labels_train)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

feature_importances = rnd_search.best_estimator_.feature_importances_
for (feature_name, feature_importance) in sorted(zip(feature_importances, predictors_train.columns), reverse=True):
    print("{} -> {}".format(feature_name, feature_importance))
# %% use optimal parameters in xgboost
import xgboost as xgb

XG_reg = xgb.XGBRegressor(objective='reg:squarederror', booster='gbtree',
                          learning_rate=rnd_search.best_params_['learning_rate'],
                          n_estimators=rnd_search.best_params_['n_estimators'])
XG_reg.fit(predictors_train, labels_train)

predictions_train = XG_reg.predict(predictors_train)
print("XGBoost Regression (Training Set)-Optimized")
metrics(labels_train, predictions_train)
print("XGBoost Regression (Test Set)- Optimized")
predictions_test = XG_reg.predict(predictors_test)
metrics(labels_test, predictions_test)


