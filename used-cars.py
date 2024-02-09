# Author: Jackson Rini

#%%
# Data Manipulation
import numpy as np 
import pandas as pd 
import warnings
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.tree import export_graphviz
import pydot
#Train Test Split
from sklearn.model_selection import train_test_split

#Scaling
from sklearn.preprocessing import StandardScaler

#Models
import optuna
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

#Deep Neural Networks:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

cars = pd.read_csv("Data/used_car_cleaned.csv", delimiter=',')
cars2 = pd.read_csv("Data/UsedCarsSA_Clean_EN.csv", delimiter=',')
warnings.filterwarnings("ignore")


# print(cars.head())
# print(cars2.head())

# CLEANING CODE

cars2 = cars2.drop(['Origin', 'Color', 'Options', 'Engine_Size', 'Fuel_Type', 'Region', 'Negotiable'], axis=1)
cars2 = cars2.rename(columns={'Make': 'car_brand', 'Type': 'car_model', 'Year': 'car_model_year', 'Gear_Type': 'car_transmission', 'Mileage': 'car_driven', 'Price': 'car_price'})
cars2 = cars2[cars.columns]
cars2 = cars2[cars2['car_price'] != 0]
cars = pd.concat([cars, cars2], ignore_index=True)
print(cars)
cars.to_csv("used_car_data.csv")
# Calculate correlations for selected columns
correlation_columns = ['car_driven', 'car_model_year', 'car_price']
correlation_matrix = cars[correlation_columns].corr()

# Create a heatmap to visualize the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
sns.pairplot(data=cars,hue='car_transmission')
plt.show()

# price distribution

plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
plt.title('Price Distribution Plot')
sns.histplot(cars['car_price'])
plt.ticklabel_format(useOffset=False, style='plain', axis=('x'))

plt.subplot(1,2,2)
plt.title('Price Spread')
sns.boxplot(y=cars['car_price'])
plt.ticklabel_format(useOffset=False, style='plain', axis=('y'))

plt.show()

#price outliers
expensive = cars.loc[cars['car_price'] > 200000]
cars = cars[cars['car_price'] <= 200000]
cars = cars[cars['car_price'] >= 5000]

plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
plt.title('Price Distribution Plot')
sns.histplot(cars['car_price'])
plt.ticklabel_format(useOffset=False, style='plain', axis=('x'))

plt.subplot(1,2,2)
plt.title('Price Spread')
sns.boxplot(y=cars['car_price'])
plt.ticklabel_format(useOffset=False, style='plain', axis=('y'))

plt.show()

# Create a box plot to visualize the distribution of car prices by transmission type
sns.boxplot(x='car_transmission', y='car_price', data=cars)
plt.show()

plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
plt.title('Year Distribution Plot')
sns.histplot(cars['car_model_year'])
plt.ticklabel_format(useOffset=False, style='plain', axis=('x'))

plt.subplot(1,2,2)
plt.title('Year Spread')
sns.boxplot(y=cars['car_model_year'])
plt.ticklabel_format(useOffset=False, style='plain', axis=('y'))

plt.show()

# Create a bar plot to visualize the average car price by car brand
brandPrice = cars.groupby(['car_brand'])['car_price'].mean().reset_index()
sns.barplot(x='car_brand', y='car_price', data=brandPrice)
plt.xticks(rotation=90)
plt.show()



# milage outliers
plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
plt.title('Mileage Distribution Plot')
sns.histplot(cars['car_driven'])
plt.ticklabel_format(useOffset=False, style='plain', axis=('x'))

plt.subplot(1,2,2)
plt.title('Mileage Spread')
sns.boxplot(y=cars['car_driven'])
plt.ticklabel_format(useOffset=False, style='plain', axis=('y'))

plt.show()

print(cars.loc[cars['car_driven'] > 500000])
cars = cars[cars['car_driven'] <= 500000]
print(cars[cars['car_transmission'] == 118008.5011120378])
sns.boxplot(y='car_driven', x='car_transmission',data=cars)
plt.show()


cars = cars[cars['car_model_year'] >= 2000]
print(cars.dtypes)

plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
plt.title('Mileage Distribution Plot')
sns.histplot(cars['car_driven'])
plt.ticklabel_format(useOffset=False, style='plain', axis=('x'))

plt.subplot(1,2,2)
plt.title('Mileage Spread')
sns.boxplot(y=cars['car_driven'])
plt.ticklabel_format(useOffset=False, style='plain', axis=('y'))

plt.show()


# encoded categorical variables and feature choosing

cars_encoded = pd.get_dummies(cars, columns=['car_brand', 'car_model', 'car_transmission'])

# Split the data into features (X) and target (y)
X = cars_encoded.drop('car_price', axis=1)
y = cars_encoded['car_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


# %%
print('Random Forest Regression')
# Generate a baseline using DummyRegressor
dummy_model = DummyRegressor(strategy='mean') 
dummy_model.fit(X_train, y_train)

# Predict using the baseline model
y_pred_baseline = dummy_model.predict(X_test)

# Evaluate the baseline model
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
print(f'Baseline Mean Absolute Error: {mae_baseline}')
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
print(f'Baseline Mean Squared Error: {mse_baseline}')
print(f'Baseline Root Mean Squared Error: {mse_baseline**0.5}')

print("Hyperparameter Optimization Step")
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 150),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }
    rf_model = RandomForestRegressor(random_state=42, **params)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

rf_best_params = study.best_params
print(f'Best Hyperparameters: {rf_best_params}')
# rf_best_params = {'n_estimators': 869, 'max_depth': 133, 'min_samples_split': 7, 'min_samples_leaf': 1, 'max_features': 'sqrt'}

# Use the best model for predictions
best_rf_model = RandomForestRegressor(random_state=42, **rf_best_params)
best_rf_model.fit(X_train, y_train)
feature_importance = best_rf_model.feature_importances_

# Get the feature names
feature_names = X_train.columns

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance DataFrame
print("Feature Importance:")
print(feature_importance_df.head(10))


y_pred = best_rf_model.predict(X_test)
y_pred_rf = y_pred
errors = abs(y_pred - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2))
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
feature_importance = best_rf_model.feature_importances_

# print(f'Feature Importance: {feature_importance}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {mse**0.5}')
print(f'R-squared: {r2}')
print('Accuracy:', round(accuracy, 2), '%.')

# %%
print("Linear Regression")
X2 = cars.drop(['car_price', 'car_transmission', 'car_model', 'car_brand'], axis=1)
y2 = cars['car_price']
# Create a Linear Regression model
linear_model = LinearRegression()

# Split the data into training and testing sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3)

# Train the Linear Regression model
linear_model.fit(X_train2, y_train2)

# Make predictions on the test set
y_pred2 = linear_model.predict(X_test2)

# Evaluate the model
mse = mean_squared_error(y_test2, y_pred2)
r2 = r2_score(y_test2, y_pred2)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# %%
print("Grandient boosting Regression")

# Generate a baseline using DummyRegressor
dummy_model = DummyRegressor(strategy='mean')
dummy_model.fit(X_train, y_train)
y_pred_baseline = dummy_model.predict(X_test)

# Evaluate baseline model
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
print(f'Baseline Mean Absolute Error: {mae_baseline}')
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
print(f'Baseline Mean Squared Error: {mse_baseline}')
print(f'Baseline Root Mean Squared Error: {mse_baseline**0.5}')

print("Hyperparameter Optimization Step")
# objective function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    xgb_model = xgb.XGBRegressor(random_state=42, **params)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
xgb_best_params = study.best_params
print(f'Best Hyperparameters: {xgb_best_params}')
# xgb_best_params = {'n_estimators': 888, 'max_depth': 12, 'learning_rate': 0.059265067129644175, 'subsample': 0.6710882119982516, 'colsample_bytree': 0.9147549947728586}
# Train the model with the best hyperparameters
best_xgb_model = xgb.XGBRegressor(random_state=42, **xgb_best_params)
best_xgb_model.fit(X_train, y_train)
y_pred = best_xgb_model.predict(X_test)
y_pred_xgb = y_pred
# Evaluate the best model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {mse**0.5}')
print(f'R-squared: {r2}')

# Calculate accuracy metrics
errors = abs(y_pred - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# %%
# Model Visualization and Analysis
from sklearn.model_selection import cross_val_score
# cross-validation
scores = cross_val_score(best_rf_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Display the mean squared error scores
print("RF Cross-Validation Mean Squared Error:", -scores.mean())

scores2 = cross_val_score(best_xgb_model, X, y, cv=5, scoring='neg_mean_squared_error')

print("RF Cross-Validation Mean Squared Error:", -scores2.mean())


# Random Forest Visualization
plt.figure(figsize=(12, 8))

# Actual Prices
plt.scatter(X_test['car_driven'], y_test, label='Actual Prices', alpha=0.5)

# Random Forest Predictions
y_pred_rf = best_rf_model.predict(X_test)
plt.scatter(X_test['car_driven'], y_pred_rf, label='RF Predictions', alpha=0.5)

plt.xlabel('Distance Traveled (car_driven)')
plt.ylabel('Car Price')
plt.title('Random Forest: Actual vs. Predicted Prices over Distance Traveled')
plt.legend()
plt.show()

# XGBoost Visualization
plt.figure(figsize=(12, 8))

# Actual Prices
plt.scatter(X_test['car_driven'], y_test, label='Actual Prices', alpha=0.5)

# XGBoost Predictions
y_pred_xgb = best_xgb_model.predict(X_test)
plt.scatter(X_test['car_driven'], y_pred_xgb, label='XGB Predictions', alpha=0.5)

plt.xlabel('Distance Traveled (car_driven)')
plt.ylabel('Car Price')
plt.title('XGBoost: Actual vs. Predicted Prices over Distance Traveled')
plt.legend()
plt.show()

# Random Forest Visualization by Top 10 Car Brands
top_brands_rf = feature_importance_df['Feature'].str.extract(r'car_brand_(.*)').dropna()[0]

plt.figure(figsize=(15, 8))

for brand in top_brands_rf:
    brand_indices = X_test[X_test[f'car_brand_{brand}'] == 1].index

    if not brand_indices.empty:  # Check if there are samples for the current brand
        plt.scatter(X_test.loc[brand_indices, 'car_driven'], y_test.loc[brand_indices], label=f'Actual Prices - {brand}', alpha=0.5)

        y_pred_rf_brand = best_rf_model.predict(X_test.loc[brand_indices])
        plt.scatter(X_test.loc[brand_indices, 'car_driven'], y_pred_rf_brand, label=f'RF Predictions - {brand}', alpha=0.5)

plt.xlabel('Distance Traveled (car_driven)')
plt.ylabel('Car Price')
plt.title('Random Forest: Actual vs. Predicted Prices by Top 10 Car Brands over Distance Traveled')
plt.legend()
plt.show()

# XGBoost Visualization by Top 10 Car Brands
top_brands_xgb = feature_importance_df['Feature'].str.extract(r'car_brand_(.*)').dropna()[0]

plt.figure(figsize=(15, 8))

for brand in top_brands_xgb:
    brand_indices = X_test[X_test[f'car_brand_{brand}'] == 1].index

    if not brand_indices.empty:  # Check if there are samples for the current brand
        plt.scatter(X_test.loc[brand_indices, 'car_driven'], y_test.loc[brand_indices], label=f'Actual Prices - {brand}', alpha=0.5)

        y_pred_xgb_brand = best_xgb_model.predict(X_test.loc[brand_indices])
        plt.scatter(X_test.loc[brand_indices, 'car_driven'], y_pred_xgb_brand, label=f'XGB Predictions - {brand}', alpha=0.5)

plt.xlabel('Distance Traveled (car_driven)')
plt.ylabel('Car Price')
plt.title('XGBoost: Actual vs. Predicted Prices by Top 10 Car Brands over Distance Traveled')
plt.legend()
plt.show()



# %%
# Feature Engineering
# Depreciation rate of car based on mileage and accounting for inf values
cars['actual_depreciation_rate'] = cars['car_price'] / np.where(cars['car_driven'] == 0, 1, cars['car_driven'])
# print(cars['actual_depreciation_rate'].describe())

# print(cars['car_brand'].value_counts())
 
X = cars[['car_brand', 'car_driven']]
y = cars['actual_depreciation_rate']

X_encoded = pd.get_dummies(X, columns=['car_brand'])
X_encoded = X_encoded.drop(['car_brand_Hummer', 'car_brand_Other'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
# print(X_test.columns)

# Random Forest model
rf_model = RandomForestRegressor(random_state=42, **rf_best_params)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# XGBoost model
xgb_model = xgb.XGBRegressor(random_state=42, **xgb_best_params)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate models
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)

print(f'RF Mean Squared Error: {mse_rf}')
print(f'XGB Mean Squared Error: {mse_xgb}')

# Compare model predictions with actual depreciation rates
results_rf = pd.DataFrame({'Actual_Price': y_test, 'Predicted_RF': y_pred_rf})
results_xgb = pd.DataFrame({'Actual_Price': y_test, 'Predicted_XGB': y_pred_xgb})

# Outliers removal to make viewing easier
results_rf = results_rf[results_rf['Actual_Price'] < 5000]
results_xgb = results_xgb[results_xgb['Actual_Price'] < 5000]

# Visualize the results with a line of best fit
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.regplot(x='Actual_Price', y='Predicted_RF', data=results_rf, line_kws={'color': 'red'})
plt.title('Random Forest Model: Actual_Price vs Predicted Depreciation Rate')
plt.xlabel('Actual Depreciation Rate')
plt.ylabel('Predicted Depreciation Rate (RF)')

plt.subplot(1, 2, 2)
sns.regplot(x='Actual_Price', y='Predicted_XGB', data=results_xgb, line_kws={'color': 'red'})
plt.title('XGBoost Model: Actual_Price vs Predicted Depreciation Rate')
plt.xlabel('Actual Depreciation Rate')
plt.ylabel('Predicted Depreciation Rate (XGB)')

plt.tight_layout()
plt.show()

results_rf['Depreciation_Rate_RF'] = ((results_rf['Actual_Price'] - results_rf['Predicted_RF']) / results_rf['Actual_Price']) * 100

# Extract encoded 'car_brand' columns
brand_columns = [col for col in X_encoded.columns if col.startswith('car_brand_')]
results_rf['Car_Brand'] = X_test[brand_columns].idxmax(axis=1).apply(lambda x: x.split('_')[-1])

# Group by Car Brand and calculate average depreciation rate
depreciation_by_brand_rf = results_rf.groupby('Car_Brand')['Depreciation_Rate_RF'].mean().reset_index()


# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Car_Brand', y='Depreciation_Rate_RF', data=depreciation_by_brand_rf)
plt.xticks(rotation=90)
plt.title('Average Depreciation Rate by Car Brand (Random Forest)')
plt.show()

# Calculate depreciation rate for XGBoost
results_xgb['Depreciation_Rate_XGB'] = ((results_xgb['Actual_Price'] - results_xgb['Predicted_XGB']) / results_xgb['Actual_Price']) * 100

brand_columns_xgb = [col for col in X_encoded.columns if col.startswith('car_brand_')]
results_xgb['Car_Brand'] = X_test[brand_columns_xgb].idxmax(axis=1).apply(lambda x: x.split('_')[-1])

# Group by Car Brand and calculate average depreciation rate for XGB
depreciation_by_brand_xgb = results_xgb.groupby('Car_Brand')['Depreciation_Rate_XGB'].mean().reset_index()

# Create a bar plot for XGBoost
plt.figure(figsize=(12, 6))
sns.barplot(x='Car_Brand', y='Depreciation_Rate_XGB', data=depreciation_by_brand_xgb)
plt.xticks(rotation=90)
plt.title('Average Depreciation Rate (XGBoost) by Car Brand')
plt.show()

# Filter car brands with more than 50 cars
popular_brands = cars['car_brand'].value_counts()[cars['car_brand'].value_counts() > 50].index

# Filter results for popular brands
popular_results_rf = results_rf[results_rf['Car_Brand'].isin(popular_brands)]
popular_results_xgb = results_xgb[results_xgb['Car_Brand'].isin(popular_brands)]

# Get the lowest 3 depreciation rates
lowest_depreciation_rf = popular_results_rf.groupby('Car_Brand')['Depreciation_Rate_RF'].mean().nsmallest(3).reset_index()
lowest_depreciation_xgb = popular_results_xgb.groupby('Car_Brand')['Depreciation_Rate_XGB'].mean().nsmallest(3).reset_index()

print("Lowest 3 Depreciation Rates (Random Forest):")
print(lowest_depreciation_rf)

print("\nLowest 3 Depreciation Rates (XGBoost):")
print(lowest_depreciation_xgb)

# Filter results for popular brands
popular_results_rf = results_rf[results_rf['Car_Brand'].isin(popular_brands)]
popular_results_xgb = results_xgb[results_xgb['Car_Brand'].isin(popular_brands)]

top_brands_rf = depreciation_by_brand_rf[depreciation_by_brand_rf['Car_Brand'].isin(popular_brands)].nlargest(5, 'Depreciation_Rate_RF')
top_brands_xgb = depreciation_by_brand_xgb[depreciation_by_brand_xgb['Car_Brand'].isin(popular_brands)].nlargest(5, 'Depreciation_Rate_XGB')

low_brands_rf = depreciation_by_brand_rf[depreciation_by_brand_rf['Car_Brand'].isin(popular_brands)].nsmallest(5, 'Depreciation_Rate_RF')
low_brands_xgb = depreciation_by_brand_xgb[depreciation_by_brand_xgb['Car_Brand'].isin(popular_brands)].nsmallest(5, 'Depreciation_Rate_XGB')


print("Top 5 Car Brands with Highest Depreciation Rates (Random Forest):")
print(top_brands_rf)

print("\nTop 5 Car Brands with Highest Depreciation Rates (XGBoost):")
print(top_brands_xgb)

plt.figure(figsize=(12, 6))
sns.barplot(x='Car_Brand', y='Depreciation_Rate_RF', data=top_brands_rf, palette='viridis')
plt.title('Top 5 Car Brands with Highest Depreciation Rates (Random Forest)')
plt.xlabel('Car Brand')
plt.ylabel('Average Depreciation Rate')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Car_Brand', y='Depreciation_Rate_XGB', data=top_brands_xgb, palette='viridis')
plt.title('Top 5 Car Brands with Highest Depreciation Rates (XGBoost)')
plt.xlabel('Car Brand')
plt.ylabel('Average Depreciation Rate')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Car_Brand', y='Depreciation_Rate_RF', data=low_brands_rf, palette='viridis')
plt.title('Top 5 Car Brands with Lowest Depreciation Rates (Random Forest)')
plt.xlabel('Car Brand')
plt.ylabel('Average Depreciation Rate')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Car_Brand', y='Depreciation_Rate_XGB', data=low_brands_xgb, palette='viridis')
plt.title('Top 5 Car Brands with Lowest Depreciation Rates (XGBoost)')
plt.xlabel('Car Brand')
plt.ylabel('Average Depreciation Rate')
plt.xticks(rotation=45)
plt.show()
# %%
