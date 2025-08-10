import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error

result = pd.read_csv('https://raw.githubusercontent.com/rr2734/rashmir/refs/heads/main/train.csv')

result= result.drop(columns=['LotFrontage', 'MasVnrArea','GarageYrBlt'])
numeric_cols=result.select_dtypes(include=['int64','float64']).columns.tolist()
object_cols = result.select_dtypes(include='object').columns.tolist()
result_encoded=pd.get_dummies(result,columns =object_cols, drop_first =True)
#print(result_encoded)
numeric_cols=result.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.pop()
numerical_features = result_encoded[numeric_cols]
categorical_features = result_encoded.drop(columns=numeric_cols)
scaler = StandardScaler()
scaled_numerical_features = pd.DataFrame(scaler.fit_transform(numerical_features), 
                                         columns=numerical_features.columns, 
                                         index=numerical_features.index)
result_final = pd.concat([scaled_numerical_features, categorical_features], axis=1)
result_final1=pd.concat([scaled_numerical_features, categorical_features,result['SalePrice']], axis=1)
result_final2=result_final1.dropna(axis=0)

#print(result_final2.info())
X = result_final2.drop('SalePrice',axis=1)
#print(result_final2.info())
Y=result_final2['SalePrice']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = X_train.reset_index(drop=True)
Y_train = Y_train.reset_index(drop=True)


model_sk = LinearRegression()
model_sk.fit(X_train, Y_train)
Y_pred_lin = model_sk.predict(X_test)
r_squared_lin = r2_score(Y_test, Y_pred_lin)
#print(f"Linear Regression R-squared: {r_squared_lin}")

# Decision Tree Regression
model_tree = DecisionTreeRegressor(random_state=42)
model_tree.fit(X_train, Y_train)
Y_pred_tree = model_tree.predict(X_test)
r2_tree = r2_score(Y_test, Y_pred_tree)
mse_tree = mean_squared_error(Y_test, Y_pred_tree)
#print(f"Decision Tree R-squared: {r2_tree}")
#print(f"Decision Tree MSE: {mse_tree}")

# Save models
with open('multilinear_regression_model.pkl', 'wb') as f:
    pickle.dump(model_sk, f)

with open('decision_tree_regressor.pkl', 'wb') as f1:
    pickle.dump(model_tree, f1)

# Load models
with open('multilinear_regression_model.pkl', 'rb') as f:
    loaded_RM = pickle.load(f)

with open('decision_tree_regressor.pkl', 'rb') as f1:
    loaded_DT = pickle.load(f1)

plt.figure(figsize=(20,10))  # set size to make it readable
tree.plot_tree(model_tree, 
               feature_names=X_train.columns, 
               filled=True, 
               rounded=True, 
               fontsize=10)
plt.show()
