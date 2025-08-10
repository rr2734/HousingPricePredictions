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

result = pd.read_csv('https://raw.githubusercontent.com/rr2734/rashmir/refs/heads/main/train.csv')

result= result.drop(columns=['LotFrontage', 'MasVnrArea','GarageYrBlt'])
numeric_cols=result.select_dtypes(include=['int64','float64']).columns.tolist()
object_cols = result.select_dtypes(include='object').columns.tolist()
result_encoded=pd.get_dummies(result,columns =object_cols, drop_first =True)
print(result_encoded)
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

X = result_final
Y=result['SalePrice']
X=sm.add_constant(X) #For statsmodels
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model_sm = sm.OLS(Y_train, X_train).fit()
model_sk = LinearRegression()
model_sk.fit(X_train, Y_train)
Y_pred = model_sk.predict(X_test)
r_squared = r2_score(Y_test, Y_pred)
print(f"R-squared: {r_squared}")

#Decision Tree Regression

result_classifier = DecisionTreeClassifier(random_state=42)
result_classifier.fit(X, Y)
Y_pred = result_classifier.predict(X_test)
X.columns.tolist()
#Too many features to create figure
accuracy = accuracy_score(Y_test, Y_pred) #


    # Save the multilinear regression model
with open('multilinear_regression_model.pkl', 'wb') as f:
    pickle.dump(model_sk, f)

    # Save the decision tree classifier model
with open('decision_tree_classifier.pkl', 'wb') as f1:
    pickle.dump(result_classifier, f1)
with open('multilinear_regression_model.pkl', 'rb') as f:
    loaded_RM = pickle.load(f)

with open('decision_tree_classifier.pkl', 'rb') as f1:

    loaded_DT = pickle.load(f1)


