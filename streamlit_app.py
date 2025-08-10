import streamlit as st
import pickle
import io
import json
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
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor 

# --- Load your saved models ---
with open('multilinear_regression_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)

with open('decision_tree_regressor.pkl', 'rb') as f:
    tree_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


train_data = pd.read_csv('https://raw.githubusercontent.com/rr2734/rashmir/refs/heads/main/train.csv')  # path to your original training file

# Identify numeric and categorical columns
numeric_cols1 = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols1 = train_data.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
numeric_cols1.remove('SalePrice')

# Compute ranges for numeric columns
numeric_ranges = {col: (train_data[col].min(), train_data[col].max()) for col in numeric_cols1}

# Compute allowed categories for categorical columns
categorical_values = {col: sorted(train_data[col].dropna().unique()) for col in categorical_cols1}

# ----------------------
# 3. Display ranges and options
# ----------------------
st.title("🏠 House Price Prediction App")
st.subheader("Upload your test data Excel file")

with st.expander("📋 View required column ranges and categories"):
    st.write("**Numeric column ranges:**")
    for col, (min_val, max_val) in numeric_ranges.items():
        st.write(f"- {col}: {min_val} to {max_val}")

    st.write("**Categorical column allowed values:**")
    for col, values in categorical_values.items():
        st.write(f"- {col}: {values}")










result = pd.read_csv('https://raw.githubusercontent.com/rr2734/rashmir/refs/heads/main/train.csv')
numeric_cols=result.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('SalePrice')
object_cols = result.select_dtypes(include='object').columns.tolist()




uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)

        # --- Validate columns ---
        missing_numeric = [col for col in numeric_cols if col not in df.columns]
        missing_categorical = [col for col in object_cols if col not in df.columns]

        if missing_numeric or missing_categorical:
            st.error(f"Missing required columns: {missing_numeric + missing_categorical}")
        else:
            st.success("File loaded successfully!")

        with open('train_features.json') as f:
            train_features = json.load(f)

        test_data= df.drop(columns=['LotFrontage', 'MasVnrArea','GarageYrBlt'], errors = 'ignore')
        id_col = test_data['Id']
        numeric_cols = [col for col in test_data.select_dtypes(include=['int64', 'float64']).columns if col not in ['SalePrice', 'Id']]
        object_cols = test_data.select_dtypes(include='object').columns.tolist()
        test_encoded=pd.get_dummies(test_data,columns =object_cols, drop_first =True)
     
        numerical_features = test_encoded[numeric_cols]
        categorical_features = test_encoded.drop(columns=numeric_cols)
        scaled_numerical_features = pd.DataFrame(scaler.fit_transform(numerical_features), 
                                         columns=numerical_features.columns, 
                                         index=numerical_features.index)
        test_final = pd.concat([scaled_numerical_features, categorical_features], axis=1)
        test_final=test_final.dropna(axis=0)
        id_col = id_col.loc[test_final.index]

        for col in train_features:
            if col not in test_final.columns:
                test_final[col] = 0

        test_final= test_final[train_features]
       



            
        # --- Make predictions ---
        linear_preds = linear_model.predict(test_final)
        linear_preds = np.exp(linear_preds) - 1
        linear_preds = np.maximum(linear_preds, 0)

        tree_preds = tree_model.predict(test_final)
        
        # --- Show results ---
        results_df = test_final.copy()
        results_df['Id'] = id_col
        results_df['Linear_Regression_Pred'] = linear_preds
        results_df['Decision_Tree_Pred'] = tree_preds

        st.subheader("Predictions")
        st.dataframe(results_df)

        # --- Download option ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, index=False)
        st.download_button(
        label="Download Predictions as Excel",
        data=output.getvalue(),
        file_name="predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:

        st.error(f"Error reading file: {e}")






























