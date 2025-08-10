import streamlit as st
import pandas as pd
import pickle
import io

# --- Load your saved models ---
with open('multilinear_regression_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)

with open('decision_tree_regressor.pkl', 'rb') as f:
    tree_model = pickle.load(f)

result = pd.read_csv('https://raw.githubusercontent.com/rr2734/rashmir/refs/heads/main/train.csv')
numeric_cols=result.select_dtypes(include=['int64', 'float64']).columns.tolist()
object_cols = result.select_dtypes(include='object').columns.tolist()
#result_encoded=pd.get_dummies(result,columns =object_cols, drop_first =True)

# --- Define required columns ---
  # example
categorical_cols = object_cols      # example

# --- UI ---
st.title("House Price Prediction from Excel")

st.write("""
Upload an Excel file containing the following columns:
- **Numeric**: {}
- **Categorical**: {}
""".format(", ".join(numeric_cols), ", ".join(categorical_cols)))

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)

        # --- Validate columns ---
        missing_numeric = [col for col in numeric_cols if col not in df.columns]
        missing_categorical = [col for col in categorical_cols if col not in df.columns]

        if missing_numeric or missing_categorical:
            st.error(f"Missing required columns: {missing_numeric + missing_categorical}")
        else:
            st.success("File loaded successfully!")

            # --- Make predictions ---
            linear_preds = linear_model.predict(df)
            tree_preds = tree_model.predict(df)

            # --- Show results ---
            results_df = df.copy()
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