import streamlit as st
import pickle
import pandas as pd
import base64

st.title("Telco Customer Churn Predictions")
st.subheader("BYOP for IPBA Batch-17, Group-F")

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

file = st.file_uploader("Please upload the file for prediction", type='csv')
required_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'SeniorCitizen',
       'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
       'MultipleLines_No', 'MultipleLines_No phone service',
       'MultipleLines_Yes', 'InternetService_DSL',
       'InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'OnlineBackup_No',
       'OnlineBackup_No internet service', 'OnlineBackup_Yes',
       'DeviceProtection_No', 'DeviceProtection_No internet service',
       'DeviceProtection_Yes', 'TechSupport_No',
       'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No',
       'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No internet service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
       'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

def create_download_link(df, filename="data_with_predictions.csv", download_filename = "Download CSV File"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{download_filename}</a>'
    return href

if st.button("Get Predictions"):        
    if file:
        test_data = pd.read_csv(file)
        
        # Check if the columns match the expected columns for prediction
        expected_columns = required_columns  # Define the expected columns for prediction
        if not set(expected_columns).issubset(set(test_data.columns)):
            st.error("Uploaded file does not contain the expected columns for prediction.")
            sample_df = pd.DataFrame(columns = required_columns)
            download_link = create_download_link(sample_df, filename = "Sample File for Telco Customer Churn.csv", download_filename= "Sample File Download")
            st.markdown(download_link, unsafe_allow_html=True)
        else:
            test_pred = loaded_model.predict(test_data)
            test_data["y_pred"] = test_pred
            st.dataframe(test_data)
            if test_pred is not None:
                download_link = create_download_link(test_data, filename = 'Predicted Telco Customer Churn.csv', download_filename= "Download predictions")
                st.markdown(download_link, unsafe_allow_html=True)
