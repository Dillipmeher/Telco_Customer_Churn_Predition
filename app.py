import streamlit as st
import pickle
import pandas as pd
import base64
from utils import *

st.title("Telco Customer Churn Predictions")
st.subheader("Churn..Turn... Fun....")
st.subheader("BYOP-D2S4G for IPBA Batch-17, Group-F")

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

file = st.file_uploader("Please upload the file for prediction", type='csv')
required_columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

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
            test_data_processed = data_preprocessing(test_data)
            test_pred = loaded_model.predict(test_data_processed)
            test_data["churn_predition"] = test_pred
            st.dataframe(test_data)
            if test_pred is not None:
                download_link = create_download_link(test_data, filename = 'Predicted Telco Customer Churn.csv', download_filename= "Download predictions")
                st.markdown(download_link, unsafe_allow_html=True)
