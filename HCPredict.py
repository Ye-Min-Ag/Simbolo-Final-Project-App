import streamlit as st
import pandas as pd
import pickle
import requests
import sklearn

st.title('2021 Myanmar Regional Poverty Headcount Prediction & Analysis')
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])

response = requests.get("https://github.com/Ye-Min-Ag/Simbolo-Final-Project-App/raw/main/my_model.pkl")
model_content = response.content
model = pickle.loads(model_content)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
if uploaded_file is not None:
    # Read the uploaded .csv file
    data = pd.read_csv(uploaded_file)
    file_X = data.iloc[:,0:-1].values
    file_Y = data.iloc[:,-1].values
    scaler.fit(file_X) 
    test_X = scaler.transform(file_X)
    predictions = model.predict(test_X)
    st.write(f'Predictions for the year 2021:')
    st.write(predictions)
    if len(predictions) > 0 and len(predictions) == len(file_Y):
        # Create a DataFrame to make it easier to work with the data
        results_df = pd.DataFrame({'Predictions': predictions, 'True Values': file_Y})
        # Line chart
        st.line_chart(results_df)
    else:
        st.error("Error: Predictions and True Values have mismatched lengths or are empty.")
    st.header("Why are Patterns not accurate?")
    
  






