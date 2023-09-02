import streamlit as st
import pandas as pd
import pickle
import requests
import sklearn

st.title('ML Prediction App')
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
response = requests.get("https://github.com/Ye-Min-Ag/Simbolo-Final-Project-App/raw/main/my_model.pkl")
model_content = response.content
model = pickle.loads(model_content)
#try:
#except Exception as e:
    #st.error(f"An error occurred while loading the model: {str(e)}")
# Load the trained model using pickle
if uploaded_file is not None:
    # Read the uploaded .csv file
    data = pd.read_csv(uploaded_file)
    file_X = data.iloc[:,0:-1].values
    file_Y = data.iloc[:,-1].values
    predictions = model.predict(file_X)
    # Display the predictions
    st.write('Predictions:')
    st.write(predictions)
    st.write('True values:')
    st.write(file_Y)
