import streamlit as st
import pandas as pd
import pickle
import requests
import sklearn
from PIL import Image

st.title('2021 Myanmar Regional Poverty Headcount Prediction & Analysis')
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
st.write('Note: This model is exclusive and need modified CSV file to make predictions.')
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
    st.header(f'Predictions for the year 2021:')
    st.write(predictions)
    if len(predictions) > 0 and len(predictions) == len(file_Y):
        # Create a DataFrame to make it easier to work with the data
        results_df = pd.DataFrame({'Predictions': predictions, 'True Values': file_Y})
        # Line chart
        st.line_chart(results_df)
    else:
        st.error("Error: Predictions and True Values have mismatched lengths or are empty.")
    st.header("Why are Patterns not accurate?")
    image = Image.open('Unknown.png')
    st.image(image, caption='Testing Data\'s Correlation Heatmap')  
    image_paths = ['Unknown-2.png','Unknown-3.png','Unknown-4.png']
    for i in image_paths:
        st.image(i, width=500, caption='Scatter Plots')
    st.write('These Heatmap and Scatter Plots reveal our dataset comes with certain data quality challenges. These may include outliers, inconsistencies, noises, etc,.')
    
    image7 = Image.open('Unknown-7.png')    
    st.image(image7, caption='Feature Importances Visualization for Training Data') 
    image8 = Image.open('Unknown-8.png')    
    st.image(image8, caption='Feature Importances Visualization for Testing Data') 
    st.write('Also, Machine Learning Models also had different important features which lead to generate inaccurate patterns.')

    image9 = Image.open('Unknown-9.png')    
    st.image(image9, caption='weights (coefficients)') 
    image10 = Image.open('Unknown-10.png')    
    st.image(image10, caption='Intercept') 
    st.write('These coefficients and intercept describe relationship between features and label.')




