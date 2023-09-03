import streamlit as st
import pandas as pd
import pickle
import requests
import sklearn
from PIL import Image

st.title('2021 Myanmar Regional Poverty Headcount Prediction & Analysis')
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
st.write('Note: This model is exclusive and need modified CSV file to make prediction')
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
    image = Image.open('Unknown.png')
    st.image(image, caption='Testing Data\'s Correlation Heatmap')  
    image_paths = ['Unknown-2.png','Unknown-3.png','Unknown-4.png']
    col1,col2,col3=st.beta_columns(3)
    for i, image_path in enumerate(image_paths):
        if i%3==0:
            with col1
                st.image(image_path, width=250, caption='Scatter Plot 1')
        elif i%3==1:
            with col2
                st.image(image_path, width=250, caption='Scatter Plot 2')
        else:
            with col3
                st.image(image_path, width=250, caption='Scatter Plot 3')
        





