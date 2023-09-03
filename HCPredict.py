import streamlit as st
import pandas as pd
import pickle
import requests
import sklearn
import geopandas as gpd
from io import BytesIO
import matplotlib.pyplot as plt
st.title('Myanmar Regional Poverty Headcount Prediction for a Specific Year')
user_input = st.text_input("For which year are you going to predict?: ")
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
st.write('Note: States & Regions must be in this order ')
data = ['Yangon - 0','Kachin - 1','Kayah - 2', 'Kayin - 3','Chin - 4','Sagaing - 5','Tanintharyi - 6','Bago - 7','Magway - 8','Mandalay - 9','Mon - 10','Rakhine - 11','Shan - 12','Ayeyarwady - 13','NayPyiTawCouncil - 14']
st.write('States & Regions')
for item in data:
    st.write(item)

response = requests.get("https://github.com/Ye-Min-Ag/Simbolo-Final-Project-App/raw/main/my_model.pkl")
model_content = response.content
model = pickle.loads(model_content)
response1 = requests.get('https://github.com/Ye-Min-Ag/Simbolo-Final-Project-App/raw/main/gadm41_MMR_1.shp')
map_content = respose1.content
#try:
#except Exception as e:
    #st.error(f"An error occurred while loading the model: {str(e)}")
# Load the trained model using pickle
from sklearn.preprocessing import MinMaxScaler
# Create a StandardScaler object
scaler = MinMaxScaler()
if uploaded_file is not None:
    # Read the uploaded .csv file
    data = pd.read_csv(uploaded_file)
    file_X = data.iloc[:,0:-1].values
    file_Y = data.iloc[:,-1].values
    scaler.fit(file_X) 
    test_X = scaler.transform(file_X)
    predictions = model.predict(test_X)
   
    if len(predictions) > 0 and len(predictions) == len(file_Y):
        # Create a DataFrame to make it easier to work with the data
        results_df = pd.DataFrame({'Predictions': predictions, 'True Values': file_Y})
        # Line chart
        st.line_chart(results_df)
    else:
        st.error("Error: Predictions and True Values have mismatched lengths or are empty.")
    # Display the predictions
    st.write(f'Predictions for the year {user_input}:')
    st.write(predictions)
    #st.write('True values:')
    #st.write(file_Y)
    # Load the Myanmar states/regions data (replace 'myanmar_shapefile.shp' with your file)
    shapefile_gdf = gpd.read_file(BytesIO(map_content))
    # Assuming 'state_region_name' is the common column
    merged_data = gdf.merge(predicted_data, left_on='state_region_name', right_on='state_region_name', how='left')
    # Plot the heatmap using predicted values
    plt.figure(figsize=(12, 8))
    plt.title(f"Heatmap of Predicted Values for {user_input}")
    gdf.plot(column='Predictions', cmap='coolwarm', linewidth=0.8, ax=plt.gca(), edgecolor='0.8', legend=True)
    plt.show()
    st.pyplot(plt)






