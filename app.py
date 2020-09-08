# Load Core Pkg
import streamlit as st
# EDA Pkg
import pandas as pd
import os
import numpy as np
#from PIL import Image

# Data Viz Pkg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ML Pkg
from tensorflow.keras.models import model_from_json
import joblib
import pickle as pkl


# #Get Value in Dictionary
# def get_value(val, my_dict):
#     for key, value in my_dict.items():
#         if val == key:
#             return value #return key


#Load Models (Simple Linear Regression)
def load_Models(model_file):
    loaded_Model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_Model


#Load Models (Linear Regression with TensorFlow)
def load_modelRTFK():
    model_weights = "static/models/model_weight.h5"
    model_json = "static/models/model.json"
    with open(model_json) as json_file:
        loaded_model = model_from_json(json_file.read())
    loaded_model.load_weights(model_weights)
    #loaded_model.summary()  # included to make it visible when model is reloaded
    return loaded_model


#Load Models (RandomForest)
def load_modelRF(model_file):
    loaded_modelRF = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_modelRF


def main():

    """Bahria Housing Price Predictor"""

    global prediction_RF
    st.title("Bahria Housing Price Predictor")
    acitvity = ['EDA', 'Prediction', 'About']
    choice = st.sidebar.selectbox('Chose An Activity', acitvity)

    # Load File
    df = pd.read_csv("data/all_FeaturedEngData_Bahria.csv", index_col =[0])
    X = df.drop('Price in PKR', axis=1)

    #EDA

    if choice == 'EDA':
        st.subheader("EDA Section")
        st.text("Exploratory Data Analysis")
        #Preview Data
        if st.checkbox("Preview Dataset"):
            number_input = st.number_input("Number to Show", 1, 20090, 1)
            st.dataframe(df.head(number_input))
        #Show Columns/Rows
        if st.button("Show Column Names Only"):
            st.write(df.columns)
        #Description
        if st.checkbox("Describe"):
            st.write(df.describe())
        #Shape
        if st.checkbox("Show Shape of Dataset"):
            st.write(df.shape)
            data_dim = st.radio("Show Dimensions by", ("Rows", "Columns"))
            if data_dim == "Rows":
                st.text("Number of Rows")
                st.write((df.shape[0]))
            elif data_dim == "Columns":
                st.text("Number of Columns")
                st.write((df.shape[1]))
            else:
                st.write(df.shape)

        #Selections
        if st.checkbox("Select by Columns"):
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect("Select Column/s", all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)

        if st.checkbox("Select by Rows"):
            selected_index = st.multiselect("Select Row/-s", df.head(20090).index)
            selected_rows = df.loc[selected_index]
            st.dataframe(selected_rows)

        # Value Counts
        if st.checkbox("Value Counts by Column"):
            all_columns1 = df.columns.tolist()
            selected_column = st.selectbox("Select A Column", all_columns1)
            #selected_columns1 = st.multiselect("Select Column/s", all_columns1)
            new_df1 = df[selected_column]
            #st.dataframe(new_df1)
            new_df1 = new_df1.iloc[:].value_counts()
            st.text("by {}".format(selected_column))
            st.write(new_df1)

        #Plots
        if st.checkbox("Show Correlation Plot[Matplotlib]"):
            plt.matshow(df.corr())
            st.pyplot()
        if st.checkbox("Show Correlation Plot[Seaborn]"):
            plt.figure(figsize=(20,10))
            st.text("Please be patient it takes to plot")
            st.write(sns.heatmap(df.corr(), annot=True))
            st.pyplot()
        if st.checkbox("Correlation by Column"):
            all_columns2 = df.columns.tolist()
            selected_column1 = st.selectbox("Select A Column", all_columns2)
            #corrby_Column = df.corr()[selected_column1].sort_values().plot(kind = 'bar')
            corrby_Column = df.corr()[selected_column1]#.sort_values()#.plot(kind='bar')
            st.write(corrby_Column)
            st.pyplot()
            # new_df2 = df[selected_column1]
            # st.dataframe(new_df2)
            # new_df2 = new_df2.iloc[:].value_counts()
            # st.text("by {}".format(selected_column))
            # st.write(new_df1)


    #df.corr()['col_name'][:-1].sort_values().plot(kind='bar')

    #PREDICTION
    elif choice == 'Prediction':
        st.subheader('Prediction Section')
        st.text("Enter appropriate values, e.g. you should not select House Type = 'House, Farm House\nin Bahria Apartments(Select Area Section). Similarly if particular 'Area Size' doesn't\nexit in specific precinct then it will give garbage results.")
        #paste mapped dictionaries
        #ML Aspect User Input
        bathrooms = st.slider("Select Number of bathrooms", 1, 10)
        bedrooms = st.slider("Select Number of bedrooms", 1, 10)
        sizeinMarla = st.number_input("Enter Size in Marla", 0.6, 159.99)
        house_Type = st.selectbox("Select House Type", ['Farm House','Flat','House','Penthouse'])
        areainBahria = st.selectbox("Select Area", df.columns[8:].tolist())

        #Processing User Input

        my_data1 = [bathrooms, bedrooms, sizeinMarla, 0, 0, 0, 0]
        my_data2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        my_data = my_data1 + my_data2

        if house_Type == 'Farm House':
            my_data[3] = 1
        elif house_Type == 'Flat':
            my_data[4] = 1
        elif house_Type == 'House':
            my_data[5] = 1
        else:
            my_data[6] = 1

        if areainBahria == 'Bahria Apartments, Bahria Town Karachi, Karachi, Sindh':
            my_data[7] = 1
        elif areainBahria == 'Bahria Farm House, Bahria Town Karachi, Karachi, Sindh':
            my_data[8] = 1
        elif areainBahria == 'Bahria Golf City, Bahria Town Karachi, Karachi, Sindh':
            my_data[9] = 1
        elif areainBahria == 'Bahria Heights, Bahria Town Karachi, Karachi, Sindh':
            my_data[10] = 1
        elif areainBahria == 'Bahria Hills, Bahria Town Karachi, Karachi, Sindh':
            my_data[11] = 1
        elif areainBahria == 'Bahria Homes - Iqbal Villas, Bahria Town - Precinct 2, Bahria Town Karachi, Karachi, Sindh':
            my_data[12] = 1
        elif areainBahria == 'Bahria Liberty Commercial, Bahria Town Karachi, Karachi, Sindh':
            my_data[13] = 1
        elif areainBahria == 'Bahria Paradise - Precinct 50, Bahria Paradise, Bahria Town Karachi, Karachi, Sindh':
            my_data[14] = 1
        elif areainBahria == 'Bahria Paradise - Precinct 51, Bahria Paradise, Bahria Town Karachi, Karachi, Sindh':
            my_data[15] = 1
        elif areainBahria == 'Bahria Paradise - Precinct 52, Bahria Paradise, Bahria Town Karachi, Karachi, Sindh':
            my_data[16] = 1
        elif areainBahria == 'Bahria Paradise - Precinct 54, Bahria Paradise, Bahria Town Karachi, Karachi, Sindh':
            my_data[17] = 1
        elif areainBahria == 'Bahria Paradise, Bahria Town Karachi, Karachi, Sindh':
            my_data[18] = 1
        elif areainBahria == 'Bahria Sports City, Bahria Town Karachi, Karachi, Sindh':
            my_data[19] = 1
        elif areainBahria == 'Bahria Tower, Bahria Town Karachi, Karachi, Sindh':
            my_data[20] = 1
        elif areainBahria == 'Bahria Town - Ali Block, Bahria Town - Precinct 12, Bahria Town Karachi, Karachi, Sindh':
            my_data[21] = 1
        elif areainBahria == 'Bahria Town - Jinnah Avenue, Bahria Town Karachi, Karachi, Sindh':
            my_data[22] = 1
        elif areainBahria == 'Bahria Town - Model Block, Bahria Town - Precinct 19, Bahria Town Karachi, Karachi, Sindh':
            my_data[23] = 1
        elif areainBahria == 'Bahria Town - Overseas Block, Bahria Town - Precinct 1, Bahria Town Karachi, Karachi, Sindh':
            my_data[24] = 1
        elif areainBahria == 'Bahria Town - Precinct 1, Bahria Town Karachi, Karachi, Sindh':
            my_data[25] = 1
        elif areainBahria == 'Bahria Town - Precinct 10, Bahria Town Karachi, Karachi, Sindh':
            my_data[26] = 1
        elif areainBahria == 'Bahria Town - Precinct 11, Bahria Town Karachi, Karachi, Sindh':
            my_data[27] = 1
        elif areainBahria == 'Bahria Town - Precinct 11-A, Bahria Town - Precinct 11, Bahria Town Karachi, Karachi, Sindh':
            my_data[28] = 1
        elif areainBahria == 'Bahria Town - Precinct 11-B, Bahria Town - Precinct 11, Bahria Town Karachi, Karachi, Sindh':
            my_data[29] = 1
        elif areainBahria == 'Bahria Town - Precinct 12, Bahria Town Karachi, Karachi, Sindh':
            my_data[30] = 1
        elif areainBahria == 'Bahria Town - Precinct 15, Bahria Town Karachi, Karachi, Sindh':
            my_data[31] = 1
        elif areainBahria == 'Bahria Town - Precinct 15-A, Bahria Town - Precinct 15, Bahria Town Karachi, Karachi, Sindh':
            my_data[32] = 1
        elif areainBahria == 'Bahria Town - Precinct 16, Bahria Town Karachi, Karachi, Sindh':
            my_data[33] = 1
        elif areainBahria == 'Bahria Town - Precinct 17, Bahria Town Karachi, Karachi, Sindh':
            my_data[34] = 1
        elif areainBahria == 'Bahria Town - Precinct 18, Bahria Town Karachi, Karachi, Sindh':
            my_data[35] = 1
        elif areainBahria == 'Bahria Town - Precinct 19, Bahria Town Karachi, Karachi, Sindh':
            my_data[36] = 1
        elif areainBahria == 'Bahria Town - Precinct 2, Bahria Town Karachi, Karachi, Sindh':
            my_data[37] = 1
        elif areainBahria == 'Bahria Town - Precinct 22, Bahria Town Karachi, Karachi, Sindh':
            my_data[38] = 1
        elif areainBahria == 'Bahria Town - Precinct 23-A, Bahria Town Karachi, Karachi, Sindh':
            my_data[39] = 1
        elif areainBahria == 'Bahria Town - Precinct 25, Bahria Town Karachi, Karachi, Sindh':
            my_data[40] = 1
        elif areainBahria == 'Bahria Town - Precinct 27, Bahria Town Karachi, Karachi, Sindh':
            my_data[41] = 1
        elif areainBahria == 'Bahria Town - Precinct 27-A, Bahria Town Karachi, Karachi, Sindh':
            my_data[42] = 1
        elif areainBahria == 'Bahria Town - Precinct 28, Bahria Town Karachi, Karachi, Sindh':
            my_data[43] = 1
        elif areainBahria == 'Bahria Town - Precinct 29, Bahria Town Karachi, Karachi, Sindh':
            my_data[44] = 1
        elif areainBahria == 'Bahria Town - Precinct 3, Bahria Town Karachi, Karachi, Sindh':
            my_data[45] = 1
        elif areainBahria == 'Bahria Town - Precinct 30, Bahria Town Karachi, Karachi, Sindh':
            my_data[46] = 1
        elif areainBahria == 'Bahria Town - Precinct 31, Bahria Town Karachi, Karachi, Sindh':
            my_data[47] = 1
        elif areainBahria == 'Bahria Town - Precinct 33, Bahria Town Karachi, Karachi, Sindh':
            my_data[48] = 1
        elif areainBahria == 'Bahria Town - Precinct 35, Bahria Sports City, Bahria Town Karachi, Karachi, Sindh':
            my_data[49] = 1
        elif areainBahria == 'Bahria Town - Precinct 36, Bahria Sports City, Bahria Town Karachi, Karachi, Sindh':
            my_data[50] = 1
        elif areainBahria == 'Bahria Town - Precinct 4, Bahria Town Karachi, Karachi, Sindh':
            my_data[51] = 1
        elif areainBahria == 'Bahria Town - Precinct 6, Bahria Town Karachi, Karachi, Sindh':
            my_data[52] = 1
        elif areainBahria == 'Bahria Town - Precinct 7, Bahria Town Karachi, Karachi, Sindh':
            my_data[53] = 1
        elif areainBahria == 'Bahria Town - Precinct 8, Bahria Town Karachi, Karachi, Sindh':
            my_data[54] = 1
        elif areainBahria == 'Bahria Town - Quaid Villas, Bahria Town - Precinct 2, Bahria Town Karachi, Karachi, Sindh':
            my_data[55] = 1
        elif areainBahria == 'Bahria Town Karachi, Karachi, Sindh':
            my_data[56] = 1
        elif areainBahria == 'Dominion Twin Towers, Bahria Town Karachi, Karachi, Sindh':
            my_data[57] = 1
        elif areainBahria == 'Others, Bahria Town Karachi, Karachi, Sindh':
            my_data[58] = 1


        ser = pd.Series(my_data, index= X.columns)
        st.text("You have Entered Following Data")
        #st.info(ser)
        st.write(ser)

        #For API
        ser_Dict = ser.to_dict()
        # st.write(ser_Dict)
        # st.json(ser_Dict)

        # Make Predictions
        st.subheader("Prediction")
        if st.checkbox("Make Predictions"):
            single_Sample = ser.to_numpy().reshape((1, -1))
            all_my_List = ["Choose Model","Linear Regression", "Linear Regression with Tensor Flow", "Random Forest"]
            #Model Selection
            model_Choice =  st.selectbox("Model Choice", all_my_List)
            if st.button("Predict"):

                if model_Choice == "Linear Regression":
                    model_Predictor = load_Models("static/models/01_RregressionWithoutScalling_Bahria.pkl")
                    my_bar = st.progress(0)
                    for p in range(100):
                        my_bar.progress(p + 1)
                        prediction = model_Predictor.predict(single_Sample)
                        prediction = int(prediction)
                    st.text("Price Prediction: PRK  {0:.2f} ".format(prediction))
                    st.success("Finished")

                elif model_Choice == "Linear Regression with Tensor Flow":

                    # Load scaler that was fitted on training data
                    with open("static/models/scaler.pkl", "rb") as infile:
                        scaler = pkl.load(infile)
                        ser_lRTFK = scaler.transform(ser.to_numpy().reshape(-1, 59))  # Note: not fit_transform

                        model_PredictorlRTFK = load_modelRTFK()

                    my_bar = st.progress(0)
                    for p in range(100):
                        my_bar.progress(p + 1)
                        prediction_lRTFK = model_PredictorlRTFK.predict(ser_lRTFK)
                        prediction_lRTFK = int(prediction_lRTFK)
                    st.success("Price Prediction: PRK  {0:.2f} ".format(prediction_lRTFK))


                elif model_Choice == "Random Forest":

                    model_rF = load_modelRF("static/models/01_RandomForestRegressor_Bahria.pkl")

                    my_bar = st.progress(0)
                    for p in range(100):
                        my_bar.progress(p + 1)
                        prediction_RF = model_rF.predict(single_Sample)
                        prediction_RF = int(prediction_RF)
                    st.text("Price Prediction: PKR {0:.2f}".format(prediction_RF))


    if choice == 'About':
        st.subheader("Wep App made with Streamlit by: ")
        st.info("mazqoty.01@gmail.com")



if __name__ == '__main__':
    main()









