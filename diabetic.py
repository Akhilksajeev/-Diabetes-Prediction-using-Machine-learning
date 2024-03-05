# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:20:12 2024

@author: akhil
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open("C:\Anaconda\envs\Machinelearning\Model_deployment\Model\diabetes_disease_detection_model.sav", 'rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
  
def main():
    # Giving a title
    st.title('Diabetes Prediction Web App')

    # Getting the input data from the user
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # Code for Prediction
    diagnosis = ''

    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        # Validate input values (optional)
        if not all([Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            st.warning("Please fill in all input fields.")
        else:
            # Convert input values to float
            input_data = [float(Glucose), float(BloodPressure), float(SkinThickness),
                          float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]

            # Debugging: Print input data
            print("Input Data:", input_data)

            # Make prediction
            try:
                diagnosis = diabetes_prediction(input_data)
                print("Diagnosis:", diagnosis)
            except Exception as e:
                print("Error during prediction:", e)

    st.success(diagnosis)

    
    
    
    
if __name__ == '__main__':
    main()
    