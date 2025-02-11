import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


diabetes_data=pd.read_csv(r'C:\Users\Harsh Giri\OneDrive\Documents\!Programing Language\Python\Internship\Microsoft SAP AICTE\Dataset\diabetes.csv')

diabetes_data['Outcome'].value_counts()
diabetes_data.groupby('Outcome').mean()
X = diabetes_data.drop(columns='Outcome',axis=1)
Y = diabetes_data['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_data['Outcome']    

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)



X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


st.title("Diabetes Metrics Input")

pregnancies = st.number_input("Number of Pregnancies")
glucose = st.number_input("Glucose Level")
blood_pressure = st.number_input("Blood Pressure")
skin_thickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin Level")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")
input_data = (pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,dpf,age)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)


if st.button("Predict Diabetes"):
    if (prediction[0]==0):
        st.success("Subject is Not Diabetic, No Need to Worry")
    else :
        st.warning("Subject is Diabetic, Please Refer to a Doctor")
    


