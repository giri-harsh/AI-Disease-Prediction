import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st
import random

# Data Collection & Analysis
parkinsons_data = pd.read_csv(r'C:\Users\Harsh Giri\OneDrive\Documents\!Programing Language\Python\Internship\Microsoft SAP AICTE\Dataset\parkinsons.csv')



# Grouping the data based on the target variable and calculating the mean only for numeric columns
numeric_columns = parkinsons_data.select_dtypes(include=[np.number])  # Select only numeric columns
# print(numeric_columns.groupby('status').mean())





# Data Pre-Processing
X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']

# Splitting the data into training and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Data Standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Model Training (Support Vector Machine Model)
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Model Evaluation - Accuracy Score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
# print('Accuracy score of training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
# print('Accuracy score of test data : ', test_data_accuracy)




st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="wide"  # Use wide layout for better alignment
)


# st.title("Parkinson's Disease Prediction System")


# st.header("Enter the following details:")

# name = st.text_input("Enter the name of the person:")
# MDVP_Fo = st.number_input("Enter MDVP:Fo(Hz):", format="%.6f")
# MDVP_Fhi = st.number_input("Enter MDVP:Fhi(Hz):", format="%.6f")
# MDVP_Flo = st.number_input("Enter MDVP:Flo(Hz):", format="%.6f")
# MDVP_Jitter = st.number_input("Enter MDVP:Jitter(%):", format="%.6f")
# MDVP_Jitter_Abs = st.number_input("Enter MDVP:Jitter(Abs):", format="%.6f")
# MDVP_RAP = st.number_input("Enter MDVP:RAP:", format="%.6f")
# MDVP_PPQ = st.number_input("Enter MDVP:PPQ:", format="%.6f")
# Jitter_DDP = st.number_input("Enter Jitter:DDP:", format="%.6f")
# MDVP_Shim = st.number_input("Enter MDVP:Shimmer:", format="%.6f")
# MDVP_Shim_dB = st.number_input("Enter MDVP:Shimmer(dB):", format="%.6f")
# Shimmer_APQ3 = st.number_input("Enter Shimmer:APQ3:", format="%.6f")
# Shimmer_APQ5 = st.number_input("Enter Shimmer:APQ5:", format="%.6f")
# MDVP_APQ = st.number_input("Enter MDVP:APQ:", format="%.6f")
# Shimmer_DDA = st.number_input("Enter Shimmer:DDA:", format="%.6f")
# NHR = st.number_input("Enter NHR:", format="%.6f")
# HNR = st.number_input("Enter HNR:", format="%.6f")
# RPDE = st.number_input("Enter RPDE:", format="%.6f")
# DFA = st.number_input("Enter DFA:", format="%.6f")
# spread1 = st.number_input("Enter spread1:", format="%.6f")
# spread2 = st.number_input("Enter spread2:", format="%.6f")
# D2 = st.number_input("Enter D2:", format="%.6f")
# PPE = st.number_input("Enter PPE:", format="%.6f")




# if st.button("Predict Parkinson's"):
#     # Collect input data
#     input_data = [
#         MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Jitter_Abs,
#         MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shim, MDVP_Shim_dB,
#         Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR,
#         HNR, RPDE, DFA, spread1, spread2, D2, PPE
#     ]
    

#     # Convert to numpy array
#     input_data_as_numpy_array = np.asarray(input_data)

#     # Reshape and scale the input data
#     input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
#     std_data = scaler.transform(input_data_reshaped)

#     # Make prediction
#     prediction = model.predict(std_data)

#     # Display result
#     if prediction[0] == 0:
#         st.success(f"{name} is not likely to have Parkinson's Disease.")
#     else:
#         st.warning(f"{name} is likely to have Parkinson's Disease.")








# Set page configuration


# Add a custom background image
page_bg = """
<style>
    body {
        background-image: url("https://www.transparenttextures.com/patterns/cubes.png");
        background-size: cover;
    }
    .stButton>button {
        background-color: #4CAF50; 
        color: white;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        border-radius: 8px;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title with a hospital emoji
st.title("🧑‍⚕️ Parkinson's Disease Prediction")

# Add a random Parkinson's disease fact
facts = [
    "Parkinson's disease affects approximately 10 million people worldwide.",
    "Men are 1.5 times more likely to develop Parkinson's disease than women.",
    "The average age of onset for Parkinson's disease is 60.",
    "Exercise can help manage Parkinson's symptoms and improve quality of life.",
    "There is no cure for Parkinson's disease, but treatment options are available to manage symptoms."
]
st.sidebar.title("🧠 Did You Know?")
st.sidebar.info(random.choice(facts))

# Align the input boxes in columns
st.subheader("Please enter the following details:")

# Create columns for alignment
col1, col2, col3 = st.columns(3)

with col1:
    name = st.text_input("Enter the name of the person:")
    MDVP_Fo = st.number_input("Enter MDVP:Fo(Hz):", format="%.6f")
    MDVP_Fhi = st.number_input("Enter MDVP:Fhi(Hz):", format="%.6f")
    MDVP_Flo = st.number_input("Enter MDVP:Flo(Hz):", format="%.6f")
    MDVP_Jitter = st.number_input("Enter MDVP:Jitter(%):", format="%.6f")
    MDVP_Jitter_Abs = st.number_input("Enter MDVP:Jitter(Abs):", format="%.6f")

with col2:
    MDVP_RAP = st.number_input("Enter MDVP:RAP:", format="%.6f")
    MDVP_PPQ = st.number_input("Enter MDVP:PPQ:", format="%.6f")
    Jitter_DDP = st.number_input("Enter Jitter:DDP:", format="%.6f")
    MDVP_Shim = st.number_input("Enter MDVP:Shimmer:", format="%.6f")
    MDVP_Shim_dB = st.number_input("Enter MDVP:Shimmer(dB):", format="%.6f")
    Shimmer_APQ3 = st.number_input("Enter Shimmer:APQ3:", format="%.6f")

with col3:
    Shimmer_APQ5 = st.number_input("Enter Shimmer:APQ5:", format="%.6f")
    MDVP_APQ = st.number_input("Enter MDVP:APQ:", format="%.6f")
    Shimmer_DDA = st.number_input("Enter Shimmer:DDA:", format="%.6f")
    NHR = st.number_input("Enter NHR:", format="%.6f")
    HNR = st.number_input("Enter HNR:", format="%.6f")
    RPDE = st.number_input("Enter RPDE:", format="%.6f")
    

# Final column for remaining inputs
DFA =st.number_input("Enter DFA:", format="%.6f")
spread1 = st.number_input("Enter spread1:", format="%.6f")
spread2 = st.number_input("Enter spread2:", format="%.6f")
D2 = st.number_input("Enter D2:", format="%.6f")
PPE = st.number_input("Enter PPE:", format="%.6f")

# Submit button
if st.button("🧠 Predict Parkinson's"):
    input_data = [
        MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Jitter_Abs,
        MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shim, MDVP_Shim_dB,
        Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR,
        HNR, RPDE, DFA, spread1, spread2, D2, PPE
    ]
    
 # Convert to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape and scale the input data
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)

    # Make prediction
    prediction = model.predict(std_data)

    # Display result
    if prediction[0] == 0:
        st.success(f"{name} is not likely to have Parkinson's Disease.")
    else:
        st.warning(f"{name} is likely to have Parkinson's Disease.")

