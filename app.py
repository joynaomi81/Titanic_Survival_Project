import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('/content/final_model.pkl')

# Title of the app
st.title('Titanic Survival Prediction')

# Input fields for user data
pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3])
sex = st.selectbox('Sex', ['Male', 'Female'])

# Convert inputs into a dataframe
data = {'Pclass': [pclass], 'Sex': [1 if sex == 'Male' else 0]}  # Encode Male as 1 and Female as 0
input_df = pd.DataFrame(data)

# Prediction button
if st.button('Predict'):
    # Make prediction
    prediction = model.predict(input_df)
    
    # Display the result
    if prediction[0] == 1:
        st.success('This passenger would have survived!')
    else:
        st.error('This passenger would not have survived.')

