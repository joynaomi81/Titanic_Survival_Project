import streamlit as st
import joblib
import pandas as pd
import os

# Check if the model file exists
if os.path.exists('pretrained_model_pkl'):
    st.success("The model has been successfully loaded!")
    
    # Load the pre-trained model
    try:
        model = joblib.load('pretrained_model_pkl')
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()
else:
    st.error("The model file does not exist.")
    st.stop()  # Stop execution if the model file is missing

# Streamlit app title
st.title("Titanic Survival Prediction App")

# Collecting input data from user
st.header("Input Passenger Details")

# Input fields for user data
pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3])
sex = st.selectbox('Sex', ['Male', 'Female'])

# Convert inputs into a DataFrame for the model
data = {
    'Pclass': [pclass],
    'Sex': [1 if sex == 'Male' else 0]  # Encode 'Male' as 1 and 'Female' as 0
}
input_df = pd.DataFrame(data)

# Prediction button
if st.button('Predict'):
    try:
        prediction = model.predict(input_df)
        # Display the result
        if prediction[0] == 1:
            st.success("This passenger would have survived!")
        else:
            st.error("This passenger would not have survived.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
