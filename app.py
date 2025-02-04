

import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
import os

# Load the models using pickle (Assuming model_dir is where you saved them)
model_dir = 'xgb_models_pickle'  
loaded_models = {}
for target_column in ['Y1', 'Y2', 'Y3', 'Y4']:  # Replace with your actual target columns
    model_file = os.path.join(model_dir, f'xgb_model_{target_column}.pkl')
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        loaded_models[target_column] = model
        print(f"Model for {target_column} loaded from: {model_file}")
    except FileNotFoundError:
        print(f"Model file for {target_column} not found: {model_file}")

# Streamlit app
st.title("Malaria Burden classification")

# Define the mapping for prediction labels
prediction_mapping = {
    'Y1': 'Lowest risk to malaria ',
    'Y2': 'Low risk to malaria',
    'Y3': 'High risk to malaria',
    'Y4': 'Highest risk to malaria'
}

# Input fields for features (Replace with your actual feature names)
feature_names = [
    'number of household members', 
    'Number of children 5 and under', 
    'Number of mosquito bed nets', 
    'Number of children under mosquito bed net previous night', 
    'Ideal number of children', 
    'Husband/partner worked in last 7 days/12 months', 
    'Cost of treatment of fever', 
    'Number of days after fever began sought advice or treatment', 
    'duration of pregnancy', 
    'proportion of children under 5 and below over the total no of household', 
    'propotion of ideal children under 5 and below over the no. of ideal children in the household', 
    'levels burden children under 5 who slept under mosquito net.', 
    'level of burden associated with proportion of mosquito net per no. of household.', 
    'level of burden associated with proportion of children under mosquito net previous night over ideal children.', 
    'Household has: electricity', 
    'Result of the malaria test', 
    'currently pregnant', 
    'Main floor material', 
    'Children under 5 slept under mosquito bed net last night (household questionnaire)', 
    'Type of mosquito bed net(s) slept under last night', 
    'Type of cooking fuel', 
    'Wall Material', 
    'RELATIONSHIP TO HOUSEHOLD HEAD', 
    'Respondents Currently Working', 
    'Literacy', 
    'Wealth Index'
]

input_data = {}
all_valid = True  # Flag to track if all inputs are valid

# Collect user input and validate
for feature in feature_names:
    value = st.text_input(feature, "0")  # Default value is set to "0"
    if not value.replace('.', '', 1).isdigit():  # Check if the input is not a valid number
        st.warning(f"Please enter a valid numerical value for {feature}.")
        all_valid = False  # Set flag to False if any input is invalid
        input_data[feature] = 0.0  # Assign default value for invalid input
    else:
        input_data[feature] = float(value)  # Convert valid input to float

# Create a DataFrame from input data
if all_valid:  # Proceed only if all inputs are valid
    # Get the feature names from one of the loaded models
    expected_feature_names = loaded_models['Y1'].feature_names

    # Ensure the input data matches the expected feature names
    input_df = pd.DataFrame([input_data], columns=expected_feature_names)

    # Fill missing columns with default values (if any are missing)
    for feature in expected_feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0.0  # Default value for missing features

    # Reorder columns to match the expected feature names
    input_df = input_df[expected_feature_names]

    # Predict button
    if st.button("Predict"):
        predictions = {}
        for target_column in loaded_models:
            prediction = loaded_models[target_column].predict(xgb.DMatrix(input_df))[0]
            predictions[target_column] = prediction
        
        # Find the target variable with the highest prediction value
        predicted_target = max(predictions, key=predictions.get)
        
        # Map the predicted target to a human-readable label
        mapped_prediction = prediction_mapping.get(predicted_target, "Unknown")

        # Display predictions
        # 
        st.write(f"**Predicted target variable: {mapped_prediction}**")
else:st.write("## Predictions:")
    
# st.warning("Please ensure all inputs are numerical before predicting.")






# import streamlit as st
# import pandas as pd
# import xgboost as xgb
# import pickle
# import os

# # Load the models using pickle (Assuming model_dir is where you saved them)
# model_dir = 'xgb_models_pickle'  
# loaded_models = {}
# for target_column in ['Y1', 'Y2', 'Y3', 'Y4']:  # Replace with your actual target columns
#     model_file = os.path.join(model_dir, f'xgb_model_{target_column}.pkl')
#     try:
#         with open(model_file, 'rb') as f:
#             model = pickle.load(f)
#         loaded_models[target_column] = model
#         print(f"Model for {target_column} loaded from: {model_file}")
#     except FileNotFoundError:
#         print(f"Model file for {target_column} not found: {model_file}")





# # Streamlit app
# st.title("XGBoost Model Prediction")

# # Input fields for features (Replace with your actual feature names)
# feature_names = ['number of household members', 'Number of children 5 and under', 'Number of mosquito bed nets', 'Number of children under mosquito bed net previous night', 'Ideal number of children', 'Husband/partner worked in last 7 days/12 months', 'Cost of treatment of fever', 'Number of days after fever began sought advice or treatment', 'duration of pregnancy', 'proportion of children under 5 and below over the total no of household', 'propotion of ideal children under 5 and below over the no. of ideal children in the household', 'levels  burden children under 5 who slept under mosquito net.', 'level of burden associated with proportion of mosquito net per no. of household.', 'level of burden associated with  proportion of children under mosquito net previous night over ideal children.', 'Household has: electricity', 'Result of the malaria test', 'currently pregnant', 'Main floor material', 'Children under 5 slept under mosquito bed net last night (household questionnaire)', 'Type of mosquito bed net(s) slept under last night', 'Type of cooking fuel', 'Wall Material', 'RELATIONSHIP TO HOUSEHOLD HEAD', 'Respondents Currently Working', 'Literacy', 'Wealth Index']
# input_data = {}
# for feature in feature_names:
#     input_data[feature] = st.number_input(feature)

# # Create a DataFrame from input data
# input_df = pd.DataFrame([input_data])

# # Predict button
# if st.button("Predict"):
#     # Predict for each target variable
#     predictions = {}
#     for target_column in loaded_models:
#         prediction = loaded_models[target_column].predict(xgb.DMatrix(input_df))[0]
#         predictions[target_column] = prediction
    
#     # Find the target variable with the highest prediction value
#     predicted_target = max(predictions, key=predictions.get)
    
#     # Display predictions
#     st.write("## Predictions:")
#     # st.write(predictions)
#     st.write(f"**Predicted target variable: {predicted_target}**")




