import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = 'Salary_prediction.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load dataset to get unique values for categorical columns
data_path = 'Salary_Dataset_with_Extra_Features.csv'
df = pd.read_csv(data_path)

# Streamlit app title
st.title("Salary Prediction App")

# Collect user input
rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

company_name = st.selectbox("Company Name", df["Company Name"].unique())
job_title = st.selectbox("Job Title", df["Job Title"].unique())
salaries_reported = st.number_input("Salaries Reported", min_value=1, max_value=361, value=1, step=1)
location = st.selectbox("Location", df["Location"].unique())
employment_status = st.selectbox("Employment Status", df["Employment Status"].unique())
job_roles = st.selectbox("Job Roles", df["Job Roles"].unique())

# Convert categorical input into numerical values (encoding)
company_value = list(df["Company Name"].unique()).index(company_name)
job_title_value = list(df["Job Title"].unique()).index(job_title)
location_value = list(df["Location"].unique()).index(location)
employment_status_value = list(df["Employment Status"].unique()).index(employment_status)
job_roles_value = list(df["Job Roles"].unique()).index(job_roles)

# Add placeholder values for missing features (experience, education_value)
features = np.array([[rating, salaries_reported, company_value, job_title_value, location_value, 0, 1]])

# Predict salary
if st.button("Predict Salary"):
    predicted_salary = model.predict(features)[0]
    st.success(f"Predicted Salary: {predicted_salary:,.2f}")
