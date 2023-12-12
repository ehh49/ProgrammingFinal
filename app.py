import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


s = pd.read_csv('social_media_usage.csv')
print(f'The dimensions of the dataframe are: {s.shape}')

def clean_sm(x):
    if x == 1:
        return 1
    else:
        return 0
    


# Create the new DataFrame
ss = s[['income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()
ss['sm_li'] = s['web1h'].apply(clean_sm)

# Filter the DataFrame based on the conditions
ss = ss[(ss['income'] <= 9) & (ss['educ2'] <= 8) & (ss['par'] <= 2) & ((ss['marital'] == 1) | (ss['marital'] == 6)) & (ss['age'] <= 98)].dropna()

y = ss['sm_li']
X = ss.drop('sm_li', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)



#format to make streamlit look pretty


st.title('LinkedIn Usage Predictor')
#Who Uses LinkedIn?
## By: Emily Hendrickson
## Building and Deploying a Machine Learning App in Python to Predict LinkedIn Users
print('-' * 50)
##### Sometimes it seems like the whole world uses LinkedIn. Of course, while it is a popular social networking site and useful for marketing purposes, not everyone uses it. In this project, you are working with the marketing analytics team of your organization and have been tasked with evaluating options for promoting the business on different mediums. Your CEO would like you to analyze data on social media habits among the US public and build a model that takes predicts social media usage--in this case whether someone uses LinkedIn--as a function of individual attributes and demographics. The goal is for the marketing team to use what you create to examine options and platforms to target for marketing campaigns and potential segments of customers. The application must be interactive, publicly hosted, and use machine learning to produce predictions in real-time.

# Define a dictionary that maps numbers to income ranges
income_options = {
    1: "1-Less than $10,000",
    2: "2-10 to under $20,000",
    3: "3-20 to under $30,000",
    4: "4-30 to under $40,000",
    5: "5-40 to under $50,000",
    6: "6-50 to under $75,000",
    7: "7-75 to under $100,000",
    8: "8-100 to under $150,000",
    9: "9-$150,000 or more"
}

# Collect user input
income_range = st.selectbox('Income Range', options=list(income_options.values()))

# Extract the number from the selected income range
income_range_encoded = int(income_range.split('-')[0])

# Define a dictionary that maps numbers to education levels
education_options = {
    1: "1-Less than high school (Grades 1-8 or no formal schooling)",
    2: "2-High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
    3: "3-High school graduate (Grade 12 with diploma or GED certificate)",
    4: "4-Some college, no degree (includes some community college)",
    5: "5-Two-year associate degree from a college or university",
    6: "6-Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
    7: "7-Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
    8: "8-Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"
}

# Collect user input
education_level = st.selectbox('Education Level', options=list(education_options.values()))

# Extract the number from the selected education level
education_level_encoded = int(education_level.split('-')[0])


# Collecting user inputs
income_range = st.selectbox('Income Range', options=[1, 2, 3, 4, 5, 6, 7, 8, 9])
education_level = st.selectbox('Education Level', options=[1, 2, 3, 4, 5, 6, 7, 8])
parental_status = st.selectbox('Parental Status', options=['Yes', 'No'])
marriage_status = st.selectbox('Marriage Status', options=['Yes', 'No'])
female = st.selectbox('Female', options=['Yes', 'No'])
age = st.slider('Age', min_value=18, max_value=100)

# Encoding binary variables
parental_status_encoded = 1 if parental_status == 'Yes' else 0
marriage_status_encoded = 1 if marriage_status == 'Yes' else 0
female_encoded = 1 if female == 'Yes' else 0

# Creating a feature vector
features = [income_range, education_level, parental_status_encoded, marriage_status_encoded, female_encoded, age]

#Predictions
if st.button('Predict'):
    prediction = model.predict([features])
    if prediction[0] == 1:
        st.write('The person is likely to use LinkedIn.')
    else:
        st.write('The person is unlikely to use LinkedIn.')





























print(f'Model Accuracy: {model.score(X_test, y_test)}')




