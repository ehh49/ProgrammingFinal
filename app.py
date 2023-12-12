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

#Predictions
if st.button('Predict'):
    prediction = model.predict([features])
    if prediction[0] == 1:
        st.write('The person is likely to use LinkedIn.')
    else:
        st.write('The person is unlikely to use LinkedIn.')





























print(f'Model Accuracy: {model.score(X_test, y_test)}')




