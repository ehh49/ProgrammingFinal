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


st.markdown





























print(f'Model Accuracy: {model.score(X_test, y_test)}')
print(f'Confusion Matrix: \n{cm}')



