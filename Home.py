import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder,RobustScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier



#🛑 Code to set the Dashboard format to wide (the content will fill the entire width of the page instead of having wide margins)
def do_stuff_on_page_load():
    st.set_page_config(layout="wide")
do_stuff_on_page_load()

#Set Header
#🛑 Code to set the header
st.header('Model metrics v1', anchor=None)



#🛑 Code to import the dataset
df = pd.read_csv('https://miles-become-a-data-scientist.s3.us-east-2.amazonaws.com/J3/M3/data/train.csv')

st.session_state['df'] = df 
#🛑 Code to persist the DataFrame between pages of the same Dashboard. Without this, any other page would need to re import the DataFrame and save it to df again.
#st.session_state['df'] = df 


# Group the data by multiple columns and calculate the count
grouped_data = df.groupby(['OCCUPATION_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'NAME_INCOME_TYPE','TARGET']).size().reset_index(name='Count')





# Create the sunburst chart using Plotly Express
fig = px.sunburst(grouped_data, path=['OCCUPATION_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'NAME_INCOME_TYPE',
    'TARGET'],height = 1900, values='Count',
                  title='Sunburst Chart of Education, Occupation, and Target')

# Show the plot
fig.show()



