import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go


#🛑 Code to set the Dashboard format to wide (the content will fill the entire width of the page instead of having wide margins)
def do_stuff_on_page_load():
    st.set_page_config(layout="wide")
do_stuff_on_page_load()

#Set Header
#🛑 Code to set the header
st.header('Feature analysis', anchor=None)

col1, col2,col3,col4= st.columns(4)

col1.metric("Requests",6396208)
default_rate = ("0.31 %")
col2.metric("Default Rate",default_rate)
target_repay =("91,9%")
col3.metric("TARGET Repay",target_repay,"+")
target_default =("8.1%")
col4.metric("TARGET Default",target_default,"-")            

#🛑 Code to import the dataset
df = pd.read_csv('https://miles-become-a-data-scientist.s3.us-east-2.amazonaws.com/J3/M3/data/train.csv')

#🛑 Code to persist the DataFrame between pages of the same Dashboard. Without this, any other page would need to re import the DataFrame and save it to df again.
st.session_state['df'] = df 

# Group the data by multiple columns and calculate the count
grouped_data = df.groupby(['OCCUPATION_TYPE',
        'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE',
        'NAME_INCOME_TYPE','TARGET']).size().reset_index(name='Count')

default_count = df[df['TARGET'] == 1].shape[0]
repay_count = df[df['TARGET'] == 0].shape[0]

with st.expander("Explorer Sunburst", expanded=True):

    # Create the sunburst chart using Plotly Express
    fig = px.sunburst(grouped_data, path=['OCCUPATION_TYPE',
        'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE',
        'NAME_INCOME_TYPE',
        'TARGET'],height = 1900, values='Count',
                      title='Sunburst Chart of Education, Occupation and Target')
    
    st.plotly_chart(fig,use_container_width=True)
    

with st.expander("Income Type Distribution"):
    
    income_counts = df['NAME_INCOME_TYPE'].value_counts()
    fig2 = px.bar(income_counts, x=income_counts.index, y=income_counts.values,
              labels={'x': 'Income Type', 'y': 'Count'}, title='Income Type Distribution')
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2,use_container_width=True)
    
with st.expander("Gender Distribution"):    
    gender_counts = df['CODE_GENDER'].value_counts()
    fig3 = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index,
                  title='Gender Distribution')
    st.plotly_chart(fig3,use_container_width=True)
