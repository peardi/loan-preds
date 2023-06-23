import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
from datetime import datetime, timedelta
from PIL import Image
import requests
import io

url = "https://app-api-eu-west-3.milesinthesky.education/leaderboard/geordie_history?groupName=Loud Squirrel"

payload = {}
headers = {
  'x-api-key': '4A1ZxPSude5K1HjyaZ5y67Mrzkv4R8KW4OJtyaYH'
}

response = requests.request("GET", url, headers=headers, data=payload)

urlData = response.content
df_csv = pd.read_csv(io.StringIO(urlData.decode('utf-8')))

o = df_csv.dropna(subset=['PREDICTED_TARGET'])


def do_stuff_on_page_load():
    st.set_page_config(layout="wide")
  
# Function to calculate expected profit
def expected_profit(y_true, y_pred, amt_credit):
    TN_indices = np.where((y_true == 0) & (y_pred == 0))
    FN_indices = np.where((y_true == 1) & (y_pred == 0))
    TN_profit = np.sum(0.25 * amt_credit.iloc[TN_indices])
    FN_cost = np.sum(amt_credit.iloc[FN_indices])
    profit = TN_profit - FN_cost
    return profit

do_stuff_on_page_load()

st.header('Live Preds', anchor=None)

# Drop rows with missing values in 'PREDICTED_TARGET'
o = df_csv.dropna(subset=['PREDICTED_TARGET'])

# Calculate the cost for each prediction
o['Cost'] = 0
o.loc[(o['TARGET'] == 0) & (o['PREDICTED_TARGET'] == 1), 'Cost'] = -0.25 * o['AMT_CREDIT']
o.loc[(o['TARGET'] == 1) & (o['PREDICTED_TARGET'] == 0), 'Cost'] = -1 * o['AMT_CREDIT']




# Calculate the average cost for each group of 20 predictions
o['Group'] = (o.index // 20) + 1
grouped_avg_cost = o.groupby('Group')['Cost'].mean().reset_index()

# Calculate the overall average cost
avg_cost = grouped_avg_cost['Cost'].mean()

# Create the scatter plot using Plotly
fig = px.scatter(grouped_avg_cost, x='Group', y='Cost',
                 title='Avg loss',
                 labels={'Group': 'Group of 10', 'Cost': 'Avg Cost'},
                 hover_data=['Group', 'Cost'])

# Select metric option
metric = st.selectbox('Select Metric', ['Cost', 'Profit'])

# Calculate and display the selected metric
if metric == 'Cost':
    st.subheader('Overall Cost')
    st.text(f'{avg_cost:.2f}')
elif metric == 'Profit':
    st.subheader('Overall Profit')
    profit = expected_profit(o['TARGET'], o['PREDICTED_TARGET'], o['AMT_CREDIT'])
    st.text(f'{profit:.2f}')

# Add a line for the average cost
fig.add_shape(type='line', x0=grouped_avg_cost['Group'].min(), y0=metric, x1=grouped_avg_cost['Group'].max(), y1=metric,
              line=dict(color='red', dash='dash'), name='nn')

# Display the scatter plot in Streamlit
st.plotly_chart(fig)

# Display the overall average cost
st.subheader('Overall Average Cost')
st.text(f'{avg_cost:.2f}')

