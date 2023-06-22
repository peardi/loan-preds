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

do_stuff_on_page_load()

st.header('Live Preds', anchor=None)

#Import Data
df = st.session_state['df']
# Calculate the cost for each prediction
o = o.copy()  # Create a copy of the DataFrame to suppress the warning

# Calculate the cost for each prediction
o['Cost'] = 0  # Initialize the 'Cost' column
o.loc[(o['TARGET'] == 0) & (o['PREDICTED_TARGET'] == 1), 'Cost'] = -0.25 * o['AMT_CREDIT']
o.loc[(o['TARGET'] == 1) & (o['PREDICTED_TARGET'] == 0), 'Cost'] = -1 * o['AMT_CREDIT']

# Calculate the average cost for each group of 20 predictions
o['Group'] = (o.index // 20) + 1
grouped_avg_cost = o.groupby('Group')['Cost'].mean().reset_index()

# Calculate the overall average cost
avg_cost = grouped_avg_cost['Cost'].mean()

# Create the scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(grouped_avg_cost['Group'].astype(int), grouped_avg_cost['Cost'])
plt.xlabel('Group')
plt.ylabel('Average Cost')

# Add a line for the average cost
plt.axhline(y=avg_cost, color='r', linestyle='--', label='Average Cost')

# Display the plot
plt.legend()
plt.tight_layout()

# Save the plot to a temporary file
plot_path = 'temp_plot.png'
plt.savefig(plot_path)
plt.close()

# Display the scatter plot in Streamlit
st.title('Average Cost by Group')
st.text('Scatter Plot of Average Cost for Each Group of 20 Predictions')
st.image(plot_path)

# Display the overall average cost
st.subheader('Overall Average Cost')
st.text(f'{avg_cost:.2f}')

# Remove the temporary plot file
import os
os.remove(plot_path)
