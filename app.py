import streamlit as st
st.set_page_config(layout="wide")
st.markdown("<h3 style='color: black; font-family: Times New Roman;'>Ticket Analysis Dashboard </h3>", unsafe_allow_html=True)
import random
Ticket_agent = ['Rohan', 'Karthik', 'Rishabh', 'Tejaswar', 'Binod', 'Advaita', 'Gaurav', 'Rehber', 'Anuranan', 'Jathin']
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

@st.cache_data
def load_data(file):
    data=pd.read_csv(file)
    return data

uploaded_file = st.file_uploader("Choose a file:")

if uploaded_file is None:
    st.info("Upload a file through config",icon="ℹ️")
    st.stop()

df=load_data(uploaded_file)



#st.markdown("<h3 style='color: black; font-family: Times New Roman;'>Sampled Customer Ratings Over Time</h3>", unsafe_allow_html=True)

import matplotlib.font_manager as fm

# Specify the Times New Roman font
times_new_roman_font = fm.FontProperties(fname=plt.matplotlib.get_data_path() + "/fonts/ttf/times.ttf")

col1,col2,col3=st.columns([0.38,0.38,0.24])

with col1:
    priority_counts = df['Ticket Priority'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%', startangle=90, colors=['yellow', 'red', 'orange', 'green'],
        wedgeprops=dict(width=0.2), textprops={'fontsize': 8, 'fontname': 'Times New Roman'})
    ax1.set_title('Ticket Priority', fontsize=10, fontname='Times New Roman')
    col1.pyplot(fig1)

with col2:
    channel_counts = df['Ticket Channel'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.pie(channel_counts, labels=channel_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors,
        wedgeprops=dict(width=0.2), textprops={'fontsize': 8, 'fontname': 'Times New Roman'})
    ax2.set_title('Ticket Channel', fontsize=10, fontname='Times New Roman')
    col2.pyplot(fig2)

df['Ticket Agent'] = [random.choice(Ticket_agent) for _ in range(len(df))]
df['First Response Time'] = pd.to_datetime(df['First Response Time'])
df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'])
# Calculate the time taken to resolve
df['Time taken to resolve'] = abs((df['Time to Resolution']-df['First Response Time']).dt.total_seconds() / 3600)
average_time_to_resolve = df['Time taken to resolve'].mean()
average_rating = df['Customer Satisfaction Rating'].mean()
total_tickets = len(df)
closed_tickets = df['Time to Resolution'].count()
open_tickets=total_tickets-closed_tickets

with col3:
    st.markdown("<h3 style='color: black;font-weight: normal;font-family: Times New Roman;'>Average time to Resolve </h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:35px;font-weight: bold;font-family: Times New Roman;'>{average_time_to_resolve:.2f} hours</p>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: black;font-weight: normal;font-family: Times New Roman;'>Average Rating </h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:35px;font-weight: bold;font-family: Times New Roman;'>{average_rating:.2f} Stars</p>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: black;font-weight: normal;font-family: Times New Roman;'>Open Tickets</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:35px;font-weight: bold;font-family: Times New Roman;'>{open_tickets:} </p>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: black;font-weight: normal;font-family: Times New Roman;'>Tickets Closed</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:35px;font-weight: bold;font-family: Times New Roman;'>{closed_tickets:} </p>", unsafe_allow_html=True)

# Generate sample data for rating and ticket agent distribution
rating_data = {'Date': pd.date_range(start='2022-01-01', periods=len(df)), 'Rating': [random.randint(1, 5) for _ in range(len(df))]}
rating_df = pd.DataFrame(rating_data)

# Sample a subset of data for plotting
sampled_df = rating_df.sample(n=100)  # Adjust the number based on your preference

c1, c2, c3 = st.columns([0.33, 0.33, 0.33])

# Create a line graph for sampled customer ratings
with c1:
    st.markdown("<h3 style='text-align: center;'>Customer Ratings Over Time</h3>", unsafe_allow_html=True)
    fig_rating = plt.figure(figsize=(5,4))
    sns.lineplot(x='Date', y='Rating', data=sampled_df)
    st.pyplot(fig_rating)
    

# Create a bar chart for ticket agent distribution with different colors
with c3:
    st.markdown("<h3 style='text-align: center;'>Ticket Agents Distribution</h3>", unsafe_allow_html=True)
    fig_agents = plt.figure(figsize=(5, 4))
    agent_counts = df['Ticket Agent'].value_counts()
    agent_colors = sns.color_palette("husl", n_colors=len(agent_counts))
    sns.barplot(x=agent_counts.values, y=agent_counts.index, palette=agent_colors)  # Swap x and y
    plt.xticks(rotation=0, ha="center")  # Set rotation to 0 for horizontal labels
    st.pyplot(fig_agents)

# Display the first few rows of the DataFrame
with c2:
    age_data = df['Customer Age']
    st.markdown("<h3 style='text-align: center;'>Age Distribution</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(age_data, bins=20, color='skyblue', edgecolor='black')
    avg_age = age_data.mean()
    ax.axvline(avg_age, color='red', linestyle='dashed', linewidth=2, label=f'Average Age: {avg_age:.2f}')
    ax.set_xlabel('Customer Age')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)
    

column1,column2=st.columns([0.6,0.4])
random_df = pd.DataFrame(
    np.random.randn(100,2) / [50, 50] + [40.7355836, -73.9815983],
    columns=['lat', 'lon'])
with column1:
    st.map(random_df)
with column2:
    st.dataframe(df.head(10))

    
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import os
from getpass import getpass
import pandas as pd

# Get OpenAI API key :- os.environ['OPENAI_API_KEY'] = getpass('Enter your OpenAI API key')

api_key='ENTER YOUR API KEY HERE'
os.environ['OPENAI_API_KEY'] = api_key

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS
)

with column2:
    custom_question = st.text_input("**Ask a question:**")
    if custom_question:
        response = agent.invoke(custom_question)
        st.write("Agent's response:")
        st.write(response['output'])  # Assuming the response is a dictionary with 'output' key
    else:
        st.write("Waiting for a question...")




