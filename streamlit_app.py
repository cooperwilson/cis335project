import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# Page title
st.set_page_config(page_title='Quarterback Statistics 2001-2023')
st.title('Quarterback Statistics 2001-2023')

# Load data
df = pd.read_csv('data/passing_cleaned.csv')
df
