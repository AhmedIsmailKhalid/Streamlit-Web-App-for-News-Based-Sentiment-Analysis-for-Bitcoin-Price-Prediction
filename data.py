import os
from xml.etree.ElementTree import TreeBuilder
import pandas as pd
from os import listdir
import streamlit as st
from st_aggrid import AgGrid
from process_uploaded_data import process
from feature_engineering import create


def get_data(news_df, btc_df):
    global links, btc
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
    st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

    options = st.radio('', ('Show Default Data', 'Upload Data to Create New Feature Set', 'Show Uploaded Data'))
    
    if options == 'Show Default Data' :
        col1, col2 = st.columns(2) 
        with col1 :
            data_options = st.selectbox('Select Which Data to Display', ['','Processed News Articles Data', 'Bitcoin Hourly Price Data'])

        with col2 :
            theme = st.selectbox('Select Theme for Table', ['streamlit', 'light', 'dark', 'blue', 'fresh', 'material'])
                
        if data_options == 'Processed News Articles Data' :
            # st.markdown('Processed News Articles Data')
            AgGrid(news_df.drop('temp_list', axis=1), theme=theme)

        if data_options == 'Bitcoin Hourly Price Data' :
            # st.markdown('Bitcoin Hourly Price Data')
            AgGrid(btc_df.drop(['Unix Timestamp', 'Symbol'], axis=1), fit_columns_on_grid_load=True, theme=theme)
    
    if options == 'Upload Data to Create New Feature Set' :
        try :
            new_links, new_btc, new_feature_set = create()
        except :
            st.write('')

    if options == 'Show Uploaded Data' :
        col1, col2 = st.columns(2) 
        uploaded_path = os.path.join('.','uploaded data')
        files_dir = [f for f in listdir(uploaded_path) if os.path.isfile(os.path.join(uploaded_path, f))]
        with col1 :
            data_options = st.selectbox('Select File to Display', ['']+files_dir)

        with col2 :
            theme = st.selectbox('Select Theme for Table', ['streamlit', 'light', 'dark', 'blue', 'fresh', 'material'])

        
        for i in range(len(files_dir)) :
            if data_options == files_dir[i] :
                if 'BTC' in files_dir[i] :
                    file = pd.read_csv(os.path.join('uploaded data',files_dir[i]), skiprows=1)
                    AgGrid(file, theme=theme, fit_columns_on_grid_load=True)
                else :
                    file = pd.read_csv(os.path.join('uploaded data',files_dir[i]))
                    AgGrid(file, theme=theme, fit_columns_on_grid_load=True)
            if data_options != files_dir[i] :
                continue