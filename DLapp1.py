import os
import streamlit as st
import pandas as pd
import numpy as np

def main() :
    st.title('DataFrame Creation for Prediction')
    st.markdown('Creating a dataframe fro userinput for prediction')

    data_dir = os.path.join('.','data')
    model_dir = os.path.join('.','models')

    # unless function or arguments change, cache outputs to disk and use cached outputs anytime app re-runs
    @st.cache(persist=True)
    def load_data() :
        train_data = pd.read_csv(os.path.join(data_dir,'wdbc train.csv'))	#pd.read_csv(train_path)
        
        return train_data

    
    df = load_data()
    columns = []
    for col in df.columns :
        columns.append(col)
    
    #meanradius = st.sidebar.number_input('Mean Radius', key = 'mean radius')
    #meanarea = st.sidebar.number_input('Mean Area', key = 'mean area')
    meansmoothness = st.sidebar.number_input('Mean Smoothness', key = 'mean smoothness')
    meancompactness = st.sidebar.number_input('Mean Compactness', key = 'mean compactness')
    meanconcavity = st.sidebar.number_input('Mean Concavity', key = 'mean concavity')
    meanconcavepoints = st.sidebar.number_input('Mean Concavepoints', key = 'mean concavepoints')
    meansymmetry = st.sidebar.number_input('Mean Symmetry', key = 'mean symmetry')
    meanfractaldimension = st.sidebar.number_input('Mean Fractal Dimension', key = 'mean fractal dimension')

    query_data = [{'meanradius' : meanradius, 'meanarea' : meanarea, 'meansmoothness' : meansmoothness, 'meanconcavepoints' : meanconcavepoints, 
            'meanconcavepoints' : meanconcavepoints, 'meansymmetry' : meansymmetry, 'meanfractaldimension' : meanfractaldimension}]

    features_list = ['meanradius', 'meanarea']

    features_select = st.multiselect('Select Features', features_list, key='features select')

    if features_select == 'meanradius' :
        meanradius = st.sidebar.number_input('Mean Radius', key = 'mean radius')

    elif features_select == 'meanradius' :
        meanarea = st.sidebar.number_input('Mean Area', key = 'mean area')
    
    query_df = pd.DataFrame(data = query_data)
    st.table(query_df)

if __name__ == '__main__' :
    main()