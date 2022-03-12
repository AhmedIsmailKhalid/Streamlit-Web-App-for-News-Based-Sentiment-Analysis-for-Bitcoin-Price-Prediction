import os
import pickle
from pyexpat import model
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# @st.cache(persist=True, suppress_st_warning=True)
def train():
    data_dir = os.path.join('.','data')
    model_dir = os.path.join('.','models')

    
    bnb = BernoulliNB()
    gnb = GaussianNB()
    knn = KNeighborsClassifier(algorithm='auto', metric='euclidean', n_neighbors=10, weights='distance')
    lr = LogisticRegression(C=0.0001, penalty='l1', solver='liblinear')
    rf = RandomForestClassifier(n_estimators=20, min_samples_split=3, min_samples_leaf=3, max_leaf_nodes=5, max_features='log2', max_depth=None, criterion='entropy')
    svc = SVC(kernel='rbf', gamma='scale', C=0.001)
    mlp = MLPClassifier(solver='adam', learning_rate='adaptive', hidden_layer_sizes=50, alpha=0.1, activation='relu')
    
    df = pd.read_csv(os.path.join(data_dir,'feature_set.csv'))
    
    row1, row2 = st.columns((20,1))
    with row1 :
        placeholder = st.empty()
        placeholder.dataframe(df)
        # AgGrid(df)


    col1,col2 = st.columns(2)
    with col1 :
        models = st.selectbox('Choose Machine Learning Classification Algorithm', ('','Bernoulli Naive Bayes', 'Gaussian Naive Bayes', 'K-Nearest Neighbors', 
                                                                                'Logisitic Regression', 'Random Forest', 'Support Vector Machine', 'Multi-Layer Perceptron'))

    

        if models == '' :
            """`Please choose an algorithm!`"""

    with col2 :
        scaling = st.selectbox('Choose a scaling techinque', ('','No Scaling', 'Standard Scaling', 'Min-Max Scaling', 'Robust Scaling'))

        if scaling == '' :
            placeholder.empty()
            with placeholder.container():
                # st.write('`Feature Set with the chosen scaling will appear here`!')
                st.markdown("<p style='text-align: center; color: orangered;'><b>Feature Set with the chosen scaling will appear here!</b></p>", unsafe_allow_html=True)
            """`Please select scaling technique`"""

    # if models == '' :
    #     """`Please choose an algorithm!`"""

    col3, col4 = st.columns(2)
        

    if scaling == 'No Scaling' :
        placeholder.empty()
        placeholder.dataframe(df)

    if scaling == 'Standard Scaling' :
        scaler = StandardScaler()
        df = df.copy(deep=True)
        df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
        with row1 :
            placeholder.empty()
            placeholder.dataframe(df)
            # AgGrid(df)

    if scaling == 'Min-Max Scaling' :
        scaler = MinMaxScaler()
        df = df.copy(deep=True)
        df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
        with row1 :
            placeholder.empty()
            placeholder.dataframe(df)
            # AgGrid(df)

    if scaling == 'Robust Scaling' :
        scaler = RobustScaler()
        df = df.copy(deep=True)
        df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
        placeholder.empty()
        placeholder.dataframe(df)
        # AgGrid(df)
    
    R1, R2 = st.columns((20, 1))

    if models == 'Bernoulli Naive Bayes' :
        if scaling == '' :
            st.warning('Please select scaling technique')
        
        if scaling != '' :
            X = df.drop('Class', axis=1)
            y = df['Class']
            bnb.fit(X, y)
            bnb_y_pred = bnb.predict(X)
            bnb_acc = accuracy_score(y, bnb_y_pred)
            bnb_f1 = f1_score(y, bnb_y_pred)
            with R1 :
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: green;'><b>Bernoulli Naive Bayes Results</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {bnb_acc}</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {bnb_f1}</b></p>", unsafe_allow_html=True)

            with col3 :
                save_button = st.button('Save Selected Model')

            with col4 :
                delete_button = st.button('Delete Selected Model')

            if save_button :
                filename = models
                st.info('Saving Bernoulli Naive Bayes Model')
                pickle.dump(bnb, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                st.success('Bernoulli Naive Bayes Model saved successfully!')

            if delete_button :
                try :
                    os.remove(os.path.join('.','models', models+'.sav'))
                except :
                    st.error('File not found! Please make sure the model is already saved')

    if models == 'Gaussian Naive Bayes' :
        if scaling == '' :
            st.warning('Please select scaling technique')
        
        if scaling != '' :
            X = df.drop('Class', axis=1)
            y = df['Class']
            gnb.fit(X, y)
            gnb_y_pred = gnb.predict(X)
            gnb_acc = accuracy_score(y, gnb_y_pred)
            gnb_f1 = f1_score(y, gnb_y_pred)
            with R1 :
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: green;'><b>Gaussian Naive Bayes Results</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {gnb_acc}</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {gnb_f1}</b></p>", unsafe_allow_html=True)

            with col3 :
                save_button = st.button('Save Selected Model')

            with col4 :
                delete_button = st.button('Delete Selected Model')

            if save_button :
                filename = models
                st.info('Saving Gaussian Naive Bayes Model')
                pickle.dump(bnb, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                st.success('Gaussian Naive Bayes Model saved successfully!')

            if delete_button :
                try :
                    st.info('Deleting Gaussian Naive Bayes Model')
                    os.remove(os.path.join('.','models', models+'.sav'))
                    st.success('Gaussian Naive Bayes Model deleted successfully!')
                except :
                    st.error('File not found! Please make sure the model is already saved')

    if models == 'K-Nearest Neighbors' :
        if scaling == '' :
            st.warning('Please select scaling technique')
        
        if scaling != '' :
            X = df.drop('Class', axis=1)
            y = df['Class']
            knn.fit(X, y)
            knn_y_pred = knn.predict(X)
            knn_acc = accuracy_score(y, knn_y_pred)
            knn_f1 = f1_score(y, knn_y_pred)
            with R1 :
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: green;'><b>K-Nearest Neighbors Results</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {knn_acc}</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {knn_f1}</b></p>", unsafe_allow_html=True)

            with col3 :
                save_button = st.button('Save Selected Model')

            with col4 :
                delete_button = st.button('Delete Selected Model')

            if save_button :
                filename = models
                st.info('Saving K-Nearest Neighbors Model')
                pickle.dump(bnb, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                st.success('K-Nearest Neighbors Model saved successfully!')

            if delete_button :
                try :
                    st.info('Deleting K-Nearest Neighbors Model')
                    os.remove(os.path.join('.','models', models+'.sav'))
                    st.success('K-Nearest Neighbors Model deleted successfully!')
                except :
                    st.error('File not found! Please make sure the model is already saved')

    if models == 'Logisitic Regression' :
        if scaling == '' :
            st.warning('Please select scaling technique')
        
        if scaling != '' :
            X = df.drop('Class', axis=1)
            y = df['Class']
            lr.fit(X, y)
            lr_y_pred = lr.predict(X)
            lr_acc = accuracy_score(y, lr_y_pred)
            lr_f1 = f1_score(y, lr_y_pred)
            with R1 :
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: green;'><b>Logistic Regression Results</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {lr_acc}</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {lr_f1}</b></p>", unsafe_allow_html=True)

            with col3 :
                save_button = st.button('Save Selected Model')

            with col4 :
                delete_button = st.button('Delete Selected Model')
            
            if save_button :
                filename = models
                st.info('Saving Logistic Regression Model')
                pickle.dump(bnb, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                st.success('Logisitic Regression Model saved successfully!')

            if delete_button :
                try :
                    st.info('Deleting Logistic Regression Model')
                    os.remove(os.path.join('.','models', models+'.sav'))
                    st.success('Logistic Regression Model deleted successfully!')
                except :
                    st.error('File not found! Please make sure the model is already saved')

    if models == 'Random Forest' :
        if scaling == '' :
            st.warning('Please select scaling technique')
        
        if scaling != '' :
            X = df.drop('Class', axis=1)
            y = df['Class']
            rf.fit(X, y)
            rf_y_pred = rf.predict(X)
            rf_acc = accuracy_score(y, rf_y_pred)
            rf_f1 = f1_score(y, rf_y_pred)
            with R1 :
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: green;'><b>Random Forest Results</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {rf_acc}</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {rf_f1}</b></p>", unsafe_allow_html=True)

            with col3 :
                save_button = st.button('Save Selected Model')

            with col4 :
                delete_button = st.button('Delete Selected Model')

            if save_button :
                filename = models
                st.info('Saving Random Forest Model')
                pickle.dump(bnb, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                st.success('Random Forest Model saved successfully!')

            if delete_button :
                try :
                    st.info('Deleting Random Forest Model')
                    os.remove(os.path.join('.','models', models+'.sav'))
                    st.success('Random Forest Model deleted successfully!')
                except :
                    st.error('File not found! Please make sure the model is already saved')

    if models == 'Support Vector Machine' :
        if scaling == '' :
            st.warning('Please select scaling technique')
        
        if scaling != '' :
            X = df.drop('Class', axis=1)
            y = df['Class']
            svc.fit(X, y)
            svc_y_pred = svc.predict(X)
            svc_acc = accuracy_score(y, svc_y_pred)
            svc_f1 = f1_score(y, svc_y_pred)
            with R1 :
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: green;'><b>Support Vector Machine Results</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {svc_acc}</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {svc_f1}</b></p>", unsafe_allow_html=True)

            with col3 :
                save_button = st.button('Save Selected Model')

            with col4 :
                delete_button = st.button('Delete Selected Model')
            
            if save_button :
                filename = models
                st.info('Saving Support Vector Machine Model')
                pickle.dump(bnb, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                st.success('Support Vector Machine Model saved successfully!')

            if delete_button :
                try :
                    st.info('Deleting Support Vector Machine Model')
                    os.remove(os.path.join('.','models', models+'.sav'))
                    st.success('Support Vector Machine Model deleted successfully!')
                except :
                    st.error('File not found! Please make sure the model is already saved')

    if models == 'Multi-Layer Perceptron' :
        if scaling == '' :
            st.warning('Please select scaling technique')
        
        if scaling != '' :
            X = df.drop('Class', axis=1)
            y = df['Class']
            mlp.fit(X, y)
            mlp_y_pred = mlp.predict(X)
            mlp_acc = accuracy_score(y, mlp_y_pred)
            mlp_f1 = f1_score(y, mlp_y_pred)
            with R1 :
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown('<p></p>', unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: green;'><b>Multi-Layer Perceptron Results</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {mlp_acc}</b></p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {mlp_f1}</b></p>", unsafe_allow_html=True)

            with col3 :
                save_button = st.button('Save Selected Model')

            with col4 :
                delete_button = st.button('Delete Selected Model')

            if save_button :
                filename = models
                st.info('Saving Multi-Layer Perceptron Model')
                pickle.dump(bnb, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                st.success('Multi-Layer Perceptron Model saved successfully!')

            if delete_button :
                try :
                    st.info('Deleting Multi-Layer Perceptron Model')
                    os.remove(os.path.join('.','models', models+'.sav'))
                    st.success('Multi-Layer Perceptron Model deleted successfully!')
                except :
                    st.error('File not found! Please make sure the model is already saved')

    
    
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)