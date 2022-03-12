import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from skopt import BayesSearchCV
from hyperband import HyperbandSearchCV

def tune(feature_set):
    # model = st.selectbox('Choose Machine Learning Classification Algorithm', ('','Bernoulli Naive Bayes', 'Gaussian Naive Bayes', 'K-Nearest Neighbors', 
    #                                                                             'Logisitic Regression', 'Random Forest', 'Support Vector Machine', 'Multi-Layer Perceptron'))

    col1,col2,col3,col4 = st.columns(4)
    with col1 :
        model = st.selectbox('Choose Machine Learning Classification Algorithm', ('','Bernoulli Naive Bayes', 'Gaussian Naive Bayes', 'K-Nearest Neighbors', 
                                                                                'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Multi-Layer Perceptron'))

    with col2 :
        scaling = st.selectbox('Choose a scaling techinque', ('','No Scaling', 'Standard Scaling', 'Min-Max Scaling', 'Robust Scaling'))

    with col3 :
        retrain = st.selectbox('Retrain with the optimal parameters', ('','No', 'Yes'))

    with col4 :
        results = st.selectbox('Show Results (Only if Retrain is Yes)', ('', 'No', 'Yes'))

    
    if scaling == 'Standard Scaling' :
        scaler = StandardScaler()
        feature_set[feature_set.columns[1:]] = scaler.fit_transform(feature_set[feature_set.columns[1:]])
    
    if scaling == 'Min-Max Scaling' :
        scaler = MinMaxScaler()
        feature_set[feature_set.columns[1:]] = scaler.fit_transform(feature_set[feature_set.columns[1:]])

    if scaling == 'Robust Scaling' :
        scaler = RobustScaler()
        feature_set[feature_set.columns[1:]] = scaler.fit_transform(feature_set[feature_set.columns[1:]]) 

    tscv = TimeSeriesSplit(gap=0, n_splits=113)

    
    
    if model == 'Bernoulli Naive Bayes' :
        parameters = {'alpha':np.linspace(0.0,1.0,11), 'binarize':np.linspace(0.0,1.0,11), 'fit_prior':[True, False]}
        params_df = pd.DataFrame({'Alpha': [' , '.join(map(str, np.round(parameters['alpha'], 1)))],    'Binarize': [' , '.join(map(str, np.round(parameters['binarize'], 1)))], 
        'Fir Prior':[' , '.join(map(str, parameters['fit_prior']))]})

        
        st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Bernoulli Naive Bayes</p>", unsafe_allow_html=True)
        st.dataframe(params_df)

        if scaling == '' :
            st.warning('Please select scaling technique!')

        if retrain == '' :
            st.warning('Please select to retrain model with optimal parameters or not')

        if retrain == 'No' :
            retrain = False
        
        if retrain == 'Yes' :
            retrain = True

        if (scaling != '') and (retrain != ''):

            col_5, col_6, col_7 = st.columns((3.3,3,1))

            with col_5 :
                st.write('' )


            with col_6 :
                button = st.button('Tune parameters!')

            with col_7 :
                st.write('')

            if button :
                
                st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')

                start = datetime.now()

                clf = GridSearchCV(estimator=BernoulliNB(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                
                clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                end = datetime.now()

                print("Tune Fit Time:", end - start)

                st.success('Hyperparameter Tuning Done')
                st.write("Tuning Time:", end - start)

                st.write('Best parameters for Bernoulli Naive Bayes :\n', clf.best_params_)

                col8, col9 = st.columns((20, 1))

                y_pred = clf.predict(feature_set.drop('Class',axis=1))
                # st.write(y_pred)

                acc = accuracy_score(feature_set['Class'], y_pred)
                f1 = f1_score(feature_set['Class'], y_pred)

                with col8 :
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; color: green;'><b>Bernoulli Naive Bayes Results</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)

    if model == 'Gaussian Naive Bayes' :
        parameters = {'var_smoothing':np.linspace(1e-9,1.0,10)}
        params_df = pd.DataFrame({'Variable Smoothing': [' , '.join(map(str, parameters['var_smoothing']))]})

        
        st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Gaussian Naive Bayes</p>", unsafe_allow_html=True)
        st.dataframe(params_df)

        if scaling == '' :
            st.warning('Please select scaling technique!')

        if retrain == '' :
            st.warning('Please select to retrain model with optimal parameters or not')

        if retrain == 'No' :
            retrain = False
        
        if retrain == 'Yes' :
            retrain = True

        if (scaling != '') and (retrain != ''):

            col_5, col_6, col_7 = st.columns((3.3,3,1))

            with col_5 :
                st.write('' )


            with col_6 :
                button = st.button('Tune parameters!')

            with col_7 :
                st.write('')

            if button :
                
                st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')

                start = datetime.now()

                clf = GridSearchCV(estimator=GaussianNB(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                
                clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                end = datetime.now()

                print("Tune Fit Time:", end - start)

                st.success('Hyperparameter Tuning Done')
                st.write("Tuning Time:", end - start)

                st.write('Best parameters for Gaussian Naive Bayes :\n', clf.best_params_)

                col8, col9 = st.columns((20, 1))

                y_pred = clf.predict(feature_set.drop('Class',axis=1))
                # st.write(y_pred)

                acc = accuracy_score(feature_set['Class'], y_pred)
                f1 = f1_score(feature_set['Class'], y_pred)

                with col8 :
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; color: green;'><b>Gaussian Naive Bayes Results</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)

    if model == 'K-Nearest Neighbors' :
        parameters = {'n_neighbors':range(1,11), 'weights':['uniform', 'distance'], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'], 'metric':['euclidean', 'manhattan', 'minkowski']}
        params_df = pd.DataFrame({'No. of Neighbors': [' , '.join(map(str, parameters['n_neighbors']))],   'Weights': [' , '.join(map(str, parameters['weights']))],
                                  'Algorithm': [' , '.join(map(str, parameters['algorithm']))]})
        
        st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for K-Nearest Neighbors</p>", unsafe_allow_html=True)
        st.dataframe(params_df)

        if scaling == '' :
            st.warning('Please select scaling technique!')

        if retrain == '' :
            st.warning('Please select to retrain model with optimal parameters or not')

        if retrain == 'No' :
            retrain = False
        
        if retrain == 'Yes' :
            retrain = True

        if (scaling != '') and (retrain != ''):

            col_5, col_6, col_7 = st.columns((3.3,3,1))

            with col_5 :
                st.write('' )


            with col_6 :
                knn_button = st.button('Tune parameters!')

            with col_7 :
                st.write('')

            if knn_button :
                
                st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')

                start = datetime.now()

                clf = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                
                clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                end = datetime.now()

                print("Tune Fit Time:", end - start)

                st.success('Hyperparameter Tuning Done')
                st.write("Tuning Time:", end - start)

                st.write('Best parameters for K-Nearest Neighbors :\n', clf.best_params_)

                col8, col9 = st.columns((20, 1))

                y_pred = clf.predict(feature_set.drop('Class',axis=1))
                # st.write(y_pred)

                acc = accuracy_score(feature_set['Class'], y_pred)
                f1 = f1_score(feature_set['Class'], y_pred)

                with col8 :
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; color: green;'><b>K-Nearest Neighbors Results</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)

    if model == 'Logistic Regression' :
        parameters = {'penalty':['l1','l2','elasticnet'], 'C':[0.0001,0.001,0.01, 0.1,1.0,10],'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        params_df = pd.DataFrame({'Penalty': [' , '.join(map(str, parameters['penalty']))],   'C': [' , '.join(map(str, parameters['C']))],
                                  'Solver': [' , '.join(map(str, parameters['solver']))]})
        
        st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Logistic Regression</p>", unsafe_allow_html=True)
        st.dataframe(params_df)

        if scaling == '' :
            st.warning('Please select scaling technique!')

        if retrain == '' :
            st.warning('Please select to retrain model with optimal parameters or not')

        if retrain == 'No' :
            retrain = False
        
        if retrain == 'Yes' :
            retrain = True

        if (scaling != '') and (retrain != ''):

            col_5, col_6, col_7 = st.columns((3.3,3,1))

            with col_5 :
                st.write('' )


            with col_6 :
                button = st.button('Tune parameters!')

            with col_7 :
                st.write('')

            if button :
                
                st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')

                start = datetime.now()

                clf = GridSearchCV(estimator=LogisticRegression(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                
                clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                end = datetime.now()

                print("Tune Fit Time:", end - start)

                st.success('Hyperparameter Tuning Done')
                st.write("Tuning Time:", end - start)

                st.write('Best parameters for Logistic Regression :\n', clf.best_params_)

                col8, col9 = st.columns((20, 1))

                y_pred = clf.predict(feature_set.drop('Class',axis=1))
                # st.write(y_pred)

                acc = accuracy_score(feature_set['Class'], y_pred)
                f1 = f1_score(feature_set['Class'], y_pred)

                with col8 :
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; color: green;'><b>Logistic Regression Results</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
    
    if model == 'Random Forest' :
        parameters = {'n_estimators':[10,20,30,40,50], 'criterion':['gini', 'entropy'],
              'max_depth':[8,16,32,64,128,256,512,1024,None], 'min_samples_split':range(1,6),
              'min_samples_leaf':range(1,6),
              'max_leaf_nodes':[2,3,4,5,None],'max_features':['auto', 'sqrt', 'log2']}

        params_df = pd.DataFrame({'No of Estimators': [' , '.join(map(str, parameters['n_estimators']))],   'Criterion': [' , '.join(map(str, parameters['criterion']))],
                                  'Maximum Depth': [' , '.join(map(str, parameters['max_depth']))], 'Minimum Samples Split': [' , '.join(map(str, parameters['min_samples_split']))], 
                                  'Minimum Sample Leafs': [' , '.join(map(str, parameters['min_samples_leaf']))], 'Maximum Leaf Nodes': [' , '.join(map(str, parameters['max_leaf_nodes']))], 
                                  'Maximum Features': [' , '.join(map(str, parameters['max_features']))]})
        
        st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Random Forest</p>", unsafe_allow_html=True)
        st.dataframe(params_df)

        if scaling == '' :
            st.warning('Please select scaling technique!')

        if retrain == '' :
            st.warning('Please select to retrain model with optimal parameters or not')

        if retrain == 'No' :
            retrain = False
        
        if retrain == 'Yes' :
            retrain = True

        if (scaling != '') and (retrain != ''):

            col_5, col_6, col_7 = st.columns((3.3,3,1))

            with col_5 :
                st.write('' )


            with col_6 :
                button = st.button('Tune parameters!')

            with col_7 :
                st.write('')

            if button :
                
                st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')

                start = datetime.now()

                clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                
                clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                end = datetime.now()

                print("Tune Fit Time:", end - start)

                st.success('Hyperparameter Tuning Done')
                st.write("Tuning Time:", end - start)

                st.write('Best parameters for Random Forest :\n', clf.best_params_)

                col8, col9 = st.columns((20, 1))

                y_pred = clf.predict(feature_set.drop('Class',axis=1))
                # st.write(y_pred)

                acc = accuracy_score(feature_set['Class'], y_pred)
                f1 = f1_score(feature_set['Class'], y_pred)

                with col8 :
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; color: green;'><b>Random Forest Results</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)

    if model == 'Support Vector Machine' :
        parameters = {'kernel':['linear','poly','rbf','sigmoid','precomputed'], 'C':[0.001,0.01,0.1],
              'gamma':['scale','auto']} # add coef0, shrinking, probability, tol, decision_function_shape

        params_df = pd.DataFrame({'Kernel': [' , '.join(map(str, parameters['kernel']))],   'C': [' , '.join(map(str, parameters['C']))],
                                  'Gamma': [' , '.join(map(str, parameters['gamma']))]})
        
        st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Support Vector Machine</p>", unsafe_allow_html=True)
        st.dataframe(params_df)

        if scaling == '' :
            st.warning('Please select scaling technique!')

        if retrain == '' :
            st.warning('Please select to retrain model with optimal parameters or not')

        if retrain == 'No' :
            retrain = False
        
        if retrain == 'Yes' :
            retrain = True

        if (scaling != '') and (retrain != ''):

            col_5, col_6, col_7 = st.columns((3.3,3,1))

            with col_5 :
                st.write('' )


            with col_6 :
                button = st.button('Tune parameters!')

            with col_7 :
                st.write('')

            if button :
                
                st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')

                start = datetime.now()

                clf = GridSearchCV(estimator=SVC(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                
                clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                end = datetime.now()

                print("Tune Fit Time:", end - start)

                st.success('Hyperparameter Tuning Done')
                st.write("Tuning Time:", end - start)

                st.write('Best parameters for Support Vector Machine :\n', clf.best_params_)

                col8, col9 = st.columns((20, 1))

                y_pred = clf.predict(feature_set.drop('Class',axis=1))
                # st.write(y_pred)

                acc = accuracy_score(feature_set['Class'], y_pred)
                f1 = f1_score(feature_set['Class'], y_pred)

                with col8 :
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; color: green;'><b>Support Vector Machine Results</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)

    if model == 'Multi-Layer Perceptron' :
        parameters = {'hidden_layer_sizes':[10,20,50,100],
              'alpha':[0.001,0.01, 0.1], 'activation':['identity','logistic','tanh','relu'],
             'solver':['lbfgs','sgd','adam'], 'learning_rate':['constant','invscaling','adaptive']} # add batch_size, alpha, early_stopping

        params_df = pd.DataFrame({'Hidden Layer Sizes': [' , '.join(map(str, parameters['hidden_layer_sizes']))],   'Alpha': [' , '.join(map(str, parameters['alpha']))],
                                  'Solver': [' , '.join(map(str, parameters['solver']))], 'Learning Rate': [' , '.join(map(str, parameters['learning_rate']))]})
        
        st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Support Vector Machine</p>", unsafe_allow_html=True)
        st.dataframe(params_df)

        if scaling == '' :
            st.warning('Please select scaling technique!')

        if retrain == '' :
            st.warning('Please select to retrain model with optimal parameters or not')

        if retrain == 'No' :
            retrain = False
        
        if retrain == 'Yes' :
            retrain = True

        if (scaling != '') and (retrain != ''):

            col_5, col_6, col_7 = st.columns((3.3,3,1))

            with col_5 :
                st.write('' )


            with col_6 :
                button = st.button('Tune parameters!')

            with col_7 :
                st.write('')

            if button :
                
                st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')

                start = datetime.now()

                clf = GridSearchCV(estimator=MLPClassifier(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                
                clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                end = datetime.now()

                print("Tune Fit Time:", end - start)

                st.success('Hyperparameter Tuning Done')
                st.write("Tuning Time:", end - start)

                st.write('Best parameters for Multi-Layer Perceptron :\n', clf.best_params_)

                col8, col9 = st.columns((20, 1))

                y_pred = clf.predict(feature_set.drop('Class',axis=1))
                # st.write(y_pred)

                acc = accuracy_score(feature_set['Class'], y_pred)
                f1 = f1_score(feature_set['Class'], y_pred)

                with col8 :
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    # st.markdown('<p></p>', unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; color: green;'><b>Multi-Layer Perceptron</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)