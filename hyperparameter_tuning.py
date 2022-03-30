import os
import time
import pickle
from datetime import datetime
from unittest import result
import pandas as pd
import numpy as np
import streamlit as st
from stqdm import stqdm
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, f1_score
from skopt import BayesSearchCV

def tune(feature_set):
    # model = st.selectbox('Choose Machine Learning Classification Algorithm', ('','Bernoulli Naive Bayes', 'Gaussian Naive Bayes', 'K-Nearest Neighbors', 
    #                                                                             'Logisitic Regression', 'Random Forest', 'Support Vector Machine', 'Multi-Layer Perceptron'))

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
    st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

    # train_options = st.radio('Training Options', ('Use default feature set', 'Upload new feature set', 'Create new feature set', 'Perform Hyperparameter Tuning'))
    tune_options = st.radio('Tuning Options', ('Use provided hyperparameter search space', 'Create and use a new hyperparameter search space'))

    jumps = 1

    tscv = TimeSeriesSplit(gap=0, n_splits=113)

    if tune_options == 'Use provided hyperparameter search space' :
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1 :
            model = st.selectbox('Choose Machine Learning Classification Algorithm', ('','Bernoulli Naive Bayes', 'Gaussian Naive Bayes', 'K-Nearest Neighbors', 
                                                                                    'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Multi-Layer Perceptron'))

        with col2 :
            scaling = st.selectbox('Choose a scaling techinque', ('','No Scaling', 'Standard Scaling', 'Min-Max Scaling', 'Robust Scaling'))

        with col3 :
            retrain = st.selectbox('Retrain with the optimal parameters', ('','No', 'Yes'))

        with col4 :
            strategy = st.selectbox('Choose an Optimization Strategy', ('', 'Grid Search CV', 'Random Search CV', 'Halving Grid Search CV')) #, 'Bayesian Optimization'))
        
        with col5 :
            results = st.selectbox('Show Results (Will Show Only if Retrain is Yes)', ('', 'No', 'Yes'))
        
        if scaling == 'Standard Scaling' :
            scaler = StandardScaler()
            feature_set[feature_set.columns[1:]] = scaler.fit_transform(feature_set[feature_set.columns[1:]])
        
        if scaling == 'Min-Max Scaling' :
            scaler = MinMaxScaler()
            feature_set[feature_set.columns[1:]] = scaler.fit_transform(feature_set[feature_set.columns[1:]])

        if scaling == 'Robust Scaling' :
            scaler = RobustScaler()
            feature_set[feature_set.columns[1:]] = scaler.fit_transform(feature_set[feature_set.columns[1:]]) 

        
        
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

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                col_5, col_6, col_7 = st.columns((3.3,3,1))
 
                with col_5 :
                    st.write('' )

                with col_6 :
                    button = st.button('Tune parameters!')

                with col_7 :
                    st.write('')

                if button :

                    if strategy == 'Grid Search CV' :
                        clf = GridSearchCV(estimator=BernoulliNB(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                    if strategy == 'Random Search CV' :
                        clf = RandomizedSearchCV(estimator=BernoulliNB(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    
                    if strategy == 'Halving Grid Search CV' :
                        clf = HalvingGridSearchCV(estimator=BernoulliNB(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                    # if strategy == 'Bayesian Optimization' :
                    #     clf = BayesSearchCV(estimator=BernoulliNB(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    

                    placeholder = st.empty()
                    with placeholder.container():
                        st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                        grid = ParameterGrid(parameters)
                        st.write (f"The total number of parameters-combinations is: {len(grid)}")
                        st.write (f"The total number of iterations is: {len(grid)*113}")

                    start = datetime.now()

                    clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                    end = datetime.now()

                    placeholder.empty()

                    print("Tune Fit Time:", end - start)

                    st.success('Hyperparameter Tuning Done')
                    st.write("Tuning Time:", end - start)

                    st.write('Best parameters for Bernoulli Naive Bayes :\n', clf.best_params_)

                    col8, col9 = st.columns((20, 1))

                    try :
                        y_pred = clf.predict(feature_set.drop('Class',axis=1))

                        acc = accuracy_score(feature_set['Class'], y_pred)
                        f1 = f1_score(feature_set['Class'], y_pred)

                        if results == 'Yes' :
                            with col8 :
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                st.markdown("<p style='text-align: center; color: green;'><b>Bernoulli Naive Bayes Results</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
                                st.markdown('<p></p>', unsafe_allow_html=True)
                                st.markdown('<p></p>', unsafe_allow_html=True)
                        
                        c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                        with c3 :
                            save_button = st.button('Save model')
                        
                        with c4 :
                            delete_button = st.button('Delete model')

                        if save_button :
                            filename = model
                            st.info('Saving Bernoulli Naive Bayes Model')
                            pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                            st.success('Bernoulli Naive Bayes Model saved successfully!')
                        
                        if delete_button :
                            try :
                                os.remove(os.path.join('.','models', model+'.sav'))
                            except :
                                st.error('File not found! Please make sure the model is already saved')

                    except Exception:
                        if strategy == 'Grid Search CV' :
                            st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                        if strategy == 'Random Search CV' :
                            st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

        if model == 'Gaussian Naive Bayes' :
            parameters = {'var_smoothing':np.linspace(1e-9,1.0,10)}
            params_df = pd.DataFrame({'Variable Smoothing': [' , '.join(map(str, parameters['var_smoothing']))]})

            
            st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Gaussian Naive Bayes</p>", unsafe_allow_html=True)
            st.dataframe(params_df)

            if scaling == '' :
                st.warning('Please select scaling technique!')

            if retrain == '' :
                st.warning('Please select to retrain model with optimal parameters or not')

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                col_5, col_6, col_7 = st.columns((3.3,3,1))
 
                with col_5 :
                    st.write('' )

                with col_6 :
                    button = st.button('Tune parameters!')

                with col_7 :
                    st.write('')

                if button :

                    if strategy == 'Grid Search CV' :
                        clf = GridSearchCV(estimator=GaussianNB(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                    if strategy == 'Random Search CV' :
                        clf = RandomizedSearchCV(estimator=GaussianNB(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    
                    if strategy == 'Halving Grid Search CV' :
                        clf = HalvingGridSearchCV(estimator=GaussianNB(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                    # if strategy == 'Bayesian Optimization' :
                    #     clf = BayesSearchCV(estimator=GaussianNB(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    

                    placeholder = st.empty()
                    with placeholder.container():
                        st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                        grid = ParameterGrid(parameters)
                        st.write (f"The total number of parameters-combinations is: {len(grid)}")
                        st.write (f"The total number of iterations is: {len(grid)*113}")

                    start = datetime.now()

                    clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                    end = datetime.now()

                    placeholder.empty()

                    print("Tune Fit Time:", end - start)

                    st.success('Hyperparameter Tuning Done')
                    st.write("Tuning Time:", end - start)

                    st.write('Best parameters for Gaussian Naive Bayes :\n', clf.best_params_)

                    col8, col9 = st.columns((20, 1))

                    try :
                        y_pred = clf.predict(feature_set.drop('Class',axis=1))

                        acc = accuracy_score(feature_set['Class'], y_pred)
                        f1 = f1_score(feature_set['Class'], y_pred)

                        if results == 'Yes' :
                            with col8 :
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                st.markdown("<p style='text-align: center; color: green;'><b>Gaussian Naive Bayes Results</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
                                st.markdown('<p></p>', unsafe_allow_html=True)
                                st.markdown('<p></p>', unsafe_allow_html=True)

                        c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                        with c3 :
                            save_button = st.button('Save model')
                        
                        with c4 :
                            delete_button = st.button('Delete model')

                        if save_button :
                            filename = model
                            st.info('Saving Bernoulli Naive Bayes Model')
                            pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                            st.success('Bernoulli Naive Bayes Model saved successfully!')
                        
                        if delete_button :
                            try :
                                os.remove(os.path.join('.','models', model+'.sav'))
                            except :
                                st.error('File not found! Please make sure the model is already saved')

                    except Exception:
                        if strategy == 'Grid Search CV' :
                            st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                        if strategy == 'Random Search CV' :
                            st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

        if model == 'K-Nearest Neighbors' :
            parameters = {'n_neighbors':range(1,11), 'weights':['uniform', 'distance'], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'], 'metric':['euclidean', 'manhattan', 'minkowski']}
            params_df = pd.DataFrame({'No. of Neighbors': [' , '.join(map(str, parameters['n_neighbors']))],   'Weights': [' , '.join(map(str, parameters['weights']))],
                                  'Algorithm': [' , '.join(map(str, parameters['algorithm']))], 'Metric': [' , '.join(map(str, parameters['metric']))]})
            
            st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for K-Nearest Neighbors</p>", unsafe_allow_html=True)
            st.dataframe(params_df)

            if scaling == '' :
                st.warning('Please select scaling technique!')

            if retrain == '' :
                st.warning('Please select to retrain model with optimal parameters or not')

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                col_5, col_6, col_7 = st.columns((3.3,3,1))
 
                with col_5 :
                    st.write('' )

                with col_6 :
                    button = st.button('Tune parameters!')

                with col_7 :
                    st.write('')

                if button :

                    if strategy == 'Grid Search CV' :
                        clf = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                    if strategy == 'Random Search CV' :
                        clf = RandomizedSearchCV(estimator=KNeighborsClassifier(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    
                    if strategy == 'Halving Grid Search CV' :
                        clf = HalvingGridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                    # if strategy == 'Bayesian Optimization' :
                    #     clf = BayesSearchCV(estimator=KNeighborsClassifier(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    

                    placeholder = st.empty()
                    with placeholder.container():
                        st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                        grid = ParameterGrid(parameters)
                        st.write (f"The total number of parameters-combinations is: {len(grid)}")
                        st.write (f"The total number of iterations is: {len(grid)*113}")

                    start = datetime.now()

                    clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                    end = datetime.now()

                    placeholder.empty()

                    print("Tune Fit Time:", end - start)

                    st.success('Hyperparameter Tuning Done')
                    st.write("Tuning Time:", end - start)

                    st.write('Best parameters for  K-Nearest Neighbors  :\n', clf.best_params_)

                    col8, col9 = st.columns((20, 1))

                    try :
                        y_pred = clf.predict(feature_set.drop('Class',axis=1))

                        acc = accuracy_score(feature_set['Class'], y_pred)
                        f1 = f1_score(feature_set['Class'], y_pred)

                        if results == 'Yes' :
                            with col8 :
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                st.markdown("<p style='text-align: center; color: green;'><b>K-Nearest Neighbors Results</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
                                st.markdown('<p></p>', unsafe_allow_html=True)
                                st.markdown('<p></p>', unsafe_allow_html=True)
                                

                        c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                        with c3 :
                            save_button = st.button('Save model')
                        
                        with c4 :
                            delete_button = st.button('Delete model')

                        if save_button :
                            filename = model
                            st.info('Saving Bernoulli Naive Bayes Model')
                            pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                            st.success('Bernoulli Naive Bayes Model saved successfully!')
                        
                        if delete_button :
                            try :
                                os.remove(os.path.join('.','models', model+'.sav'))
                            except :
                                st.error('File not found! Please make sure the model is already saved')


                    except Exception:
                        if strategy == 'Grid Search CV' :
                            st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                        if strategy == 'Random Search CV' :
                            st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

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

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                col_5, col_6, col_7 = st.columns((3.3,3,1))

                with col_5 :
                    st.write('' )

                with col_6 :
                    button = st.button('Tune parameters!')

                with col_7 :
                    st.write('')

                if button :

                    if strategy == 'Grid Search CV' :
                        clf = GridSearchCV(estimator=LogisticRegression(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                    if strategy == 'Random Search CV' :
                        clf = RandomizedSearchCV(estimator=LogisticRegression(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    
                    if strategy == 'Halving Grid Search CV' :
                        clf = HalvingGridSearchCV(estimator=LogisticRegression(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                    # if strategy == 'Bayesian Optimization' :
                    #     clf = BayesSearchCV(estimator=LogisticRegression(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    

                    placeholder = st.empty()
                    with placeholder.container():
                        st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                        grid = ParameterGrid(parameters)
                        st.write (f"The total number of parameters-combinations is: {len(grid)}")
                        st.write (f"The total number of iterations is: {len(grid)*113}")

                    start = datetime.now()
                   
                    clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                    end = datetime.now()

                    print("Tune Fit Time:", end - start)

                    st.success('Hyperparameter Tuning Done')
                    st.write("Tuning Time:", end - start)

                    st.write('Best parameters for Logistic Regression :\n', clf.best_params_)

                    col8, col9 = st.columns((20, 1))

                    try :
                        y_pred = clf.predict(feature_set.drop('Class',axis=1))
                        # st.write(y_pred)

                        acc = accuracy_score(feature_set['Class'], y_pred)
                        f1 = f1_score(feature_set['Class'], y_pred)

                        if results == 'Yes' :
                            with col8 :
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                st.markdown("<p style='text-align: center; color: green;'><b>Logistic Regression</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
                                st.markdown('<p></p>', unsafe_allow_html=True)
                                st.markdown('<p></p>', unsafe_allow_html=True)

                        c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                        with c3 :
                            save_button = st.button('Save model')
                        
                        with c4 :
                            delete_button = st.button('Delete model')

                        if save_button :
                            filename = model
                            st.info('Saving Bernoulli Naive Bayes Model')
                            pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                            st.success('Bernoulli Naive Bayes Model saved successfully!')
                        
                        if delete_button :
                            try :
                                os.remove(os.path.join('.','models', model+'.sav'))
                            except :
                                st.error('File not found! Please make sure the model is already saved')

                    except Exception:
                        if strategy == 'Grid Search CV' :
                            st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                        if strategy == 'Random Search CV' :
                            st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

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

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                col_5, col_6, col_7 = st.columns((3.3,3,1))
 
                with col_5 :
                    st.write('' )

                with col_6 :
                    button = st.button('Tune parameters!')

                with col_7 :
                    st.write('')

                if button :

                    if strategy == 'Grid Search CV' :
                        clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                    if strategy == 'Random Search CV' :
                        clf = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    
                    if strategy == 'Halving Grid Search CV' :
                        clf = HalvingGridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                    # if strategy == 'Bayesian Optimization' :
                    #     clf = BayesSearchCV(estimator=RandomForestClassifier(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    

                    placeholder = st.empty()
                    with placeholder.container():
                        st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                        grid = ParameterGrid(parameters)
                        st.write (f"The total number of parameters-combinations is: {len(grid)}")
                        st.write (f"The total number of iterations is: {len(grid)*113}")

                    start = datetime.now()

                    clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                    end = datetime.now()

                    placeholder.empty()

                    print("Tune Fit Time:", end - start)

                    st.success('Hyperparameter Tuning Done')
                    st.write("Tuning Time:", end - start)

                    st.write('Best parameters for    :\n', clf.best_params_)

                    col8, col9 = st.columns((20, 1))

                    try :
                        y_pred = clf.predict(feature_set.drop('Class',axis=1))

                        acc = accuracy_score(feature_set['Class'], y_pred)
                        f1 = f1_score(feature_set['Class'], y_pred)

                        if results == 'Yes' :
                            with col8 :
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                st.markdown("<p style='text-align: center; color: green;'><b>Random Forest Results</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
                                st.markdown('<p></p>', unsafe_allow_html=True)
                                st.markdown('<p></p>', unsafe_allow_html=True)

                        c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                        with c3 :
                            save_button = st.button('Save model')
                        
                        with c4 :
                            delete_button = st.button('Delete model')

                        if save_button :
                            filename = model
                            st.info('Saving Bernoulli Naive Bayes Model')
                            pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                            st.success('Bernoulli Naive Bayes Model saved successfully!')
                        
                        if delete_button :
                            try :
                                os.remove(os.path.join('.','models', model+'.sav'))
                            except :
                                st.error('File not found! Please make sure the model is already saved')

                    except Exception:
                        if strategy == 'Grid Search CV' :
                            st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                        if strategy == 'Random Search CV' :
                            st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

        if model == 'Support Vector Machine' :
            parameters = {'kernel':['linear','poly','rbf','sigmoid'], 'C':[0.001,0.01,0.1],
              'gamma':['scale','auto']} # add coef0, shrinking, probability, tol, decision_function_shape
            params_df = pd.DataFrame({'Kernel': [' , '.join(map(str, parameters['kernel']))],   'C': [' , '.join(map(str, parameters['C']))],
                                  'Gamma': [' , '.join(map(str, parameters['gamma']))]})

            
            st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Support Vector Machine</p>", unsafe_allow_html=True)
            st.dataframe(params_df)

            if scaling == '' :
                st.warning('Please select scaling technique!')

            if retrain == '' :
                st.warning('Please select to retrain model with optimal parameters or not')

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                col_5, col_6, col_7 = st.columns((3.3,3,1))
 
                with col_5 :
                    st.write('' )

                with col_6 :
                    button = st.button('Tune parameters!')

                with col_7 :
                    st.write('')

                if button :

                    if strategy == 'Grid Search CV' :
                        clf = GridSearchCV(estimator=SVC(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                    if strategy == 'Random Search CV' :
                        clf = RandomizedSearchCV(estimator=SVC(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    
                    if strategy == 'Halving Grid Search CV' :
                        clf = HalvingGridSearchCV(estimator=SVC(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                    # if strategy == 'Bayesian Optimization' :
                    #     clf = BayesSearchCV(estimator=SVC(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    

                    placeholder = st.empty()
                    with placeholder.container():
                        st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                        grid = ParameterGrid(parameters)
                        st.write (f"The total number of parameters-combinations is: {len(grid)}")
                        st.write (f"The total number of iterations is: {len(grid)*113}")

                    start = datetime.now()

                    clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                    end = datetime.now()

                    placeholder.empty()

                    print("Tune Fit Time:", end - start)

                    st.success('Hyperparameter Tuning Done')
                    st.write("Tuning Time:", end - start)

                    st.write('Best parameters for    :\n', clf.best_params_)

                    col8, col9 = st.columns((20, 1))

                    try :
                        y_pred = clf.predict(feature_set.drop('Class',axis=1))

                        acc = accuracy_score(feature_set['Class'], y_pred)
                        f1 = f1_score(feature_set['Class'], y_pred)

                        if results == 'Yes' :
                            with col8 :
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                st.markdown("<p style='text-align: center; color: green;'><b>Support Vector Machine Results</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)

                        c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                        with c3 :
                            save_button = st.button('Save model')
                        
                        with c4 :
                            delete_button = st.button('Delete model')

                        if save_button :
                            filename = model
                            st.info('Saving Bernoulli Naive Bayes Model')
                            pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                            st.success('Bernoulli Naive Bayes Model saved successfully!')
                        
                        if delete_button :
                            try :
                                os.remove(os.path.join('.','models', model+'.sav'))
                            except :
                                st.error('File not found! Please make sure the model is already saved')

                    except Exception:
                        if strategy == 'Grid Search CV' :
                            st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                        if strategy == 'Random Search CV' :
                            st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

        if model == 'Multi-Layer Perceptron' :
            parameters = {'hidden_layer_sizes':[10,20,50,100],
              'alpha':[0.001,0.01, 0.1], 'activation':['identity','logistic','tanh','relu'],
             'solver':['lbfgs','sgd','adam'], 'learning_rate':['constant','invscaling','adaptive']} # add batch_size, alpha, early_stopping
            params_df = pd.DataFrame({'Hidden Layer Sizes': [' , '.join(map(str, parameters['hidden_layer_sizes']))],   'Alpha': [' , '.join(map(str, parameters['alpha']))],
                                  'Solver': [' , '.join(map(str, parameters['solver']))], 'Learning Rate': [' , '.join(map(str, parameters['learning_rate']))]})

            
            st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Multi-Layer Perceptron</p>", unsafe_allow_html=True)
            st.dataframe(params_df)

            if scaling == '' :
                st.warning('Please select scaling technique!')

            if retrain == '' :
                st.warning('Please select to retrain model with optimal parameters or not')

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                col_5, col_6, col_7 = st.columns((3.3,3,1))
 
                with col_5 :
                    st.write('' )

                with col_6 :
                    button = st.button('Tune parameters!')

                with col_7 :
                    st.write('')

                if button :

                    if strategy == 'Grid Search CV' :
                        clf = GridSearchCV(estimator=MLPClassifier(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                    if strategy == 'Random Search CV' :
                        clf = RandomizedSearchCV(estimator=MLPClassifier(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    
                    if strategy == 'Halving Grid Search CV' :
                        clf = HalvingGridSearchCV(estimator=MLPClassifier(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                    # if strategy == 'Bayesian Optimization' :
                    #     clf = BayesSearchCV(estimator=MLPClassifier(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                    

                    placeholder = st.empty()
                    with placeholder.container():
                        st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                        grid = ParameterGrid(parameters)
                        st.write (f"The total number of parameters-combinations is: {len(grid)}")
                        st.write (f"The total number of iterations is: {len(grid)*113}")

                    start = datetime.now()

                    clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                    end = datetime.now()

                    placeholder.empty()

                    print("Tune Fit Time:", end - start)

                    st.success('Hyperparameter Tuning Done')
                    st.write("Tuning Time:", end - start)

                    st.write('Best parameters for    :\n', clf.best_params_)

                    col8, col9 = st.columns((20, 1))

                    try :
                        y_pred = clf.predict(feature_set.drop('Class',axis=1))

                        acc = accuracy_score(feature_set['Class'], y_pred)
                        f1 = f1_score(feature_set['Class'], y_pred)

                        if results == 'Yes' :
                            with col8 :
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                # st.markdown('<p></p>', unsafe_allow_html=True)
                                st.markdown("<p style='text-align: center; color: green;'><b>Multi-Layer Perceptron Results</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
                                st.markdown('<p></p>', unsafe_allow_html=True)
                                st.markdown('<p></p>', unsafe_allow_html=True)

                        c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                        with c3 :
                            save_button = st.button('Save model')
                        
                        with c4 :
                            delete_button = st.button('Delete model')

                        if save_button :
                            filename = model
                            st.info('Saving Bernoulli Naive Bayes Model')
                            pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                            st.success('Bernoulli Naive Bayes Model saved successfully!')
                        
                        if delete_button :
                            try :
                                os.remove(os.path.join('.','models', model+'.sav'))
                            except :
                                st.error('File not found! Please make sure the model is already saved')

                    except Exception:
                        if strategy == 'Grid Search CV' :
                            st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                        if strategy == 'Random Search CV' :
                            st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')


    if tune_options == 'Create and use a new hyperparameter search space' :
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1 :
            model = st.selectbox('Choose Machine Learning Classification Algorithm', ('','Bernoulli Naive Bayes', 'Gaussian Naive Bayes', 'K-Nearest Neighbors', 
                                                                                    'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Multi-Layer Perceptron'))

        with col2 :
            scaling = st.selectbox('Choose a scaling techinque', ('','No Scaling', 'Standard Scaling', 'Min-Max Scaling', 'Robust Scaling'))

        with col3 :
            retrain = st.selectbox('Retrain with the optimal parameters', ('','No', 'Yes'))

        with col4 :
            strategy = st.selectbox('Choose an Optimization Strategy', ('', 'Grid Search CV', 'Random Search CV', 'Halving Grid Search CV')) #, 'Bayesian Optimization'))
        
        with col5 :
            results = st.selectbox('Show Results (Will Show Only if Retrain is Yes)', ('', 'No', 'Yes'))
        
        if scaling == 'Standard Scaling' :
            scaler = StandardScaler()
            feature_set[feature_set.columns[1:]] = scaler.fit_transform(feature_set[feature_set.columns[1:]])
        
        if scaling == 'Min-Max Scaling' :
            scaler = MinMaxScaler()
            feature_set[feature_set.columns[1:]] = scaler.fit_transform(feature_set[feature_set.columns[1:]])

        if scaling == 'Robust Scaling' :
            scaler = RobustScaler()
            feature_set[feature_set.columns[1:]] = scaler.fit_transform(feature_set[feature_set.columns[1:]]) 

        
        if model == 'Bernoulli Naive Bayes' :
            if scaling == '' :
                st.warning('Please select scaling technique!')

            if retrain == '' :
                st.warning('Please select to retrain model with optimal parameters or not')

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                # placeholder = st.empty()

                # with placeholder.container() :
                col1,col2,col3,col4,col5,col6 = st.columns(6)
                                    
                with col1 :
                    alpha_min = st.number_input('Enter Minimum Value for Alpha')
                with col2 :
                    alpha_max = st.number_input('Enter Maximum Value for Alpha')
                with col3 :
                    alpha_step_size = int(st.text_input('Enter Number of Values to generate for Alpha', value=5))
                with col4 :
                    binarize_min = st.number_input('Enter Minimum Value for Binarize')
                with col5 :
                    binarize_max = st.number_input('Enter Maximum Value for Binarize')
                with col6 :
                    binarize_step_size = int(st.text_input('Enter Number of Values to generate for Binarize', value=5)) 
            
                parameters = {'alpha':np.linspace(alpha_min,alpha_max,int(alpha_step_size)), 'binarize':np.linspace(binarize_min,binarize_max,binarize_step_size), 'fit_prior':[True, False]}
                params_df = pd.DataFrame({'Alpha': [' , '.join(map(str, np.round(parameters['alpha'], 1)))],    'Binarize': [' , '.join(map(str, np.round(parameters['binarize'], 1)))], 
                'Fir Prior':[' , '.join(map(str, parameters['fit_prior']))]})

                st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Bernoulli Naive Bayes</p>", unsafe_allow_html=True)
                st.dataframe(params_df)

                if scaling == '' :
                    st.warning('Please select scaling technique!')

                if retrain == '' :
                    st.warning('Please select to retrain model with optimal parameters or not')

                if strategy == '' :
                    st.warning('Please select a optimization/search strategy')

                if results == '' :
                    st.warning('Please select to show retrain results or not')

                if retrain == 'No' :
                    retrain = False
                
                if retrain == 'Yes' :
                    retrain = True

                if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                    col_5, col_6, col_7 = st.columns((3.3,3,1))
    
                    with col_5 :
                        st.write('' )

                    with col_6 :
                        button = st.button('Tune parameters!')

                    with col_7 :
                        st.write('')

                    if button :

                        if strategy == 'Grid Search CV' :
                            clf = GridSearchCV(estimator=BernoulliNB(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                        if strategy == 'Random Search CV' :
                            clf = RandomizedSearchCV(estimator=BernoulliNB(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        
                        if strategy == 'Halving Grid Search CV' :
                            clf = HalvingGridSearchCV(estimator=BernoulliNB(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                        # if strategy == 'Bayesian Optimization' :
                        #     clf = BayesSearchCV(estimator=BernoulliNB(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        

                        placeholder = st.empty()
                        with placeholder.container():
                            st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                            grid = ParameterGrid(parameters)
                            st.write (f"The total number of parameters-combinations is: {len(grid)}")
                            st.write (f"The total number of iterations is: {len(grid)*113}")

                        start = datetime.now()

                        clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                        end = datetime.now()

                        placeholder.empty()

                        print("Tune Fit Time:", end - start)

                        st.success('Hyperparameter Tuning Done')
                        st.write("Tuning Time:", end - start)

                        st.write('Best parameters for Bernoulli Naive Bayes :\n', clf.best_params_)

                        col8, col9 = st.columns((20, 1))

                        try :
                            y_pred = clf.predict(feature_set.drop('Class',axis=1))

                            acc = accuracy_score(feature_set['Class'], y_pred)
                            f1 = f1_score(feature_set['Class'], y_pred)

                            if results == 'Yes' :
                                with col8 :
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown("<p style='text-align: center; color: green;'><b>Bernoulli Naive Bayes Results</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                            
                            c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                            with c3 :
                                save_button = st.button('Save model')
                            
                            with c4 :
                                delete_button = st.button('Delete model')

                            if save_button :
                                filename = model
                                st.info('Saving Bernoulli Naive Bayes Model')
                                pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                                st.success('Bernoulli Naive Bayes Model saved successfully!')
                            
                            if delete_button :
                                try :
                                    os.remove(os.path.join('.','models', model+'.sav'))
                                except :
                                    st.error('File not found! Please make sure the model is already saved')

                        except Exception:
                            if strategy == 'Grid Search CV' :
                                st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                            if strategy == 'Random Search CV' :
                                st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

        if model == 'Gaussian Naive Bayes' :
            if scaling == '' :
                st.warning('Please select scaling technique!')

            if retrain == '' :
                st.warning('Please select to retrain model with optimal parameters or not')

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):
                col1,col2,col3 = st.columns(3)
                                    
                with col1 :
                    var_min = st.number_input('Enter Minimum Value for Variable Smoothing')
                with col2 :
                    var_max = st.number_input('Enter Maximum Value for Variable Smoothing')
                with col3 :
                    var_step_size = int(st.text_input('Enter Number of Values to generate for Variable Smoothing', value=5))
            
                parameters = {'var_smoothing':np.linspace(var_min,var_max,var_step_size)}
                params_df = pd.DataFrame({'Variable Smoothing': [' , '.join(map(str, parameters['var_smoothing']))]})

                st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Gaussian Naive Bayes</p>", unsafe_allow_html=True)
                st.dataframe(params_df)

                if scaling == '' :
                    st.warning('Please select scaling technique!')

                if retrain == '' :
                    st.warning('Please select to retrain model with optimal parameters or not')

                if strategy == '' :
                    st.warning('Please select a optimization/search strategy')

                if results == '' :
                    st.warning('Please select to show retrain results or not')

                if retrain == 'No' :
                    retrain = False
                
                if retrain == 'Yes' :
                    retrain = True

                if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                    col_5, col_6, col_7 = st.columns((3.3,3,1))
    
                    with col_5 :
                        st.write('' )

                    with col_6 :
                        button = st.button('Tune parameters!')

                    with col_7 :
                        st.write('')

                    if button :

                        if strategy == 'Grid Search CV' :
                            clf = GridSearchCV(estimator=GaussianNB(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                        if strategy == 'Random Search CV' :
                            clf = RandomizedSearchCV(estimator=GaussianNB(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        
                        if strategy == 'Halving Grid Search CV' :
                            clf = HalvingGridSearchCV(estimator=GaussianNB(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                        # if strategy == 'Bayesian Optimization' :
                        #     clf = BayesSearchCV(estimator=GaussianNB(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        

                        placeholder = st.empty()
                        with placeholder.container():
                            st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                            grid = ParameterGrid(parameters)
                            st.write (f"The total number of parameters-combinations is: {len(grid)}")
                            st.write (f"The total number of iterations is: {len(grid)*113}")

                        start = datetime.now()

                        clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                        end = datetime.now()

                        placeholder.empty()

                        print("Tune Fit Time:", end - start)

                        st.success('Hyperparameter Tuning Done')
                        st.write("Tuning Time:", end - start)

                        st.write('Best parameters for Gaussian Naive Bayes :\n', clf.best_params_)

                        col8, col9 = st.columns((20, 1))

                        try :
                            y_pred = clf.predict(feature_set.drop('Class',axis=1))

                            acc = accuracy_score(feature_set['Class'], y_pred)
                            f1 = f1_score(feature_set['Class'], y_pred)

                            if results == 'Yes' :
                                with col8 :
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown("<p style='text-align: center; color: green;'><b>Gaussian Naive Bayes Results</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                            
                            c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                            with c3 :
                                save_button = st.button('Save model')
                            
                            with c4 :
                                delete_button = st.button('Delete model')

                            if save_button :
                                filename = model
                                st.info('Saving Gaussian Naive Bayes Model')
                                pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                                st.success('Gaussian Naive Bayes Model saved successfully!')
                            
                            if delete_button :
                                try :
                                    os.remove(os.path.join('.','models', model+'.sav'))
                                except :
                                    st.error('File not found! Please make sure the model is already saved')

                        except Exception:
                            if strategy == 'Grid Search CV' :
                                st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                            if strategy == 'Random Search CV' :
                                st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

        if model == 'K-Nearest Neighbors' :
            if scaling == '' :
                st.warning('Please select scaling technique!')

            if retrain == '' :
                st.warning('Please select to retrain model with optimal parameters or not')

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):
                col1,col2,col3,col4,col5 = st.columns(5)
                                    
                with col1 :
                    try :
                        neighbors_min = int(st.text_input('Enter Minimum Value for # of Nearest Neighbors', value=10))
                    except ValueError :
                        st.write('Enter a number only')
                    except UnboundLocalError :
                        st.write('')
                with col2 :
                    try :
                        neighbors_max = int(st.text_input('Enter Maximum Value for # of Nearest Neighbors', value=10))
                    except ValueError :
                        st.write('Enter a number only')
                    except UnboundLocalError :
                        st.write('')
                with col3 :
                    weights = st.multiselect('Select Weights', ('uniform', 'distance'))
                with col4 :
                    algorithm = st.multiselect('Select Algorithms', ('auto', 'ball_tree', 'kd_tree', 'brute'))
                with col5 :
                    metric = st.multiselect('Select Metrics', ('euclidean', 'manhattan', 'minkowski'))


                parameters = {'n_neighbors':range(neighbors_min,neighbors_max), 'weights':[w for w in weights], 'algorithm':[a for a in algorithm], 'metric':[m for m in metric]}
                params_df = pd.DataFrame({'No. of Neighbors': [' , '.join(map(str, parameters['n_neighbors']))],   'Weights': [' , '.join(map(str, parameters['weights']))],
                                  'Algorithm': [' , '.join(map(str, parameters['algorithm']))], 'Metric': [' , '.join(map(str, parameters['metric']))]})

                st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for K-Nearest Neighbors</p>", unsafe_allow_html=True)
                st.dataframe(params_df)

                if scaling == '' :
                    st.warning('Please select scaling technique!')

                if retrain == '' :
                    st.warning('Please select to retrain model with optimal parameters or not')

                if strategy == '' :
                    st.warning('Please select a optimization/search strategy')

                if results == '' :
                    st.warning('Please select to show retrain results or not')

                if retrain == 'No' :
                    retrain = False
                
                if retrain == 'Yes' :
                    retrain = True

                if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                    col_5, col_6, col_7 = st.columns((3.3,3,1))
    
                    with col_5 :
                        st.write('' )

                    with col_6 :
                        button = st.button('Tune parameters!')

                    with col_7 :
                        st.write('')

                    if button :

                        if strategy == 'Grid Search CV' :
                            clf = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                        if strategy == 'Random Search CV' :
                            clf = RandomizedSearchCV(estimator=KNeighborsClassifier(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        
                        if strategy == 'Halving Grid Search CV' :
                            clf = HalvingGridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                        # if strategy == 'Bayesian Optimization' :
                        #     clf = BayesSearchCV(estimator=KNeighborsClassifier(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        

                        placeholder = st.empty()
                        with placeholder.container():
                            st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                            grid = ParameterGrid(parameters)
                            st.write (f"The total number of parameters-combinations is: {len(grid)}")
                            st.write (f"The total number of iterations is: {len(grid)*113}")

                        start = datetime.now()

                        clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                        end = datetime.now()

                        placeholder.empty()

                        print("Tune Fit Time:", end - start)

                        st.success('Hyperparameter Tuning Done')
                        st.write("Tuning Time:", end - start)

                        st.write('Best parameters for K-Nearest Neighbors :\n', clf.best_params_)

                        col8, col9 = st.columns((20, 1))

                        try :
                            y_pred = clf.predict(feature_set.drop('Class',axis=1))

                            acc = accuracy_score(feature_set['Class'], y_pred)
                            f1 = f1_score(feature_set['Class'], y_pred)

                            if results == 'Yes' :
                                with col8 :
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown("<p style='text-align: center; color: green;'><b>K-Nearest Neighbors Results</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                            
                            c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                            with c3 :
                                save_button = st.button('Save model')
                            
                            with c4 :
                                delete_button = st.button('Delete model')

                            if save_button :
                                filename = model
                                st.info('Saving K-Nearest Neighbors Model')
                                pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                                st.success('K-Nearest Neighbors Model saved successfully!')
                            
                            if delete_button :
                                try :
                                    os.remove(os.path.join('.','models', model+'.sav'))
                                except :
                                    st.error('File not found! Please make sure the model is already saved')

                        except Exception:
                            if strategy == 'Grid Search CV' :
                                st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                            if strategy == 'Random Search CV' :
                                st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')     

        if model == 'Logistic Regression' :
            if scaling == '' :
                st.warning('Please select scaling technique!')

            if retrain == '' :
                st.warning('Please select to retrain model with optimal parameters or not')

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):
                col1,col2,col3 = st.columns(3)
                                    
                with col1 :
                    penalty = st.multiselect('Select Penalty', ('l1', 'l2', 'elasticnet', 'none'))
                    solver = st.multiselect('Select Solver', ('newton-cg','lbfgs','liblinear','sag','saga'))
                with col2 :
                    tol_min = float(st.number_input('Enter Minimum Value for Tolerance', value=0.001, key='tolerance minimum'))
                    tol_max = float(st.number_input('Enter Maximum Value for Tolerance', value=1.0, key='tolerance maximum'))
                    try :
                        tol_step_size = int(st.text_input('Enter Number of Value to Generate for Tolerance', value=1, key='tolerance step size'))
                    except ValueError :
                        st.write('Enter a number only')
                    except UnboundLocalError :
                        st.write('')
                with col3 :
                    try :
                        c_min = float(st.number_input('Enter Minimum Value for C (Penalty)', value=0.001, key='c minimum'))
                    except ValueError :
                        st.write('Enter a number only')
                    except UnboundLocalError :
                        st.write('')
                    try :
                        c_max = float(st.number_input('Enter Maximum Value for C (Penalty)', value=1.0, key='c maximum'))
                    except ValueError :
                        st.write('Enter a number only')
                    except UnboundLocalError :
                        st.write('')
                    try :
                        c_step_size = int(st.text_input('Enter Number of Value to Generate for C (Penalty)', value=1, key='c step size'))
                    except ValueError :
                        st.write('Enter a number only')
                    except UnboundLocalError :
                        st.write('')


                parameters = {'penalty':[p for p in penalty], 'dual':[True, False], 'tol':np.linspace(tol_min,tol_max,tol_step_size), 'C':np.linspace(c_min, c_max,c_step_size), 'solver':[s for s in solver]}
                
                params_df = pd.DataFrame({'Penalty': [' , '.join(map(str, parameters['penalty']))], 'Dual': [' , '.join(map(str, parameters['dual']))], 
                'Tolerance': [' , '.join(map(str, parameters['tol']))], 'C': [' , '.join(map(str, parameters['C']))], 'Solver': [' , '.join(map(str, parameters['solver']))]})

                st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Logistic Regression</p>", unsafe_allow_html=True)
                st.dataframe(params_df)

                if scaling == '' :
                    st.warning('Please select scaling technique!')

                if retrain == '' :
                    st.warning('Please select to retrain model with optimal parameters or not')

                if strategy == '' :
                    st.warning('Please select a optimization/search strategy')

                if results == '' :
                    st.warning('Please select to show retrain results or not')

                if retrain == 'No' :
                    retrain = False
                
                if retrain == 'Yes' :
                    retrain = True

                if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                    col_5, col_6, col_7 = st.columns((3.3,3,1))
    
                    with col_5 :
                        st.write('' )

                    with col_6 :
                        button = st.button('Tune parameters!')

                    with col_7 :
                        st.write('')

                    if button :

                        if strategy == 'Grid Search CV' :
                            clf = GridSearchCV(estimator=LogisticRegression(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                        if strategy == 'Random Search CV' :
                            clf = RandomizedSearchCV(estimator=LogisticRegression(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        
                        if strategy == 'Halving Grid Search CV' :
                            clf = HalvingGridSearchCV(estimator=LogisticRegression(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                        # if strategy == 'Bayesian Optimization' :
                        #     clf = BayesSearchCV(estimator=LogisticRegression(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        

                        placeholder = st.empty()
                        with placeholder.container():
                            st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                            grid = ParameterGrid(parameters)
                            st.write (f"The total number of parameters-combinations is: {len(grid)}")
                            st.write (f"The total number of iterations is: {len(grid)*113}")

                        start = datetime.now()

                        clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                        end = datetime.now()

                        placeholder.empty()

                        print("Tune Fit Time:", end - start)

                        st.success('Hyperparameter Tuning Done')
                        st.write("Tuning Time:", end - start)

                        st.write('Best parameters for Logistic Regression :\n', clf.best_params_)

                        col8, col9 = st.columns((20, 1))

                        try :
                            y_pred = clf.predict(feature_set.drop('Class',axis=1))

                            acc = accuracy_score(feature_set['Class'], y_pred)
                            f1 = f1_score(feature_set['Class'], y_pred)

                            if results == 'Yes' :
                                with col8 :
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown("<p style='text-align: center; color: green;'><b>Logistic Regression Results</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                            
                            c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                            with c3 :
                                save_button = st.button('Save model')
                            
                            with c4 :
                                delete_button = st.button('Delete model')

                            if save_button :
                                filename = model
                                st.info('Saving Logistic Regression Model')
                                pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                                st.success('Logistic Regression Model saved successfully!')
                            
                            if delete_button :
                                try :
                                    os.remove(os.path.join('.','models', model+'.sav'))
                                except :
                                    st.error('File not found! Please make sure the model is already saved')

                        except Exception:
                            if strategy == 'Grid Search CV' :
                                st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                            if strategy == 'Random Search CV' :
                                st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

        if model == 'Random Forest' :
            if scaling == '' :
                st.warning('Please select scaling technique!')

            if retrain == '' :
                st.warning('Please select to retrain model with optimal parameters or not')

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):
                col1,col2,col3,col4, col5,col6,col7 = st.columns(7)
                                    
                with col1 :
                    estimators_min = int(st.text_input('Enter Minimum Value for Estimators', value=1))
                    estimators_max = int(st.text_input('Enter Maximum Value for Estimators', value=10))
                    estimators_step_size = int(st.text_input('Enter Step Size Value for Estimators', value=1))
                with col2 :
                    criterion = st.multiselect('Select Criterion', ('gini', 'entropy'))
                    max_features = st.multiselect('Select Max Features', ('auto', 'sqrt', 'log2'))
                    class_weights = st.multiselect('Select Class Weights', ('balanced', 'balanced_subsample'))
                with col3 :
                    max_depth_min = int(st.text_input('Enter Minimum Value of Max Depth', value=1, key='max depth min'))
                    max_depth_max = int(st.text_input('Enter Minimum Value of Max Depth', value=10, key='max depth max'))
                    max_depth_step_size = int(st.text_input('Enter Number of Values to Generate for Max Depth', value=1, key='max depth step size'))
                with col4 :
                    min_samples_split_min = int(st.text_input('Enter Minimum Value of Minimum Samples Split', value=1, key='min samples split min'))
                    min_samples_split_max = int(st.text_input('Enter Maximum Value of Minimum Samples Split', value=10, key='min samples split max'))
                    min_samples_split_step_size = int(st.text_input('Enter Number of Values to Generate for Minimum Samples Split', value=1, key='min samples split step size'))
                with col5 :
                    min_samples_leaf_min = int(st.text_input('Enter Minimum Value of Minimum Samples Leaf', value=1, key='min samples leaf min'))
                    min_samples_leaf_max = int(st.text_input('Enter Maximum Value of Minimum Samples Leaf', value=10, key='min samples leaf max'))
                    min_samples_leaf_step_size = int(st.text_input('Enter Number of Values to Generate for Minimum Samples Leaf', value=1, key='min samples leaf step size'))
                with col6 :
                    max_leaf_nodes_min = int(st.text_input('Enter Minimum Value of Maximum Leaf Nodes', value=1, key='max leaf nodes min'))
                    max_leaf_nodes_max = int(st.text_input('Enter Maximum Value of Maximum Leaf Nodes', value=10, key='max leaf nodes max'))
                    max_leaf_nodes_step_size = int(st.text_input('Enter Number of Values to Generate for Maximum Leaf Nodes', value=1, key='max leaf nodes step size'))
                with col7 :
                    max_samples_min = int(st.text_input('Enter Minimum Value of Maximum Leaf Nodes', value=1, key='max samples nodes min'))
                    max_samples_max = int(st.text_input('Enter Maximum Value of Maximum Leaf Nodes', value=10, key='max samples nodes max'))
                    max_samples_step_size = int(st.text_input('Enter Number of Values to Generate for Maximum Leaf Nodes', value=1, key='max samples nodes step size'))


                # parameters = {'penalty':[p for p in penalty], 'dual':[True, False], 'tol':np.linspace(tol_min,tol_max,tol_step_size), 'C':np.linspace(c_min, c_max,c_step_size), 'solver':[s for s in solver]}
                parameters = {'n_estimators':np.arange(estimators_min,estimators_max,estimators_step_size), 'criterion':[c for c in criterion], 'max_features':[f for f in max_features],
                'max_depth':np.arange(max_depth_min,max_depth_max,max_depth_step_size), 'min_samples_split':np.arange(min_samples_split_min,min_samples_split_max,min_samples_split_step_size),
                'min_samples_leaf':np.arange(min_samples_split_min,min_samples_split_max,min_samples_split_step_size),
                'max_leaf_nodes':np.arange(max_leaf_nodes_min, max_leaf_nodes_max, max_leaf_nodes_step_size), 'max_samples':np.arange(max_samples_min, max_samples_max, max_samples_step_size)}
            
                params_df = pd.DataFrame({'Number of Estimators': [' , '.join(map(str, parameters['n_estimators']))], 'Criterion': [' , '.join(map(str, parameters['criterion']))], 
                'Max Features': [' , '.join(map(str, parameters['max_features']))], 'Max Depth': [' , '.join(map(str, parameters['max_depth']))], 
                'Min Samples Split': [' , '.join(map(str, parameters['min_samples_split']))], 'Min Samples Leaf': [' , '.join(map(str, parameters['min_samples_leaf']))],
                'Max Leaf Nodes': [' , '.join(map(str, parameters['max_leaf_nodes']))], 'Max Samples': [' , '.join(map(str, parameters['max_samples']))]})

                st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Random Forest</p>", unsafe_allow_html=True)
                st.dataframe(params_df)

                if scaling == '' :
                    st.warning('Please select scaling technique!')

                if retrain == '' :
                    st.warning('Please select to retrain model with optimal parameters or not')

                if strategy == '' :
                    st.warning('Please select a optimization/search strategy')

                if results == '' :
                    st.warning('Please select to show retrain results or not')

                if retrain == 'No' :
                    retrain = False
                
                if retrain == 'Yes' :
                    retrain = True

                if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                    col_5, col_6, col_7 = st.columns((3.3,3,1))
    
                    with col_5 :
                        st.write('' )

                    with col_6 :
                        button = st.button('Tune parameters!')

                    with col_7 :
                        st.write('')

                    if button :

                        if strategy == 'Grid Search CV' :
                            clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                        if strategy == 'Random Search CV' :
                            clf = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        
                        if strategy == 'Halving Grid Search CV' :
                            clf = HalvingGridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                        # if strategy == 'Bayesian Optimization' :
                        #     clf = BayesSearchCV(estimator=RandomForestClassifier(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        

                        placeholder = st.empty()
                        with placeholder.container():
                            st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                            grid = ParameterGrid(parameters)
                            st.write (f"The total number of parameters-combinations is: {len(grid)}")
                            st.write (f"The total number of iterations is: {len(grid)*113}")

                        start = datetime.now()

                        clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                        end = datetime.now()

                        placeholder.empty()

                        print("Tune Fit Time:", end - start)

                        st.success('Hyperparameter Tuning Done')
                        st.write("Tuning Time:", end - start)

                        st.write('Best parameters for Random Forest :\n', clf.best_params_)

                        col8, col9 = st.columns((20, 1))

                        try :
                            y_pred = clf.predict(feature_set.drop('Class',axis=1))

                            acc = accuracy_score(feature_set['Class'], y_pred)
                            f1 = f1_score(feature_set['Class'], y_pred)

                            if results == 'Yes' :
                                with col8 :
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown("<p style='text-align: center; color: green;'><b>Random Forest Results</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                            
                            c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                            with c3 :
                                save_button = st.button('Save model')
                            
                            with c4 :
                                delete_button = st.button('Delete model')

                            if save_button :
                                filename = model
                                st.info('Saving Random Forest Model')
                                pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                                st.success('Random Forest Model saved successfully!')
                            
                            if delete_button :
                                try :
                                    os.remove(os.path.join('.','models', model+'.sav'))
                                except :
                                    st.error('File not found! Please make sure the model is already saved')

                        except Exception:
                            if strategy == 'Grid Search CV' :
                                st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                            if strategy == 'Random Search CV' :
                                st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

        if model == 'Support Vector Machine' :
            if scaling == '' :
                st.warning('Please select scaling technique!')

            if retrain == '' :
                st.warning('Please select to retrain model with optimal parameters or not')

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):
                col1,col2,col3,col4,col5 = st.columns(5)
                                    
                with col1 :
                    C_min = int(st.text_input('Enter Minimum Value for C', value=1))
                    C_max = int(st.text_input('Enter Maximum Value for C', value=10))
                    C_step_size = int(st.text_input('Enter Step Size Value for C', value=1))
                with col2 :
                    degree_min = int(st.text_input('Enter Minimum Value for Degree of Polynomial', value=1))
                    degree_max = int(st.text_input('Enter Maximum Value for Degree of Polynomial', value=10))
                    degree_step_size = int(st.text_input('Enter Step Size Value for Degree of Polynomial', value=1))
                with col3 :
                    coef0_min = int(st.text_input('Enter Minimum Value for Independent Term in Kernel Function', value=1))
                    coef0_max = int(st.text_input('Enter Maximum Value for Independent Term in Kernel Function', value=10))
                    coef0_step_size = int(st.text_input('Enter Step Size Value for Independent Term in Kernel Function', value=1))
                with col4 :
                    tol_min = int(st.text_input('Enter Minimum Value for Tolerance for Stopping', value=1))
                    tol_max = int(st.text_input('Enter Maximum Value for Tolerance for Stopping', value=10))
                    tol_step_size = int(st.text_input('Enter Step Size Value for Tolerance for Stopping', value=1))
                with col5 :
                    gamma = st.multiselect('Select Gamma Values', ['scale', 'auto'])
                    kernel = st.multiselect('Select Kernel Values', ['linear', 'poly', 'rbf', 'sigmoid'])
                    shrinking = st.multiselect('Select Shrinking', [True, False])
                    prob = st.multiselect('Select Probabilities', [True, False])

                parameters = {'C':np.arange(C_min,C_max,C_step_size), 'degree':np.arange(degree_min,degree_max,degree_step_size), 
                'coef0':np.arange(coef0_min,coef0_max,coef0_step_size), 'tol':np.arange(tol_min,tol_max,tol_step_size), 'gamma':gamma, 'kernel':kernel, 'shrinking':shrinking, 'probability':prob}
            
                params_df = pd.DataFrame({'C': [' , '.join(map(str, parameters['C']))], 'Degree': [' , '.join(map(str, parameters['degree']))], 
                'Coef0': [' , '.join(map(str, parameters['coef0']))], 'Tolerance': [' , '.join(map(str, parameters['tol']))], 
                'Gamma': [' , '.join(map(str, parameters['gamma']))], 'Kernel': [' , '.join(map(str, parameters['kernel']))],
                'Shrinking': [' , '.join(map(str, parameters['shrinking']))]})

                st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Support Vector MachineSVC</p>", unsafe_allow_html=True)
                st.dataframe(params_df)

                if scaling == '' :
                    st.warning('Please select scaling technique!')

                if retrain == '' :
                    st.warning('Please select to retrain model with optimal parameters or not')

                if strategy == '' :
                    st.warning('Please select a optimization/search strategy')

                if results == '' :
                    st.warning('Please select to show retrain results or not')

                if retrain == 'No' :
                    retrain = False
                
                if retrain == 'Yes' :
                    retrain = True

                if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                    col_5, col_6, col_7 = st.columns((3.3,3,1))
    
                    with col_5 :
                        st.write('' )

                    with col_6 :
                        button = st.button('Tune parameters!')

                    with col_7 :
                        st.write('')

                    if button :

                        if strategy == 'Grid Search CV' :
                            clf = GridSearchCV(estimator=SVC(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                        if strategy == 'Random Search CV' :
                            clf = RandomizedSearchCV(estimator=SVC(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        
                        if strategy == 'Halving Grid Search CV' :
                            clf = HalvingGridSearchCV(estimator=SVC(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                        # if strategy == 'Bayesian Optimization' :
                        #     clf = BayesSearchCV(estimator=RandomForestClassifier(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        

                        placeholder = st.empty()
                        with placeholder.container():
                            st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                            grid = ParameterGrid(parameters)
                            st.write (f"The total number of parameters-combinations is: {len(grid)}")
                            st.write (f"The total number of iterations is: {len(grid)*113}")

                        start = datetime.now()

                        clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                        end = datetime.now()

                        placeholder.empty()

                        print("Tune Fit Time:", end - start)

                        st.success('Hyperparameter Tuning Done')
                        st.write("Tuning Time:", end - start)

                        st.write('Best parameters for Support Vector Machine :\n', clf.best_params_)

                        col8, col9 = st.columns((20, 1))

                        try :
                            y_pred = clf.predict(feature_set.drop('Class',axis=1))

                            acc = accuracy_score(feature_set['Class'], y_pred)
                            f1 = f1_score(feature_set['Class'], y_pred)

                            if results == 'Yes' :
                                with col8 :
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown("<p style='text-align: center; color: green;'><b>Support Vector Machine Results</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                            
                            c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                            with c3 :
                                save_button = st.button('Save model')
                            
                            with c4 :
                                delete_button = st.button('Delete model')

                            if save_button :
                                filename = model
                                st.info('Saving Support Vector Machine Model')
                                pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                                st.success('Support Vector Machine Model saved successfully!')
                            
                            if delete_button :
                                try :
                                    os.remove(os.path.join('.','models', model+'.sav'))
                                except :
                                    st.error('File not found! Please make sure the model is already saved')

                        except Exception:
                            if strategy == 'Grid Search CV' :
                                st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                            if strategy == 'Random Search CV' :
                                st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

        if model == 'Multi-Layer Perceptron' :
            if scaling == '' :
                st.warning('Please select scaling technique!')

            if retrain == '' :
                st.warning('Please select to retrain model with optimal parameters or not')

            if strategy == '' :
                st.warning('Please select a optimization/search strategy')

            if results == '' :
                st.warning('Please select to show retrain results or not')

            if retrain == 'No' :
                retrain = False
            
            if retrain == 'Yes' :
                retrain = True

            if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):
                col1,col2,col3,col4,col5,col6 = st.columns(6)
                                    
                with col1 :
                    layers_min = int(st.text_input('Enter Minimum Value for # of Hidden Layers', value=1))
                    layers_max = int(st.text_input('Enter Maximum Value for # of Hidden Layers', value=10))
                    layers_step_size = int(st.text_input('Enter Step Size Value for # of Hidden Layers', value=1))
                with col2 :
                    Alpha_min = int(st.text_input('Enter Minimum Value for Alpha', value=1))
                    Alpha_max = int(st.text_input('Enter Maximum Value for Alpha', value=10))
                    Alpha_step_size = int(st.text_input('Enter Step Size Value for Alpha', value=1))
                with col3 :
                    lr_min = float(st.text_input('Enter Minimum Value for Learning Rate', value=0.001))
                    lr_max = float(st.text_input('Enter Maximum Value for Learning Rate', value=10))
                    lr_step_size = int(st.text_input('Enter Step Size Value for Learning Rate', value=1))
                with col4 :
                    power_t_min = int(st.text_input('Enter Minimum Value for Power T', value=1))
                    power_t_max = int(st.text_input('Enter Maximum Value for Power T', value=10))
                    power_t_step_size = int(st.text_input('Enter Step Size Value for Power T', value=1))
                with col5 :
                    tol_min = float(st.text_input('Enter Minimum Value for Stopping Tolerance', value=1))
                    tol_max = float(st.text_input('Enter Maximum Value for Stopping Tolerance', value=10))
                    tol_step_size = int(st.text_input('Enter Step Size Value for Stopping Tolerance', value=1))
                with col6 :
                    activation = st.multiselect('Select Activation Functions', ['identity', 'logistic', 'tanh', 'relu'])
                    solver = st.multiselect('Select Solvers', ['lbfgs', 'sgd', 'adam'])
                    lrs = st.multiselect('Select Learning Rate Types', ['constant', 'invscaling', 'adaptive'])
                    prob = st.multiselect('Select Shrinking', [True, False])

                parameters = {'C':np.arange(C_min,C_max,C_step_size), 'IGNORE':[c for c in criterion], 'IGNORE':[f for f in max_features], 'degree':np.arange(degree_min,degree_max,degree_step_size), 
                'coef0':np.arange(coef0_min,coef0_max,coef0_step_size), 'tol':np.arange(tol_min,tol_min,tol_min), 'gamma':gamma, 'kernel':kernel, 'shrinking':shrinking, 'prob':prob}
            
                params_df = pd.DataFrame({'C': [' , '.join(map(str, parameters['C']))], 'Degree': [' , '.join(map(str, parameters['degree']))], 
                'Coef0': [' , '.join(map(str, parameters['coef0']))], 'Tolerance': [' , '.join(map(str, parameters['tol']))], 
                'Gamma': [' , '.join(map(str, parameters['gamma']))], 'Kernel': [' , '.join(map(str, parameters['kernel']))],
                'Shrinking': [' , '.join(map(str, parameters['shrinking']))]})

                st.markdown("<p style='text-align: center; color: green;'>Parameter Search Space for Random Forest</p>", unsafe_allow_html=True)
                st.dataframe(params_df)

                if scaling == '' :
                    st.warning('Please select scaling technique!')

                if retrain == '' :
                    st.warning('Please select to retrain model with optimal parameters or not')

                if strategy == '' :
                    st.warning('Please select a optimization/search strategy')

                if results == '' :
                    st.warning('Please select to show retrain results or not')

                if retrain == 'No' :
                    retrain = False
                
                if retrain == 'Yes' :
                    retrain = True

                if (scaling != '') and (retrain != '') and (strategy != '') and (results != ''):

                    col_5, col_6, col_7 = st.columns((3.3,3,1))
    
                    with col_5 :
                        st.write('' )

                    with col_6 :
                        button = st.button('Tune parameters!')

                    with col_7 :
                        st.write('')

                    if button :

                        if strategy == 'Grid Search CV' :
                            clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)

                        if strategy == 'Random Search CV' :
                            clf = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        
                        if strategy == 'Halving Grid Search CV' :
                            clf = HalvingGridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=tscv, min_resources=1, max_resources=10,verbose=10, n_jobs=1, refit=retrain)

                        # if strategy == 'Bayesian Optimization' :
                        #     clf = BayesSearchCV(estimator=RandomForestClassifier(), search_spaces=parameters, cv=tscv, verbose=10, n_jobs=1, refit=retrain)
                        

                        placeholder = st.empty()
                        with placeholder.container():
                            st.info('Tuning the parameters. This make take a while. Please be patient. Thank you 😊')
                            grid = ParameterGrid(parameters)
                            st.write (f"The total number of parameters-combinations is: {len(grid)}")
                            st.write (f"The total number of iterations is: {len(grid)*113}")

                        start = datetime.now()

                        clf.fit(feature_set.drop('Class',axis=1),feature_set['Class'])

                        end = datetime.now()

                        placeholder.empty()

                        print("Tune Fit Time:", end - start)

                        st.success('Hyperparameter Tuning Done')
                        st.write("Tuning Time:", end - start)

                        st.write('Best parameters for Random Forest :\n', clf.best_params_)

                        col8, col9 = st.columns((20, 1))

                        try :
                            y_pred = clf.predict(feature_set.drop('Class',axis=1))

                            acc = accuracy_score(feature_set['Class'], y_pred)
                            f1 = f1_score(feature_set['Class'], y_pred)

                            if results == 'Yes' :
                                with col8 :
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    # st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown("<p style='text-align: center; color: green;'><b>Support Vector Machine Results</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {acc}</b></p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {f1}</b></p>", unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                                    st.markdown('<p></p>', unsafe_allow_html=True)
                            
                            c1,c2,c3,c4 = st.columns((1,1,1,2.5))

                            with c3 :
                                save_button = st.button('Save model')
                            
                            with c4 :
                                delete_button = st.button('Delete model')

                            if save_button :
                                filename = model
                                st.info('Saving Support Vector Machine Model')
                                pickle.dump(clf, open(os.path.join('.','models', filename+'.sav'), 'wb'))
                                st.success('Support Vector Machine Model saved successfully!')
                            
                            if delete_button :
                                try :
                                    os.remove(os.path.join('.','models', model+'.sav'))
                                except :
                                    st.error('File not found! Please make sure the model is already saved')

                        except Exception:
                            if strategy == 'Grid Search CV' :
                                st.error('GridSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')

                            if strategy == 'Random Search CV' :
                                st.error('RandomizedSearchCV can only predict if the model is retrained with the best parameters! Please select `Retrain with the optimal parameters` to True ')
