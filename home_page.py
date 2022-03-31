import streamlit as st

def homepage() :
    st.title('Welcome')
    st.markdown("<p style='text-align: left; color: orange; font-size:16px'><b><i>Your journey to investing in Bitcoin starts here!</i></b></p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left; color: white; font-size:16px'>Let's walk you through each of the functionalities of this webapp. Each \
    tab represents the functionalities of this webapp. The name of each functionality indicates what it does, and you can find the in-depth \
    details below", unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.subheader('Data')
    st.markdown("<p style='text-align: left; color: white; font-size:16px'>The data function has 3 sub-functions  \
    <ul><li>Showing the default dataset : This is the data that is used for this project. It allows you to see the raw news articles data after \
    it has been processed to remove punctuation, stopwords as well as being lemmatized. You can also see the Bitcoin hourly price data by selecting \
    from the drop-down menu. This menu also allows you to change the theme of the table to your preferences</ul></li> \
    <ul><li>Upload data to create new feature set : This function allows you to upload the csv file containing the links for the news articles \
    and the csv file of the hourly Bitcoin price. You can then use the uploaded files to create a new feature set which you can \
    use to train your machine learning models.</ul></li> \
    <ul><li>Show Uploaded Data : This is the same as the first option, but allows you to select a uploaded file which you want to see.</ul></li></p>", unsafe_allow_html=True)

    st.markdown("<p style='text-align: left; color: #C19A6B; font-size:16px'><i>Currently, you can only train with new feature set or the default one, but \
    soon you will be able to combine multiple feature sets for training, validating/evaluating, hyperparameter tuning and serving predictions</i></p>", unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.subheader('Exploratory Data Analysis')
    st.markdown("<p style='text-align: left; color: white; font-size:16px'>This function allows you to perform exploratory data analysis on \
    either the data used for this project (i.e. the default data), or use a uploaded file to perform exploratory data analysis. Currently there are three \
    methods/options for EDA : show the top 15 most frequent words, show the 15 most frequent words by sentiment (either as bar charts or tree maps) \
    and word cloud </p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left; color: #C19A6B; font-size:16px'><i>In the future, you will be able to get the plot the Bitcoin price charts, plot the line chart\
    showing to trend of number of news articles and see the distribution of the news articles based on the sentiment.</i></p>", unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.subheader('Train Models')
    st.markdown("<p style='text-align: left; color: white; font-size:16px'>Training models function has 3 sub-functions  \
    <ul><li>Train using default feature set : This is the feature set that is used for this project. It allows you to choose a mdoel form the drop-down menu to train \
    You can also choose a scaling from a choice of three techniques. Once the model and the scaling technique is chosen, the feature set with the applied scaling \
    can be seen. The models can be trained either using the default parameters used by sklearn or use the rough-grained parameter obtained after the inital \
    hyperparameter tuning using time-series cross validation. After training, the model can either be saved or deleted using the respective buttons. The results of \
    training are displayed after the training is complete, showing the train accuracy, the test accuracy and the F1 score</ul></li> \
    <ul><li>Use created feature set(s) : This function allows you to use any feature sets to train the models.</ul></li> \
    <ul><li>Perform Hyperparameter Tuning : This allows you to perform hyperparameter tuning. You can either use the rough-grain hyperparameter search space used in this \
    project or you can create your own search space. It allows you to choose your scaling technique, as well as the optimization technique (GridSearchCV or \
    RandomizedSearchCV). If you choose to define your own search space, the hyperparameters for each algorithm will be provided along with boxes to enter values. The \
    generated search space will then be displayed in a tabular format for your convinence</ul></li></p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left; color: #C19A6B; font-size:16px'><i>Currently the training is limited by which feature set to use (can only use one \
    feature set). This will be changed very soon to allow for combining feature sets for training. The hyperparameter tuning is also limited by compute resources. \
    This will be fixed later by leverage distributed clusters on cloud platforms such as AWS SageMaker, Google Cloud Platform (GCP) and IBM Watson. Furthermore, \
    when the stable versions of other optimization techniques (HalvingGridSearch, BayesOptimization etc) will be released, they will also be included. At the moment \
    not all the hyperparameters are provided for each model due to the screen layout constraints of the UI. This will be fixed when the UI of this webapp is \
    revised in the next few iterations</i></p>", unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.subheader('Serve Predictions')
    st.markdown("<p style='text-align: left; color: white; font-size:16px'>This function allows you to choose the feature set for predicting by loading a saved model \
    to make predictions. The results of the predictions including the acutal predictions, the accuracy and the f1 score are displayed</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left; color: #C19A6B; font-size:16px'><i>Future work involves using model deployment patterns such as solo, shadow mode, \
    canary mode and blue/green mode.</i></p>", unsafe_allow_html=True)