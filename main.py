import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

iris_dataset = load_iris()
df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['Target'] = cancer.target
# taxi = pd.read_csv(r'C:\Users\rodri\Documents\Streamlit\MisraTurp\taxi_data.csv').iloc[0:1000,:]


header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()


with header:
    st.title('Welcome to this tutorial')
    st.text('In this projet I will try to deploy something.')

with dataset:
    st.header('This is where the data set will be')

    st.write(df)
    
    st.header('Worst Perimeter Distribution')
    st.bar_chart(df['worst perimeter'].value_counts().head(100))

with features:
    st.header('In here we will explore some of the features')
    
    st.markdown('* **I chose this  frist feature because blablabla**')
    st.markdown('* **I chose this second feature because blablabla**')

with modelTraining:
    st.header('Let us talk about the model training')
    
    sel_col, disp_col = st.columns(2)
    
    max_depth = sel_col.slider('Select Max Depth of the Model', min_value=10,
                               max_value=1000, value=500)
    n_estimators = sel_col.selectbox('Select the number of estimators',
                                     options=[100,200,300,400,500], index=4)
    input_featuer = sel_col.text_input('Write the feature to use as input feature',
                                       'worst perimeter')
    sel_col.text('List of Features: ')
    sel_col.write(df.columns)
    

    regr = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    
    
    X_train, X_test, y_train, y_test = train_test_split(df[[input_featuer]],
                                        cancer.target, random_state=0) 
    regr.fit(X_train, y_train)
    
    disp_col.subheader('Train Set Score')
    disp_col.write(regr.score(X_train,y_train))
    
    disp_col.subheader('Test Set Score')
    disp_col.write(regr.score(X_test,y_test))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    