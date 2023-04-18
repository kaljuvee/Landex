import streamlit as st
from prophet.plot import plot_plotly
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


#st.title('Landex ai Project')

st.write("""
# Explore Different Results
""")

st.sidebar.image("data/landex.png", use_column_width=True)

dataset_name=st.sidebar.selectbox("Select Dataset",("Estonia","France"))
land_type=st.sidebar.selectbox("Select type of land",("Forest land","Farmland",'Forest land and Farmland'))
model_name=st.sidebar.selectbox("Select Model",("Prophet","Linear Regression"))



def get_model(model_name,dataset_name,land_type):
    if model_name=='Prophet' and dataset_name=='Estonia' and land_type=='Forest land':
        df_est_forest = pd.read_csv('data/forest_land_estonia.csv')
        df_est_forest[['ds', 'y']] = df_est_forest[['year', 'avg_price_eur']]
        df_est_forest = df_est_forest[['ds', 'y']]
        m = Prophet()
        m.fit(df_est_forest)
        future = m.make_future_dataframe(periods = 20818)     
        forecast = m.predict(future.tail(1461))
        m.plot(forecast)
        fig1 = plot_plotly(m, forecast) 
        st.plotly_chart(fig1) 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        return(st.plotly_chart(fig, use_container_width=True))
    
    elif model_name=='Prophet' and dataset_name=='Estonia' and land_type=='Farmland':
        df_est_farm = pd.read_csv('data/farmland_estonia.csv')
        df_est_farm[['ds', 'y']] = df_est_farm[['year', 'avg_price_eur']]
        df_est_farm = df_est_farm[['ds', 'y']]
        
        m = Prophet()
        m.fit(df_est_farm)
        future = m.make_future_dataframe(periods = 20818)
        forecast = m.predict(future.tail(1461))
        m.plot(forecast)
        fig1 = plot_plotly(m, forecast) 
        st.plotly_chart(fig1) 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        return(st.plotly_chart(fig, use_container_width=True))    


    elif model_name=='Linear Regression' and dataset_name=='Estonia' and land_type=='Forest land and Farmland':
        df = pd.read_csv('data/maaamet_farm_forest_2022.csv')
        X = df[['year',  'number', 'average_area',
       'total_volume_eur', 'price_min', 'price_max', 'price_per_unit_min',
       'price_per_unit_max', 'price_per_unit_median',
       'standard_deviation']]
        y = df['avg_price_eur']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
        lm = LinearRegression()
        lm.fit(X_train,y_train)
        predictions = lm.predict(X_test)
        fig = px.scatter(x=y_test, y=predictions, labels={'x': 'y_test', 'y': 'predictions'})
        return(st.plotly_chart(fig, use_container_width=True))
    
    elif model_name=='Linear Regression' and dataset_name=='France' and land_type=='Forest land and Farmland':
        df = pd.read_csv('data/farm_forest_france.csv')
        df.dropna(inplace=True)
        df = pd.get_dummies(df, columns=['land_type'])
        X = df.drop('price', axis=1)
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(y_test.shape)
        print(X_test.shape)
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        fig = px.scatter(x=y_test, y=predictions, labels={'x': 'y_test', 'y': 'predictions'})
        return(st.plotly_chart(fig, use_container_width=True))


    elif model_name=='Prophet' and dataset_name=='France' and land_type=='Forest land':
        df_fra_forest = pd.read_csv('data/forest_land_france.csv',sep=';')
        df_fra_forest=df_fra_forest[df_fra_forest['country']=='FRA']
        df_fra_forest = df_fra_forest.drop(['indicator', 'Country name', 'country'], axis=1)
        df_fra_forest = df_fra_forest.rename(columns={'Indicator name': 'land_type', 'time': 'year', 'value': 'price'})
        df_fra_forest['land_type']='Forest land'
        df_fra_forest = df_fra_forest.reset_index(drop=True)
        df_fra_forest[['ds', 'y']] = df_fra_forest[['year', 'price']]
        df_fra_forest = df_fra_forest[['ds', 'y']]
        m = Prophet()
        m.fit(df_fra_forest)
        future = m.make_future_dataframe(periods=18992)
        forecast = m.predict(future.tail(1461))

        m.plot(forecast)
        fig1 = plot_plotly(m, forecast) 
        st.plotly_chart(fig1) 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        return(st.plotly_chart(fig, use_container_width=True))
    
    
    elif model_name=='Prophet' and dataset_name=='France' and land_type=='Farmland':
        df_fra_farm = pd.read_csv('data/farmland_france.csv',sep=';')
        df_fra_farm=df_fra_farm[df_fra_farm['country']=='FRA']
        df_fra_farm = df_fra_farm.drop(['indicator', 'Country name', 'country'], axis=1)
        df_fra_farm = df_fra_farm.rename(columns={'Indicator name': 'land_type', 'time': 'year', 'value': 'price'})
        df_fra_farm = df_fra_farm.reset_index(drop=True)
        df_fra_farm[['ds', 'y']] = df_fra_farm[['year', 'price']]
        df_fra_farm = df_fra_farm[['ds', 'y']]
        m = Prophet()
        m.fit(df_fra_farm)
        future = m.make_future_dataframe(periods=16070)
        forecast = m.predict(future.tail(1461))
        m.plot(forecast)
        fig1 = plot_plotly(m, forecast) 
        st.plotly_chart(fig1) 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        return(st.plotly_chart(fig, use_container_width=True))


st.subheader(dataset_name+' '+land_type+' '+model_name+' Model')
get_model(model_name,dataset_name,land_type)
#st.plotly_chart(fig, use_container_width=True)
