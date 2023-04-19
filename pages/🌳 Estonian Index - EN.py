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
from StreamlitHelper import Toc, get_img_with_href, read_df, create_table

st.set_page_config(
    page_title="Land Index",
    page_icon="data/landex.ico",
)

# inject CSS to hide row indexes and style fullscreen button
inject_style_css = """
            <style>
            /*style hide table row index*/
            thead tr th:first-child {display:none}
            tbody th {display:none}
            
            /*style fullscreen button*/
            button[title="View fullscreen"] {
                background-color: #004170cc;
                right: 0;
                color: white;
            }
            button[title="View fullscreen"]:hover {
                background-color:  #004170;
                color: white;
                }
            a { text-decoration:none;}
            </style>
            """
st.markdown(inject_style_css, unsafe_allow_html=True)

def create_paragraph(text):
    st.markdown('<span style="word-wrap:break-word;">' + text + '</span>', unsafe_allow_html=True)
 
toc = Toc()

# TITLE
st.image("data/landex.png",width=200)
st.title('Estonian Land Index')

# Overview
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#overview' target='_self'>Overview</a>
""", unsafe_allow_html=True)
toc.header('Overview')
create_paragraph('''LandEx is a startup company based in Tallinn, Estonia, with a mission to democratize land investment.

They believe that land is a great asset class that provides high-yield and low-risk returns due to its economic fundamentals, and therefore it should be accessible to anyone.

The company aims to become the largest land investment platform in Europe, providing a solution that was not previously available in the market.
The founders of LandEx, Kamel and Randy, were dissatisfied with the investment opportunities available for land investments. They found it challenging to source and manage land, and the minimum investment required was often in the thousands of euros, making it difficult for many people to access this type of investment.
As a result, they created a digital platform to provide everyone with the opportunity to invest in land, which they launched in September 2021.
LandEx is the first crowdfunding land investment platform in Europe, offering investors an opportunity to invest in land projects with a low minimum investment.

The platform enables investors to browse through a range of investment opportunities, choose the projects they want to invest in, and invest in just a few clicks. LandEx also provides investors with full transparency and control over their investments, including tracking the progress of the projects in real-time.

With LandEx's innovative and user-friendly platform, investing in land has never been more accessible. The company's mission to democratize land investment is an exciting development for those interested in investing in this asset class, providing a low-risk and high-yield investment option that was previously inaccessible to many.''')


# FIGURE - Historical Sales Volume by Land Type
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-historical-sales-volume-by-land-type' target='_self'>Historical Sales Volume by Land Type</a>
""", unsafe_allow_html=True)

df = pd.read_csv('data/maaamet_farm_forest_2022.csv')
toc.subheader('Figure - Historical Sales Volume by Land Type')
fig = px.bar(df, x='year', y='total_volume_eur',
             hover_data=['year', 'avg_price_eur', 'total_volume_eur', 'county', 'region'], color='land_type',
             labels={'avg_price_eur':'Average price (EUR per hectar)'}, height=400)
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
st.plotly_chart(fig, use_container_width=True)

# FIGURE - Relative price of land by region - point of time data (2020)
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-relative-price-of-land-by-region-point-of-time-data-2020' target='_self'>Relative price of land by region - point of time data (2020)</a>
""", unsafe_allow_html=True)
toc.subheader('Figure - Relative price of land by region - point of time data (2020)')
fig = px.treemap(df, path=['land_type', 'county', 'region'], values='total_volume_eur',
                  color='avg_price_eur', hover_data=['region'],
                  color_continuous_scale='RdBu',
                  color_continuous_midpoint=np.average(df['avg_price_eur'], weights=df['total_volume_eur']))
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
st.plotly_chart(fig, use_container_width=True)
create_paragraph('''Based on the Based on the data available up until 2020, we can observe the following trends:

Price Range - The prices for land in Hiiumaa, a remote island in Estonia, ranged from around 2400 EUR per hectare at the lower end to some of the highest prices.

Land Type - Forest land, on average, was more expensive than farm land.

These observations provide valuable insights into the current state of the land market and can help inform decision-making for those looking to buy or sell land.
''')

#FIGURE - Average price vs average plot size
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-average-price-vs-average-plot-size' target='_self'>Average price vs average plot size</a>
""", unsafe_allow_html=True)
toc.subheader('Figure - Average price vs average plot size')
fig = px.scatter(df, x="average_area", y="avg_price_eur", color="county",
                 size='total_volume_eur', hover_data=['region'])
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
st.plotly_chart(fig, use_container_width=True)

#FIGURE - Relationship between Land Area and Transaction Volume
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-average-price-vs-average-plot-size' target='_self'>Relationship between Land Area and Transaction Volume</a>
""", unsafe_allow_html=True)

toc.subheader('Figure - Relationship between Land Area and Transaction Volume')
fig = px.scatter(df, x="average_area", y="total_volume_eur", color="land_type")
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
st.plotly_chart(fig, use_container_width=True)

#FIGURE - Forest land Index
index_df = pd.read_csv('data/total_land_index.csv')
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-forest-land-index' target='_self'>Forest land Index</a>
""", unsafe_allow_html=True)

toc.subheader('Figure - Forest land Index')
forest_index_fig = px.area(index_df, x="year", y="forest_avg_eur", color_discrete_sequence=['green'])
forest_index_fig.update_yaxes(title_text='Average price in EUR per hectar, forest land')
forest_index_fig.update_xaxes(title_text='Year')
st.plotly_chart(forest_index_fig, use_container_width=True)

#FIGURE - Farm land Index
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-farm-land-index' target='_self'>Farm land Index</a>
""", unsafe_allow_html=True)

toc.subheader('Figure - Farm land Index')
farm_index_fig = px.area(index_df, x="year", y="farmland_avg_eur", color_discrete_sequence=['orange']) 
farm_index_fig.update_yaxes(title_text='Average price in EUR per hectar, farm land')
farm_index_fig.update_xaxes(title_text='Year')
st.plotly_chart(farm_index_fig, use_container_width=True)

#FIGURE - All Types of Land Index
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-all-types-of-land-index' target='_self'>All Types of Land Index</a>
""", unsafe_allow_html=True)

toc.subheader('Figure - All Types of Land Index')
total_index_fig = px.area(index_df, x="year", y="all_average_eur")
total_index_fig.update_yaxes(title_text='Average price in EUR per hectar, all lands')
total_index_fig.update_xaxes(title_text='Year')
st.plotly_chart(total_index_fig, use_container_width=True)

#FIGURE - Land Volume Index
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-land-volume-index' target='_self'>Land Volume Index</a>
""", unsafe_allow_html=True)

country_df = df.groupby(['land_type', 'year', 'county'])['total_volume_eur'].mean()
index_df = df.groupby(['year'])['total_volume_eur'].mean()
index_df.columns = ['country_index']
index_df = country_df.reset_index()
toc.subheader('Figure -Land Volume Index')
country_fig = px.area(index_df, x="year", y="total_volume_eur", color="county", line_group="land_type")
st.plotly_chart(country_fig, use_container_width=True)

#FIGURE - Land Price Prediction
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-land-price-prediction' target='_self'>Land Price Prediction</a>
""", unsafe_allow_html=True)
toc.subheader('Figure - Land Price Prediction')

st.write("""
# Explore Different Results
""")

#st.sidebar.image("data/landex.png", use_column_width=True)

dataset_name=st.selectbox("Select Dataset",("Estonia","France"))
land_type=st.selectbox("Select type of land",("Forest land","Farmland",'Forest land and Farmland'))
model_name=st.selectbox("Select Model",("Prophet","Linear Regression"))



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
