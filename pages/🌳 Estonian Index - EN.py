import streamlit as st
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
The founders of LandEx, Kamel, and Randy, were dissatisfied with the investment opportunities available for land investments. They found it challenging to source and manage land, and the minimum investment required was often in the thousands of euros, making it difficult for many people to access this type of investment.
As a result, they created a digital platform to provide everyone with the opportunity to invest in land, which they launched in September 2021.
LandEx is the first crowdfunding land investment platform in Europe, offering investors an opportunity to invest in land projects with a low minimum investment.

The platform enables investors to browse a range of investment opportunities, choose the projects they want to invest in, and invest in just a few clicks. LandEx also provides investors full transparency and control over their investments, including tracking the progress of the projects in real time.

With LandEx's innovative and user-friendly platform, investing in the land has never been more accessible. The company's mission to democratize land investment is an exciting development for those interested in investing in this asset class, providing a low-risk and high-yield investment option that was previously inaccessible to many.''')


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
create_paragraph('''The Land Index provides an overview of the fluctuations in the prices of farmland and forest land.
It's noteworthy that these prices have experienced a noticeable increase in recent years.''')

# FIGURE - Relative price of land by region - point of time data (2022)
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-relative-price-of-land-by-region-point-of-time-data-2022' target='_self'>Relative price of land by region - point of time data (2022)</a>
""", unsafe_allow_html=True)
toc.subheader('Figure - Relative price of land by region - point of time data (2022)')
fig = px.treemap(df, path=['land_type', 'county', 'region'], values='total_volume_eur',
                  color='avg_price_eur', hover_data=['region'],
                  color_continuous_scale='RdBu',
                  color_continuous_midpoint=np.average(df['avg_price_eur'], weights=df['total_volume_eur']))
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
st.plotly_chart(fig, use_container_width=True)
create_paragraph('''Based on the data available up until 2022, we can observe the following trends:

Price Range - The prices for land in Hiiumaa, a remote island in Estonia, ranged from around 2400 EUR per hectare at the lower end to some of the highest prices.

Land Type - On average, forest land was more expensive than farmland.

These observations provide valuable insights into the current state of the land market and can help inform decision-making for those looking to buy or sell land.''')

#FIGURE - Average price vs average plot size
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-average-price-vs-average-plot-size' target='_self'>Average price vs average plot size</a>
""", unsafe_allow_html=True)
toc.subheader('Figure - Average price vs average plot size')
fig = px.scatter(df, x="average_area", y="avg_price_eur", color="county",
                 size='total_volume_eur', hover_data=['region'])
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
st.plotly_chart(fig, use_container_width=True)
create_paragraph('''The provided graph visualizes the relationship between two variables, average_area, and avg_price_eur, for various counties within a given region. Each point on the graph represents a county, with the color of the point indicating the specific county and the size of the point representing the total sales volume in euros.

As we move from the top left to the bottom right, we can see a downward trend in the points, indicating that larger plots tend to have lower average prices and smaller plots tend to have higher average prices.''')

#FIGURE - Relationship between Land Area and Transaction Volume
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-relationship-between-land-area-and-transaction-volume' target='_self'>Relationship between Land Area and Transaction Volume</a>
""", unsafe_allow_html=True)

toc.subheader('Figure - Relationship between Land Area and Transaction Volume')
fig = px.scatter(df, x="average_area", y="total_volume_eur", color="land_type")
fig.update_layout(
    xaxis_title="Average Area (hectares)",
    margin=dict(l=5, r=5, t=5, b=5)
)
st.plotly_chart(fig, use_container_width=True)
create_paragraph('''The largest number of transactions were for plots that were approximately 10 hectares in size.
''')
#FIGURE - Forest land Index
index_df = pd.read_csv('data/total_land_index.csv')
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-forest-land-index' target='_self'>Forest land Index</a>
""", unsafe_allow_html=True)

toc.subheader('Figure - Forest land Index')
forest_index_fig = px.area(index_df, x="year", y="forest_avg_eur", color_discrete_sequence=['green'])
forest_index_fig.update_yaxes(title_text='The average price in EUR per hectare, forest land')
forest_index_fig.update_xaxes(title_text='Year')
st.plotly_chart(forest_index_fig, use_container_width=True)
create_paragraph('''The average price of forest land grew from about 1,000 EUR per hectare in 2000 to about 8,000 EUR per hectare in 2022.
''')
#FIGURE - Farmland Index
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-farmland-index' target='_self'>Farmland Index</a>
""", unsafe_allow_html=True)

toc.subheader('Figure - Farmland Index')
farm_index_fig = px.area(index_df, x="year", y="farmland_avg_eur", color_discrete_sequence=['orange']) 
farm_index_fig.update_yaxes(title_text='The average price in EUR per hectare, farmland')
farm_index_fig.update_xaxes(title_text='Year')
st.plotly_chart(farm_index_fig, use_container_width=True)
create_paragraph('''The average price of forest land grew from about 280 EUR per hectare in 2000 to about 4,800 EUR per hectare in 2022.
''')
#FIGURE - Farmland and Forest Land Total Index
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#figure-farmland-and-forest-land-total-index' target='_self'>Farmland and Forest Land Total Index</a>
""", unsafe_allow_html=True)

toc.subheader('Figure - Farmland and Forest Land Total Index')
total_index_fig = px.area(index_df, x="year", y="all_average_eur")
total_index_fig.update_yaxes(title_text='The average price in EUR per hectare, Farmland and Forest Land')
total_index_fig.update_xaxes(title_text='Year')
st.plotly_chart(total_index_fig, use_container_width=True)
create_paragraph('''The average price of forest land grew from about 730 EUR per hectare in 2000 to about 6,600 EUR per hectare in 2022.
''')
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
create_paragraph('''The total volume of transactions of all lands grew from about 4 million EUR in 2000 to about 58m EUR in 2022.
''')

#Top performers - Price Performance
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#top-performers-price-performance-county' target='_self'>Top performers - Price Performance (County)</a>
""", unsafe_allow_html=True)
toc.subheader('Top performers - Price Performance (County)')
df=pd.read_csv('data/maaamet_farm_forest_2022.csv')
country_df = df.groupby(['land_type', 'year', 'county'])['total_volume_eur'].mean()
index_df = df.groupby(['year'])['total_volume_eur'].mean()
index_df.columns = ['country_index']
index_df = country_df.reset_index()
sorted_data =index_df.sort_values(by='total_volume_eur', ascending=False).head(5)
st.table(sorted_data)

#FIGURE - Land Price Prediction
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EN#land-price-prediction' target='_self'>Land Price Prediction</a>
""", unsafe_allow_html=True)
toc.subheader('Land Price Prediction')


#Estonia Forest Land Prophet Model - Estonian Forest Land Prediction
st.subheader("""
     Estonia Forest Land Prophet Model - Estonian Forest Land Prediction
""")
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
st.plotly_chart(fig, use_container_width=True)

#Estonia Farmland Prophet Model - Estonian Farmland Prediction
st.subheader("""
     Estonia Farmland Prophet Model - Estonian Farmland Prediction
""")
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
st.plotly_chart(fig, use_container_width=True)

#Estonia Forest land and Farmland Prophet Model - Estonian Forest Land and Farmland Prediction
st.subheader("""
     Estonia Forest land and Farmland Prophet Model - Estonian Forest Land and Farmland Prediction
""")
df_est_for_farm = pd.read_csv('data/maaamet_farm_forest_2022.csv')
df_est_for_farm[['ds', 'y']] = df_est_for_farm[['year', 'avg_price_eur']]
df_est_for_farm = df_est_for_farm[['ds', 'y']]
m = Prophet()
m.fit(df_est_for_farm)
future = m.make_future_dataframe(periods = 20818)
forecast = m.predict(future.tail(1461))
m.plot(forecast)
fig1 = plot_plotly(m, forecast) 
st.plotly_chart(fig1) 
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
st.plotly_chart(fig, use_container_width=True)

