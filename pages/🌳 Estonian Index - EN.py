import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
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
st.image("data/landex.png",width=300)
st.title('Estonian Land Index')
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
     <a href='./Estonian_Index_-_EN#figure-total-historical-sales-volume-by-land-type' target='_self'>Total Historical Sales Volume by Land Type</a>
""", unsafe_allow_html=True)

df = pd.read_csv('data/maaamet_farm_forest_2022.csv')
toc.subheader('Figure - Total Historical Sales Volume by Land Type')
fig = px.bar(df, x='year', y='total_volume_eur',
             hover_data=['year', 'avg_price_eur', 'total_volume_eur', 'county', 'region'], color='land_type',
             labels={'avg_price_eur':'Average price (EUR per hectar)'}, height=400)
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
st.plotly_chart(fig, use_container_width=True)
create_paragraph('''The Art Index gives an overview of the rise and fall in the price of art. The price of art has made a noticeable jump in recent years. Interest in investing in art on the art auction market has skyrocketed since the pandemic.''')

# FIGURE - date and volume
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
