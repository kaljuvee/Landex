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

# DATA_DIR = 'data'

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
st.image("data//landex.png",width=200)
st.title('Eesti maaindeks')

# Overview
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#levaade' target='_self'>Ülevaade</a>
""", unsafe_allow_html=True)
toc.header('Ülevaade')
create_paragraph('''LandEx on Tallinnas asuv idufirma, mille eesmärk on demokratiseerida maainvesteeringuid.

Nad usuvad, et maa on suurepärane varaklass, mis pakub oma majanduslike põhialuste tõttu kõrget tootlust ja madalat riski ning seetõttu peaks see olema kõigile kättesaadav.

Ettevõtte eesmärk on saada suurimaks maainvesteeringute platvormiks Euroopas, pakkudes lahendust, mida varem turul ei olnud võimalik saada.
LandExi asutajad Kamel ja Randy olid rahulolematud maainvesteeringute investeerimisvõimalustega. Nad leidsid, et maa leidmine ja haldamine on keeruline ning nõutav minimaalne investeering on sageli tuhandetes eurodes, mistõttu on paljudel inimestel raske seda liiki investeeringutele ligi pääseda.
Selle tulemusena lõid nad digitaalse platvormi, et pakkuda kõigile võimalust investeerida maasse, mille nad käivitasid 2021. aasta septembris.
LandEx on esimene ühisrahastuse maainvesteeringute platvorm Euroopas, mis pakub investoritele võimalust investeerida maaprojektidesse madala minimaalse investeeringuga.

Platvorm võimaldab investoritel sirvida erinevaid investeerimisvõimalusi, valida projektid, millesse nad soovivad investeerida, ja investeerida vaid paari klikiga. LandEx pakub investoritele ka täielikku läbipaistvust ja kontrolli oma investeeringute üle, sealhulgas projektide edenemise jälgimist reaalajas.

LandExi uuendusliku ja kasutajasõbraliku platvormi abil ei ole maasse investeerimine kunagi varem olnud kättesaadavam. Ettevõtte missioon demokratiseerida maainvesteeringuid on põnev areng neile, kes on huvitatud sellesse varaklassi investeerimisest, pakkudes madala riskiga ja kõrge tootlusega investeerimisvõimalust, mis varem oli paljudele kättesaamatu.''')


# FIGURE - Historical Sales Volume by Land Type
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-ajalooline-m-gimaht-maat-pide-kaupa' target='_self'>Ajalooline müügimaht maatüüpide kaupa</a>
""", unsafe_allow_html=True)

df = pd.read_csv('data/maaamet_farm_forest_2022.csv')
toc.subheader('Joonis - ajalooline müügimaht maatüüpide kaupa')
fig = px.bar(df, x='year', y='total_volume_eur',
             hover_data=['year', 'avg_price_eur', 'total_volume_eur', 'county', 'region'], color='land_type',
             labels={'avg_price_eur':'Average price (EUR per hectar)'}, height=400)
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
st.plotly_chart(fig, use_container_width=True)
create_paragraph('''Maaindeks annab ülevaate põllu- ja metsamaa hinna kõikumisest.
Tähelepanuväärne on, et need hinnad on viimastel aastatel märgatavalt tõusnud.''')

# FIGURE - Relative price of land by region - point of time data (2022)
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-maa-suhteline-hind-piirkonniti-hetkeandmed-2022' target='_self'>Maa suhteline hind piirkonniti - ajahetke andmed (2022)</a>
""", unsafe_allow_html=True)
toc.subheader('Joonis - Maa suhteline hind piirkonniti - hetkeandmed (2022)')
fig = px.treemap(df, path=['land_type', 'county', 'region'], values='total_volume_eur',
                  color='avg_price_eur', hover_data=['region'],
                  color_continuous_scale='RdBu',
                  color_continuous_midpoint=np.average(df['avg_price_eur'], weights=df['total_volume_eur']))
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
st.plotly_chart(fig, use_container_width=True)
create_paragraph('''Kuni 2022. aastani kättesaadavate andmete põhjal võib täheldada järgmisi suundumusi:

Hinnavahemik - Hiiumaal, Eesti kaugel asuval saarel asuva maa hinnad ulatusid umbes 2400 eurost hektari kohta alumisest otsast kuni mõne kõrgeima hinnani.

Maatüüp - Keskmiselt oli metsamaa kallim kui põllumaa.

Need tähelepanekud annavad väärtusliku ülevaate maaturu praegusest olukorrast ja võivad aidata otsuste tegemisel neile, kes soovivad maad osta või müüa.''')

#FIGURE - Average price vs average plot size
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-keskmine-hind-vs-keskmine-maat-ki-suurus' target='_self'>Keskmine hind vs. keskmine maatüki suurus</a>
""", unsafe_allow_html=True)
toc.subheader('Joonis - Keskmine hind vs. keskmine maatüki suurus')
fig = px.scatter(df, x="average_area", y="avg_price_eur", color="county",
                 size='total_volume_eur', hover_data=['region'])
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
st.plotly_chart(fig, use_container_width=True)
create_paragraph('''Esitatud graafik visualiseerib kahe muutuja, keskmine_pindala ja keskmine_hind_eur, vahelist seost erinevate maakondade kohta antud piirkonnas. Iga punkt graafikul tähistab maakonda, kusjuures punkti värv näitab konkreetset maakonda ja punkti suurus tähistab müügi kogumahtu eurodes.

Liikudes vasakult ülevalt paremale allapoole, näeme punktide langustendentsi, mis näitab, et suuremate maatükkide keskmine hind on pigem madalam ja väiksemate maatükkide keskmine hind on pigem kõrgem.''')

#FIGURE - Relationship between Land Area and Transaction Volume
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-maa-ala-ja-tehingumahu-vaheline-seos' target='_self'>Maa-ala ja tehingumahu vaheline seos</a>
""", unsafe_allow_html=True)

toc.subheader('Joonis - Maa-ala ja tehingumahu vaheline seos')
fig = px.scatter(df, x="average_area", y="total_volume_eur", color="land_type")
fig.update_layout(
    xaxis_title="Average Area (hectares)",
    margin=dict(l=5, r=5, t=5, b=5)
)
st.plotly_chart(fig, use_container_width=True)
create_paragraph('''Kõige rohkem tehinguid tehti ligikaudu 10 hektari suuruste maatükkidega.
''')
#FIGURE - Forest land Index
index_df = pd.read_csv('data/total_land_index.csv')
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-metsamaa-indeks' target='_self'>Metsamaa indeks</a>
""", unsafe_allow_html=True)

toc.subheader('Joonis - Metsamaa indeks')
forest_index_fig = px.area(index_df, x="year", y="forest_avg_eur", color_discrete_sequence=['green'])
forest_index_fig.update_yaxes(title_text='Keskmine hind eurodes hektari kohta, metsamaa')
forest_index_fig.update_xaxes(title_text='Aasta')
st.plotly_chart(forest_index_fig, use_container_width=True)
create_paragraph('''Metsamaa keskmine hind kasvas umbes 1000 eurolt hektari kohta 2000. aastal umbes 8000 euroni hektari kohta 2022. aastal.
''')
#FIGURE - Farmland Index
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-p-llumajandusmaa-indeks' target='_self'>põllumajandusmaa indeks</a>
""", unsafe_allow_html=True)

toc.subheader('Joonis - põllumajandusmaa indeks')
farm_index_fig = px.area(index_df, x="year", y="farmland_avg_eur", color_discrete_sequence=['orange']) 
farm_index_fig.update_yaxes(title_text='Keskmine hind eurodes hektari kohta, põllumajandusmaa')
farm_index_fig.update_xaxes(title_text='Aasta')
st.plotly_chart(farm_index_fig, use_container_width=True)
create_paragraph('''Metsamaa keskmine hind kasvas umbes 280 eurolt hektari kohta 2000. aastal umbes 4800 euroni hektari kohta 2022. aastal.
''')
#FIGURE - Farmland and Forest Land Total Index
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-p-llu-ja-metsamaa-koguindeks' target='_self'>Põllu- ja metsamaa Koguindeks</a>
""", unsafe_allow_html=True)

toc.subheader('Joonis - Põllu- ja metsamaa koguindeks')
total_index_fig = px.area(index_df, x="year", y="all_average_eur")
total_index_fig.update_yaxes(title_text='Keskmine hind eurodes hektari kohta, põllu- ja metsamaa')
total_index_fig.update_xaxes(title_text='Aasta')
st.plotly_chart(total_index_fig, use_container_width=True)
create_paragraph('''Metsamaa keskmine hind kasvas umbes 730 eurolt hektari kohta 2000. aastal umbes 6600 euroni hektari kohta 2022. aastal.
''')
#FIGURE - Land Volume Index
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-maa-mahuindeks' target='_self'>Maa mahuindeks</a>
""", unsafe_allow_html=True)

country_df = df.groupby(['land_type', 'year', 'county'])['total_volume_eur'].mean()
index_df = df.groupby(['year'])['total_volume_eur'].mean()
index_df.columns = ['country_index']
index_df = country_df.reset_index()
toc.subheader('Joonis - Maa mahuindeks')
country_fig = px.area(index_df, x="year", y="total_volume_eur", color="county", line_group="land_type")
st.plotly_chart(country_fig, use_container_width=True)
create_paragraph('''Kõigi maade tehingute kogumaht kasvas umbes 4 miljonilt eurolt 2000. aastal umbes 58 miljoni euroni 2022. aastal.
''')

#Top performers - Price Performance (County)
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#tipptegijad-hinnatulemused-maakond' target='_self'>Tipptegijad - Hinnatulemused (maakond)</a>
""", unsafe_allow_html=True)
toc.subheader('Tipptegijad - Hinnatulemused (maakond)')
df=pd.read_csv('data/maaamet_farm_forest_2022.csv')
country_df = df.groupby(['land_type', 'year', 'county'])['total_volume_eur'].mean()
index_df = df.groupby(['year'])['total_volume_eur'].mean()
index_df.columns = ['country_index']
index_df = country_df.reset_index()
sorted_data =index_df.sort_values(by='total_volume_eur', ascending=False).head(5)
st.table(sorted_data)

#FIGURE - Land Price Prediction
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#maa-hinna-prognoosimine' target='_self'>Maa hinna prognoosimine</a>
""", unsafe_allow_html=True)
toc.subheader('Maa hinna prognoosimine')


#Estonia Forest Land Prophet Model - Estonian Forest Land Prediction
st.subheader("""
     Eesti metsamaa prohveti mudel - Eesti metsamaa prognoosimine
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
     Eesti põllumaa prohveti mudel - Eesti põllumaa prognoosimine
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
     Eesti metsamaa ja põllumaa prohveti mudel - Eesti metsamaa ja põllumaa prognoosimine
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

