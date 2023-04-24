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
st.title('Eesti maaindeks')

# Overview
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#levaade' target='_self'>Ülevaade</a>
""", unsafe_allow_html=True)
toc.header('Ülevaade')
create_paragraph('''LandEx on Tallinnas asuv idufirma, mille eesmärk on demokratiseerida maainvesteeringuid.

Nad usuvad, et maa on suurepärane varaklass, mis pakub oma majanduslike põhialuste tõttu kõrget tootlust ja madalat riski ning seetõttu peaks see olema kõigile kättesaadav.

Ettevõtte eesmärk on saada suurimaks maainvesteeringute platvormiks Euroopas, pakkudes lahendust, mida varem turul ei olnud võimalik saada.
LandExi asutajad Kamel ja Randy olid rahulolematud maainvesteeringute investeerimisvõimalustega. Nad leidsid, et maa leidmine ja haldamine on keeruline ning nõutav minimaalne investeering on sageli tuhandete eurode suurune, mistõttu on paljudel inimestel raske seda liiki investeeringuid teha.
Selle tulemusena lõid nad digitaalse platvormi, et pakkuda kõigile võimalust investeerida maasse, mille nad käivitasid 2021. aasta septembris.
LandEx on esimene ühisrahastuse maainvesteeringute platvorm Euroopas, mis pakub investoritele võimalust investeerida maaprojektidesse madala minimaalse investeeringuga.

Platvorm võimaldab investoritel sirvida erinevaid investeerimisvõimalusi, valida projektid, millesse nad soovivad investeerida, ja investeerida vaid paari klikiga. LandEx pakub investoritele ka täielikku läbipaistvust ja kontrolli oma investeeringute üle, sealhulgas projektide edenemise jälgimist reaalajas.

LandExi uuendusliku ja kasutajasõbraliku platvormi abil ei ole maasse investeerimine kunagi varem olnud kättesaadavam. Ettevõtte missioon demokratiseerida maainvesteeringuid on põnev areng neile, kes on huvitatud sellesse varaklassi investeerimisest, pakkudes madala riskiga ja kõrge tootlusega investeerimisvõimalust, mis varem oli paljudele kättesaamatu.

Translated with www.DeepL.com/Translator (free version)''')


# FIGURE - Historical Sales Volume by Land Type
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-ajalooline-m-gimaht-maat-pide-kaupa' target='_self'>Ajalooline müügimaht maatüüpide kaupa</a>
""", unsafe_allow_html=True)

df = pd.read_csv('data/maaamet_farm_forest_2022.csv')
toc.subheader('Joonis - Ajalooline müügimaht maatüüpide kaupa')
fig = px.bar(df, x='year', y='total_volume_eur',
             hover_data=['year', 'avg_price_eur', 'total_volume_eur', 'county', 'region'], color='land_type',
             labels={'avg_price_eur':'Average price (EUR per hectar)'}, height=400)
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
st.plotly_chart(fig, use_container_width=True)
create_paragraph('''Maaindeks annab ülevaate põllu- ja metsamaa hinna kõikumisest.
Tähelepanuväärne on, et need hinnad on viimastel aastatel märgatavalt tõusnud.''')

# FIGURE - Relative price of land by region - point of time data (2020)
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-maa-suhteline-hind-piirkonniti-hetkeandmed-2020' target='_self'>Maa suhteline hind piirkonniti - hetkeandmed (2020)</a>
""", unsafe_allow_html=True)
toc.subheader('Joonis - Maa suhteline hind piirkonniti - hetkeandmed (2020)')
fig = px.treemap(df, path=['land_type', 'county', 'region'], values='total_volume_eur',
                  color='avg_price_eur', hover_data=['region'],
                  color_continuous_scale='RdBu',
                  color_continuous_midpoint=np.average(df['avg_price_eur'], weights=df['total_volume_eur']))
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
st.plotly_chart(fig, use_container_width=True)
create_paragraph('''Kuni 2020. aastani kättesaadavate andmete põhjal võib täheldada järgmisi suundumusi:

Hinnavahemik - Hiiumaal, Eesti kaugel asuval saarel, varieerusid maa hinnad alates umbes 2400 eurost hektari kohta alumisest otsast kuni mõningate kõrgeimate hindadeni.

Maatüüp - Metsamaa oli keskmiselt kallim kui põllumaa.

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
create_paragraph('''Esitatud graafik visualiseerib kahe muutuja, keskmine_pindala ja keskmine_hind_eur, vahelist seost erinevate maakondade kohta antud piirkonnas. Iga punkt graafikul tähistab maakonda, kusjuures punkti värv näitab konkreetset maakonda ja punkti suurus tähistab müügi kogumahtu eurodes.''')

#FIGURE - Relationship between Land Area and Transaction Volume
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-maa-ala-ja-tehingumahu-vaheline-seos' target='_self'>Maa-ala ja tehingumahu vaheline seos</a>
""", unsafe_allow_html=True)

toc.subheader('Joonis - Maa-ala ja tehingumahu vaheline seos')
fig = px.scatter(df, x="average_area", y="total_volume_eur", color="land_type")
fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
st.plotly_chart(fig, use_container_width=True)

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

#FIGURE - Farmland Index
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-p-llumaa-indeks' target='_self'>Põllumaa indeks</a>
""", unsafe_allow_html=True)

toc.subheader('Joonis - Põllumaa indeks')
farm_index_fig = px.area(index_df, x="year", y="farmland_avg_eur", color_discrete_sequence=['orange']) 
farm_index_fig.update_yaxes(title_text='Keskmine hind eurodes hektari kohta, põllumajandusmaa')
farm_index_fig.update_xaxes(title_text='Aasta')
st.plotly_chart(farm_index_fig, use_container_width=True)

#FIGURE - All Types of Land Index
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-k-ik-maat-bid-indeks' target='_self'>Kõik maatüübid Indeks</a>
""", unsafe_allow_html=True)

toc.subheader('Joonis - Kõik maatüübid Indeks')
total_index_fig = px.area(index_df, x="year", y="all_average_eur")
total_index_fig.update_yaxes(title_text='Keskmine hind eurodes hektari kohta, kõik maad')
total_index_fig.update_xaxes(title_text='Aasta')
st.plotly_chart(total_index_fig, use_container_width=True)

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

#FIGURE - Land Price Prediction
st.sidebar.markdown("""
     <a href='./Estonian_Index_-_EE#joonis-maa-hinna-prognoos' target='_self'>Maa hinna prognoos</a>
""", unsafe_allow_html=True)
toc.subheader('Joonis - Maa hinna prognoos')

st.write("""
# Uurige erinevaid tulemusi
""")

#st.sidebar.image("data/landex.png", use_column_width=True)

dataset_name=st.selectbox("Valige andmekogum",("Eesti", "Prantsusmaa"))
land_type=st.selectbox("Valige maa liik",("Metsamaa", "Põllumaa",'Metsamaa ja põllumaa'))
model_name=st.selectbox("Valige mudel",("Prophet","Linear Regression"))



def get_model(model_name,dataset_name,land_type):
    if model_name=='Prophet' and dataset_name=='Eesti' and land_type=='Metsamaa':
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
    
    elif model_name=='Prophet' and dataset_name=='Eesti' and land_type=='Põllumaa':
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


    elif model_name=='Linear Regression' and dataset_name=='Eesti' and land_type=='Metsamaa ja Põllumaa':
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
    
    elif model_name=='Linear Regression' and dataset_name=='Prantsusmaa' and land_type=='Metsamaa ja Põllumaa':
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


    elif model_name=='Prophet' and dataset_name=='Prantsusmaa' and land_type=='Metsamaa':
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
    
    
    elif model_name=='Prophet' and dataset_name=='Prantsusmaa' and land_type=='Põllumaa':
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


st.subheader(dataset_name+' '+land_type+' '+model_name+' Mudel')
get_model(model_name,dataset_name,land_type)
#st.plotly_chart(fig, use_container_width=True)
