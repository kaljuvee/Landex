import streamlit as st
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import base64
from io import BytesIO
from util import file_util

DATA_PATH = 'data/maaamet_farm_forest_2022.csv'
DICT_PATH = 'data/region_county_dict.csv'

st.title('Maalaenu kalkulaator')

# Load the region-county mapping from a CSV file
@st.cache_data
def load_region_county_dict():
    df = pd.read_csv(DICT_PATH)
    return dict(zip(df['Region'], df['County']))

region_county_dict = load_region_county_dict()

# Function to calculate annuity payment
def calculate_annuity_payment(principal, annual_interest_rate, periods):
    monthly_interest_rate = annual_interest_rate / 12
    return principal * monthly_interest_rate / (1 - (1 + monthly_interest_rate)**-periods)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

land_type_dict = {
    "Forest land": "Metsamaa",
    "Farmland": "Põllumaa",
    "Residential": "Elamumaa",
    "Commercial": "Kaubandusmaa",
    "Industrial": "Tööstusmaa"
}

payment_frequency_dict = {
    "Monthly": "Igakuine",
    "Quarterly": "Kvartaalne",
    "Semi-Annual": "Poolaastane",
    "Annual": "Aastane"
}
# User inputs
# User selects land type in Estonian
selected_land_type_ee = st.selectbox("Maa liik:", list(land_type_dict.values()),
                    key="land_type_select")

# Reverse lookup to get the English equivalent
land_type = [key for key, value in land_type_dict.items() if value == selected_land_type_ee][0]

selected_region = st.selectbox("Vald:", 
                               list(region_county_dict.keys()), 
                               key="region_select")
selected_county = region_county_dict[selected_region]
st.write(f"Valitud maakond: **{selected_county}**")

plot_size = st.number_input("Maatüki suurus (hektarites):", min_value=1.0, step=0.1)
loan_term = st.selectbox("Laenu pikkus (kuudes):", 
                         list(range(6, 25)), 
                         key="loan_term_select")

# User selects payment frequency in Estonian
selected_payment_frequency_ee = st.selectbox("Maksete sagedus:", list(payment_frequency_dict.values()), key="payment_frequency_select")

# Reverse lookup to get the English equivalent
payment_frequency = [key for key, value in payment_frequency_dict.items() if value == selected_payment_frequency_ee][0]

loan_amount = st.number_input("Laenusumma:", min_value=1000, step=1000)



# 'Get Quote' button action
if st.button("Arvuta"):
    # Fetch avg_price_eur based on land_type, county, region
    avg_price_eur = df[(df['land_type'] == land_type) & (df['county'] == selected_county) & (df['region'] == selected_region)]['avg_price_eur'].mean()
    
    if not np.isnan(avg_price_eur):
        property_value = avg_price_eur * plot_size
        loan_to_value = 0.6  # 60%
        interest_rate = 0.08  # 8%
        max_loan_amount = property_value * loan_to_value

        # Prepare data for DataFrame
        data = {
            "Item": [
                "Maa liik",
                "Piirkond",
                "Maakond",
                "Krundi suurus (hektarites)",
                "Keskmine hind hektari kohta (EUR)",
                "Laenu periood (kuudes)",
                "Maksete sagedus",
                "Maa väärtus (hinnanguline) (EUR)",
                "Maksimaalne laenusumma (60% LTV, 8% aastas) (EUR)",
                "Igakuine tagasimakse (EUR)"
            ],
            "Value": [
                selected_land_type_ee,
                selected_region,
                selected_county,
                f"{plot_size:.1f}",
                f"{avg_price_eur:.2f}",
                f"{loan_term}",
                selected_payment_frequency_ee,
                f"{property_value:.2f}",
                f"{max_loan_amount:.2f}",
                ""  # Placeholder for monthly payment
            ]
        }

        # Annuity payment calculation
        if payment_frequency == "Monthly":
            periods = loan_term
            monthly_payment = calculate_annuity_payment(max_loan_amount, interest_rate, periods)
            data["Value"][9] = f"{monthly_payment:.2f}"  # Ensure the index is correct for monthly payment

        # Create and display DataFrame
        loan_info_df = pd.DataFrame(data)

        # Convert DataFrame to HTML and use markdown to display it with styling
        loan_info_html = loan_info_df.to_html(index=False, escape=False)
        loan_info_html = loan_info_html.replace('<th>', '<th style="font-weight: bold; background-color: #f0f0f0; text-align: left;">')
        st.markdown(loan_info_html, unsafe_allow_html=True)

        pdf_file_name = 'loan_info.pdf'
        file_util.create_pdf(loan_info_df, pdf_file_name)

    # Create download link for the PDF
        with open(pdf_file_name, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_download_link = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="{pdf_file_name}">Laadi alla (PDF)</a>'
        st.markdown(pdf_download_link, unsafe_allow_html=True)

    else:
        st.write("Valitud kombinatsiooni jaoks andmed puuduvad.")
  
    with st.form("user_details_form", clear_on_submit=True):
        name = st.text_input("Nimi:")
        email = st.text_input("E-post:")
        phone_number = st.text_input("Telefoninumber:")
        company_name = st.text_input("Ettevõtte nimi:")
        comments = st.text_input("Lisainfo:")
        submit_button = st.form_submit_button("Saada päring")

        if submit_button:
        # E-post content
            subject = "Loan Information"
            body = f"Nimi: {name}\nE-post: {email}\nPhone: {phone_number}\nCompany: {company_name}\n\nLoan Details:\n{loan_info_df.to_string(index=False)}"
        
            # Saada email
            mail.util.send_mail(subject, body)
        
st.text("Copyright 2024 Landex.ai")
