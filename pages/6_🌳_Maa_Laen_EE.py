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

st.title('Land Loan Calculator')

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

# User inputs
land_type = st.selectbox("Maatüüp:", 
                         ["Forest land", "Farmland", "Residential", "Commercial", "Industrial"], 
                         key="land_type_select")
selected_region = st.selectbox("Piirkond:", 
                               list(region_county_dict.keys()), 
                               key="region_select")
selected_county = region_county_dict[selected_region]
st.write(f"Selected Maakond: **{selected_county}**")

plot_size = st.number_input("Plot Size (in hectars)", min_value=1.0, step=0.1)
loan_term = st.selectbox("Loan Term (Months)", 
                         list(range(6, 25)), 
                         key="loan_term_select")
payment_frequency = st.selectbox("Makse sagedus", 
                                 ["Monthly", "Quarterly", "Semi-Annual", "Annual"], 
                                 key="payment_frequency_select")
loan_amount = st.number_input("Loan Amount", min_value=1000, step=1000)



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
                "Maatüüp",
                "Piirkond",
                "Maakond",
                "Krundi suurus (hektarites)",
                "Keskmine hind hektari kohta (EUR)",
                "Laenu tähtaeg (kuudes)",
                "Makse sagedus",
                "Maa väärtus (hinnanguline) (EUR)",
                "Maksimaalne laenusumma (60% LTV, 8% aastas) (EUR)",
                "Igakuine makse (EUR)"
            ],
            "Value": [
                land_type,
                selected_region,
                selected_county,
                f"{plot_size:.1f}",
                f"{avg_price_eur:.2f}",
                f"{loan_term}",
                payment_frequency,
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
        pdf_download_link = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="{pdf_file_name}">Download PDF</a>'
        st.markdown(pdf_download_link, unsafe_allow_html=True)

    else:
        st.write("Valitud kombinatsiooni jaoks andmed puuduvad.")
  
    with st.form("user_details_form", clear_on_submit=True):
        name = st.text_input("Nimi")
        email = st.text_input("E-post")
        phone_number = st.text_input("Telefoninumber")
        company_name = st.text_input("Company Nimi")
        submit_button = st.form_submit_button("Saada")

        if submit_button:
        # E-post content
            subject = "Loan Information"
            body = f"Nimi: {name}\nE-post: {email}\nPhone: {phone_number}\nCompany: {company_name}\n\nLoan Details:\n{loan_info_df.to_string(index=False)}"
        
        # E-post parameters - Replace with your actual details
            from_addr = 'info@yourdomain.com'
            to_addr = email
            smtp_user = 'your_smtp_user'
            smtp_password = 'your_smtp_password'

            # Saada email
            mail.util.send_mail(subject, body)
        
st.text("Copyright 2024 Landex.ai")
