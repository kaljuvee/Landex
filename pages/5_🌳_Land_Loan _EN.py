import streamlit as st
import pandas as pd
import numpy as np

DATA_PATH = 'data/maaamet_farm_forest_2022.csv'
DICT_PATH = 'data/region_county_dict.csv'

st.title('Land Value and Loan Calculator')

# Load the region-county mapping from a CSV file
@st.cache_data
def load_region_county_dict():
    df = pd.read_csv(DICT_PATH)
    return dict(zip(df['Region'], df['County']))

region_county_dict = load_region_county_dict()

st.title("Land Loan Application")

# Function to calculate annuity payment
def calculate_annuity_payment(principal, annual_interest_rate, periods):
    monthly_interest_rate = annual_interest_rate / 12
    return principal * monthly_interest_rate / (1 - (1 + monthly_interest_rate)**-periods)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title("Land Purchase and Loan Application")

# User inputs
land_type = st.selectbox("Land Type", 
                         ["Forest land", "Farmland", "Residential", "Commercial", "Industrial"], 
                         key="land_type_select")
selected_region = st.selectbox("Region", 
                               list(region_county_dict.keys()), 
                               key="region_select")
selected_county = region_county_dict[selected_region]
plot_size = st.number_input("Plot Size (in hectars)", min_value=1.0, step=0.1)
loan_term = st.selectbox("Loan Term (Months)", 
                         list(range(6, 25)), 
                         key="loan_term_select")
payment_frequency = st.selectbox("Payment Frequency", 
                                 ["Monthly", "Quarterly", "Semi-Annual", "Annual"], 
                                 key="payment_frequency_select")
loan_amount = st.number_input("Loan Amount", min_value=1000, step=1000)

st.write(f"Selected County: {selected_county}")

# 'Get Quote' button action
if st.button("Calculate"):
    # Fetch avg_price_eur based on land_type, county, region
    avg_price_eur = df[(df['land_type'] == land_type) & (df['county'] == selected_county) & (df['region'] == selected_region)]['avg_price_eur'].mean()
    
    if not np.isnan(avg_price_eur):
        property_value = avg_price_eur * plot_size
        loan_to_value = 0.6  # 60%
        interest_rate = 0.08  # 8%
        max_loan_amount = property_value * loan_to_value

        # Prepare data for DataFrame
        data = {
            "Item": ["Land Value (Estimated) (EUR)", "Maximum Loan Amount (60% LTV, 8% annual) (EUR)", "Monthly Payment (EUR)"],
            "Value": [f"{property_value:.2f}", f"{max_loan_amount:.2f}", ""]
        }

        # Annuity payment calculation
        if payment_frequency == "Monthly":
            periods = loan_term
            monthly_payment = calculate_annuity_payment(max_loan_amount, interest_rate, periods)
            data["Value"][2] = f"{monthly_payment:.2f}"

        # Create and display DataFrame
        loan_info_df = pd.DataFrame(data)
               # Convert DataFrame to HTML and use markdown to display it with styling
        loan_info_html = loan_info_df.to_html(index=False, escape=False)
        loan_info_html = loan_info_html.replace('<th>', '<th style="font-weight: bold; background-color: #f0f0f0; text-align: left;">')
        st.markdown(loan_info_html, unsafe_allow_html=True)

    else:
        st.write("No data available for the selected combination.")

st.text("Copyright 2024 Landex.ai")
