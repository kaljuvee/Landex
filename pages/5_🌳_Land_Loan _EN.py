import streamlit as st
import yaml

DATA_PATH = 'data/maaamet_farm_forest_2022.csv'

st.title('Land Value and Loan Calculator')

def get_land_data():
    df = pd.read_csv(DATA_PATH)
    return df
    
def get_user_input(rev_multipliers):
    # Options for land type
    land_options = ["Forest Land", "Farm Land"]
    df = get_land_data()
    county_values = df['county'].unique()
    county_list = county_values.tolist()

    reguion_values = df['region'].unique()
    region_list = reguion_values.tolist()
    
    # Create a select box for land type
    land_type = st.selectbox('Select land type:', options)
    county = st.selectbox('Select county:', county_list)
    region = st.selectbox('Select region:', region_list)
    land_size = st.text_input('Enter land size (in hectars):', '0')
    
    try:
        land_size = float(land_size)
    except ValueError:
        st.error('Please enter a valid number for land size)
        land_size = None
    sector = st.selectbox('Select Industry Sector:', options=list(rev_multipliers.keys()))
    return land_type, county, region, land_size

def calculate_valuation(ebitda, revenue, sector, rev_multipliers, ebidta_multipliers):
    if ebitda is not None and sector in rev_multipliers:
        rev_multiplier = rev_multipliers[sector]
        ebidta_multiplier = ebidta_multipliers[sector]
        valuation = (ebitda * ebidta_multiplier + revenue * rev_multiplier) / 2
        return valuation
    return None

def display_result(valuation, sector, ebitda, revenue):
    if valuation is not None:
        st.write(f'The estimated **company valuation** is: **${valuation:,.2f}**')
        st.write(f'Based on the **revenue** of: ${revenue:,.2f} and **EBITDA** of: ${ebitda:,.2f}')
        st.write(f'The company **sector** is: {sector}')
    else:
        st.write('Please enter valid inputs to calculate the valuation.')

def main():
    rev_multipliers = load_rev_multipliers()
    ebidta_multipliers = load_ebidta_multipliers()
    ebitda, revenue, sector = get_user_input(rev_multipliers)

    # Add a button to trigger the calculation
    if st.button('Calculate Valuation'):
        valuation = calculate_valuation(ebitda, revenue, sector, rev_multipliers, ebidta_multipliers)
        display_result(valuation, sector, ebitda, revenue)
    else:
        st.write('Enter the details and press "Calculate Valuation" to see the results.')

if __name__ == "__main__":
    main()

st.text("Copyright Landex.ai")
