import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go

st.set_page_config(page_title="Project Finance Pro", layout="wide", initial_sidebar_state="expanded")

# --- CSS FOR DARK MODE ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

st.title("Project Finance & Risk Simulator")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Project Assumptions")
    capex = st.number_input("Total Capex ($M)", value=100.0)
    construction_years = st.slider("Construction Period (Years)", 1, 5, 2)
    ops_years = st.slider("Operating Period (Years)", 5, 30, 20)
    
    st.header("Financing")
    gearing = st.slider("Debt Ratio (%)", 0, 90, 70) / 100
    interest_rate = st.number_input("Interest Rate (%)", value=5.0) / 100
    
    st.header("Operations")
    annual_revenue = st.number_input("Annual Revenue ($M)", value=25.0)
    annual_opex = st.number_input("Annual OpEx ($M)", value=8.0)
    discount_rate = st.number_input("WACC / Discount Rate (%)", value=8.0) / 100

# --- CALCULATIONS ENGINE ---
total_years = construction_years + ops_years
years = np.arange(0, total_years + 1)

# Initialize Cash Flows
cash_flows = pd.DataFrame(index=years)
cash_flows['Phase'] = ['Construction']*construction_years + ['Operating']*(ops_years + 1)

# Capex Schedule (Straight line for simplicity)
cash_flows['Capex'] = 0.0
cash_flows.iloc[0:construction_years, cash_flows.columns.get_loc('Capex')] = -capex / construction_years

# Debt Modeling
debt_amount = capex * gearing
equity_amount = capex * (1 - gearing)
annual_principal = debt_amount / ops_years

cash_flows['Revenue'] = 0.0
cash_flows['OpEx'] = 0.0
cash_flows.iloc[construction_years+1:, cash_flows.columns.get_loc('Revenue')] = annual_revenue
cash_flows.iloc[construction_years+1:, cash_flows.columns.get_loc('OpEx')] = -annual_opex

cash_flows['EBITDA'] = cash_flows['Revenue'] + cash_flows['OpEx']

# Debt Service
cash_flows['Principal'] = 0.0
cash_flows['Interest'] = 0.0
remaining_debt = debt_amount

for i in range(construction_years + 1, total_years + 1):
    interest = remaining_debt * interest_rate
    cash_flows.loc[i, 'Interest'] = -interest
    cash_flows.loc[i, 'Principal'] = -annual_principal
    remaining_debt -= annual_principal

cash_flows['Debt_Service'] = cash_flows['Principal'] + cash_flows['Interest']
cash_flows['CFADS'] = cash_flows['EBITDA'] # Cash Flow Available for Debt Service
cash_flows['Free_Cash_Flow'] = cash_flows['Capex'] + cash_flows['EBITDA'] + cash_flows['Debt_Service']

# --- METRICS ---
npv = npf.npv(discount_rate, cash_flows['Free_Cash_Flow'])
irr = npf.irr(cash_flows['Free_Cash_Flow'])
dscr = cash_flows.loc[construction_years+1:, 'CFADS'] / abs(cash_flows.loc[construction_years+1:, 'Debt_Service'])

# --- UI LAYOUT ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Project NPV", f"${npv:.2f}M")
col2.metric("Project IRR", f"{irr*100:.2f}%")
col3.metric("Min DSCR", f"{dscr.min():.2f}x")
col4.metric("Avg DSCR", f"{dscr.mean():.2f}x")

# --- VISUALIZATION ---
st.subheader("Cash Flow Waterfall")
fig = go.Figure()
fig.add_trace(go.Bar(x=years, y=cash_flows['Revenue'], name='Revenue', marker_color='#2ecc71'))
fig.add_trace(go.Bar(x=years, y=cash_flows['OpEx'], name='OpEx', marker_color='#e74c3c'))
fig.add_trace(go.Bar(x=years, y=cash_flows['Debt_Service'], name='Debt Service', marker_color='#f1c40f'))
fig.update_layout(barmode='relative', template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# --- MONTE CARLO SIMULATION ---
st.divider()
st.subheader("🎲 Monte Carlo Risk Simulation")
n_sims = st.select_slider("Number of Iterations", options=[100, 500, 1000], value=500)

if st.button("Run Simulation"):
    sim_irrs = []
    for _ in range(n_sims):
        # Randomize Revenue and OpEx by +/- 15%
        rand_rev = annual_revenue * np.random.uniform(0.85, 1.15)
        rand_opex = annual_opex * np.random.uniform(0.85, 1.15)
        
        sim_cf = cash_flows['Free_Cash_Flow'].copy()
        sim_cf.iloc[construction_years+1:] = (rand_rev - rand_opex) + cash_flows.loc[construction_years+1:, 'Debt_Service']
        sim_irrs.append(npf.irr(sim_cf))
    
    fig_sim = go.Figure(data=[go.Histogram(x=sim_irrs, nbinsx=30, marker_color='#3498db')])
    fig_sim.update_layout(title="Probability Distribution of IRR", template="plotly_dark", xaxis_title="IRR")
    st.plotly_chart(fig_sim, use_container_width=True)

# --- EXPORT ---
st.download_button(
    label="📥 Export Model to CSV",
    data=cash_flows.to_csv().encode('utf-8'),
    file_name='financial_model.csv',
    mime='text/csv',
)
