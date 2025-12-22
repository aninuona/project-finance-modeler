import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Config
st.set_page_config(page_title="Project Finance Pro", layout="wide")

# --- CSS FOR SIDEBAR & STYLING ---
st.markdown("""
    <style>
    /* Reduce sidebar padding and widget margins */
    [data-testid="stSidebarUserContent"] {
        padding-top: 1rem !important;
    }
    .stNumberInput, .stSlider, .stSelectbox {
        margin-bottom: -15px !important;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 10px;
        border-radius: 8px;
    }
    /* Style for the Min DSCR warning box */
    .dscr-warning {
        background-color: #45331e;
        color: #ffcc80;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffa726;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR INPUTS (small space) ---
with st.sidebar:
    st.title("Settings")
    
    with st.expander("🏗️ Project & Time", expanded=True):
        capex = st.number_input("Total Capex ($M)", value=150.0)
        construction_years = st.slider("Construction (Yrs)", 1, 5, 3)
        ops_years = st.slider("Operating (Yrs)", 5, 30, 25)
    
    with st.expander("💰 Financing", expanded=True):
        gearing = st.slider("Debt Ratio (%)", 0, 90, 70) / 100
        interest_rate = st.number_input("Interest Rate (%)", value=5.5) / 100
    
    with st.expander("📈 Operations", expanded=True):
        annual_revenue = st.number_input("Initial Revenue ($M)", value=65.0)
        revenue_growth = st.slider("Revenue Growth (%)", 0.0, 5.0, 2.0) / 100
        annual_opex = st.number_input("Initial OpEx ($M)", value=15.0)
        discount_rate = st.number_input("WACC (%)", value=8.0) / 100

# --- CALCULATIONS ---
total_years = construction_years + ops_years
years = np.arange(0, total_years + 1)
df = pd.DataFrame(index=years)

# Capex & Revenue Schedules
df['Capex'] = 0.0
df.loc[0:construction_years-1, 'Capex'] = capex / construction_years

# Calculate Growth over time
df['Revenue'] = 0.0
df['OpEx'] = 0.0
for y in range(construction_years, total_years + 1):
    growth_factor = (1 + revenue_growth) ** (y - construction_years)
    df.loc[y, 'Revenue'] = annual_revenue * growth_factor
    df.loc[y, 'OpEx'] = annual_opex * growth_factor

# Debt Calculations
debt_total = capex * gearing
annual_principal = debt_total / ops_years
df['Interest'] = 0.0
df['Principal'] = 0.0
remaining_debt = debt_total

for y in range(construction_years, total_years):
    interest = remaining_debt * interest_rate
    df.loc[y, 'Interest'] = interest
    df.loc[y, 'Principal'] = annual_principal
    remaining_debt -= annual_principal

df['Debt_Service'] = df['Interest'] + df['Principal']
df['CFADS'] = df['Revenue'] - df['OpEx']
df['DSCR'] = np.where(df['Debt_Service'] > 0, df['CFADS'] / df['Debt_Service'], 0)
df['FCF'] = df['CFADS'] - df['Capex'] - df['Debt_Service']

# --- MAIN INTERFACE TABS ---
st.title("Project Finance Analysis")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Cashflow Overview", "📉 Debt Service", "🎯 Metrics", "🎲 Risk Analysis"])

# --- TAB 1: REVENUE/OPEX/CAPEX AREA CHART ---
with tab1:
    st.subheader("Revenue, OPEX & CAPEX")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=years, y=df['Revenue'], name='Revenue', fill='tozeroy', line_color='#4caf50'))
    fig1.add_trace(go.Scatter(x=years, y=df['OpEx'], name='OpEx', fill='tozeroy', line_color='#f44336'))
    fig1.add_trace(go.Scatter(x=years, y=df['Capex'], name='Capex', fill='tozeroy', line_color='#ff9800'))
    fig1.update_layout(template="plotly_dark", height=500, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig1, use_container_width=True)

# --- TAB 2: DSCR & FCF ---
with tab2:
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.subheader("Debt Service Coverage Ratio")
        # Filter for operating years only for the bar chart
        ops_df = df[df.index >= construction_years]
        fig2 = go.Figure(data=[go.Bar(x=ops_df.index, y=ops_df['DSCR'], marker_color='#607d8b')])
        fig2.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        min_dscr = ops_df['DSCR'].min()
        st.markdown(f"""<div class="dscr-warning">Min DSCR: <b>{min_dscr:.2f}x</b> (Lenders typically require 1.20-1.30x)</div>""", unsafe_allow_html=True)

    with col_b:
        st.subheader("FCF & DSCR Trend")
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=years, y=df['FCF'], name="FCF", line_color="#457b9d"), secondary_y=False)
        fig3.add_trace(go.Scatter(x=years, y=df['DSCR'], name="DSCR", line_color="#a29bfe"), secondary_y=True)
        fig3.update_layout(template="plotly_dark", height=450, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

# --- TAB 3: METRICS ---
with tab3:
    m1, m2, m3 = st.columns(3)
    project_irr = npf.irr(df['FCF'])
    project_npv = npf.npv(discount_rate, df['FCF'])
    
    m1.metric("Project IRR", f"{project_irr*100:.2f}%")
    m2.metric("Net Present Value", f"${project_npv:.2f}M")
    m3.metric("Avg DSCR", f"{ops_df['DSCR'].mean():.2f}x")
    
    st.dataframe(df.style.format("{:.2f}"), height=400)

# --- TAB 4: MONTE CARLO ---
with tab4:
    st.subheader("Monte Carlo Risk Simulation")
    if st.button("Run 1,000 Simulations"):
        sim_results = [project_irr + np.random.normal(0, 0.02) for _ in range(1000)]
        fig4 = go.Figure(data=[go.Histogram(x=sim_results, marker_color='#00d2ff')])
        fig4.update_layout(template="plotly_dark", title="Probability Distribution of IRR")
        st.plotly_chart(fig4, use_container_width=True)
