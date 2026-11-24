import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import run_model, run_monte_carlo, model_checks


def pct(x):
    return "n/a" if x is None or np.isnan(x) else f"{x * 100:.1f}%"


def mult(x):
    return "n/a" if x is None or np.isnan(x) else f"{x:.2f}x"


CSS = """
<style>
[data-testid="stSidebarUserContent"] { padding-top: 1rem !important; }
.stMetric { background-color: #1e2130; padding: 10px; border-radius: 8px; }
</style>
"""


def main():
    st.set_page_config(page_title="Project Finance Modeler", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    # ---------- inputs ----------
    with st.sidebar:
        st.header("Inputs")
        with st.expander("Project and timing", expanded=True):
            capex = st.number_input("Total capex ($M)", value=150.0, min_value=1.0)
            construction_years = st.slider("Construction (years)", 1, 5, 3)
            ops_years = st.slider("Operating life (years)", 5, 30, 25)
        with st.expander("Financing", expanded=True):
            gearing = st.slider("Debt share of capex (%)", 0, 90, 75) / 100
            interest_rate = st.number_input("Interest rate (%)", value=5.5, min_value=0.0) / 100
        with st.expander("Operations and tax", expanded=True):
            annual_revenue = st.number_input("Year 1 revenue ($M)", value=22.0, min_value=0.0)
            revenue_growth = st.slider("Revenue growth (%/yr)", 0.0, 5.0, 2.0) / 100
            annual_opex = st.number_input("Year 1 opex ($M)", value=7.0, min_value=0.0)
            opex_growth = st.slider("Opex growth (%/yr)", 0.0, 5.0, 2.0) / 100
            tax_rate = st.slider("Tax rate (%)", 0, 40, 21) / 100
            discount_rate = st.number_input("WACC for project NPV (%)", value=8.0) / 100
        with st.expander("Hurdles"):
            dscr_hurdle = st.number_input("Lender minimum DSCR (x)", value=1.20, step=0.05)
            equity_hurdle = st.number_input("Equity target IRR (%)", value=12.0) / 100
        with st.expander("Risk simulation"):
            n_sims = st.slider("Simulations", 200, 5000, 1000, step=100)
            rev_sd = st.slider("Revenue level, 1 std dev (%)", 0, 30, 10) / 100
            opex_sd = st.slider("Opex level, 1 std dev (%)", 0, 30, 10) / 100
            overrun_max = st.slider("Worst case capex overrun (%)", 0, 50, 20) / 100
            rate_sd = st.slider("Interest rate, 1 std dev (pp)", 0.0, 3.0, 1.0) / 100
            seed = int(st.number_input("Random seed", value=42, step=1))

    base = dict(capex=capex, construction_years=construction_years, ops_years=ops_years,
                gearing=gearing, interest_rate=interest_rate,
                annual_revenue=annual_revenue, revenue_growth=revenue_growth,
                annual_opex=annual_opex, opex_growth=opex_growth,
                tax_rate=tax_rate, discount_rate=discount_rate)
    res = run_model(**base)
    tbl = res["table"]
    ops = tbl[tbl["Phase"] == "Ops"]

    # ---------- header + headline numbers ----------
    st.title("Project Finance Modeler")
    st.caption("Annual model | Construction funded with D & E in same proportion | "
               "interest during construction is added to the loan, loan is repaid in level "
               "payments over operating life")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Project IRR (unlevered, after tax)", pct(res["project_irr"]))
    m2.metric("Equity IRR", pct(res["equity_irr"]),
              delta=None if np.isnan(res["equity_irr"]) else f"{(res['equity_irr'] - equity_hurdle) * 100:+.1f} pp vs target")
    m3.metric("Project NPV at WACC", f"${res['project_npv']:.1f}M")
    m4.metric("Min DSCR", mult(res["min_dscr"]),
              delta=None if np.isnan(res["min_dscr"]) else f"{res['min_dscr'] - dscr_hurdle:+.2f}x vs lender min")
    m5.metric("Avg DSCR", mult(res["avg_dscr"]))

    tab1, tab2, tab3, tab4 = st.tabs(["Cash flows", "Debt", "Risk simulation", "Checks and full table"])

    # ---------- cash flows ----------
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=tbl.index, y=tbl["Project FCF"], name="Project FCF", marker_color="#5c7cfa"))
        fig.add_trace(go.Bar(x=tbl.index, y=tbl["Equity FCF"], name="Equity FCF", marker_color="#94d82d"))
        fig.add_trace(go.Scatter(x=tbl.index, y=tbl["EBITDA"], name="EBITDA", line_color="#ffa94d"))
        fig.update_layout(template="plotly_dark", barmode="group", height=450,
                          xaxis_title="Year", yaxis_title="$M", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Project FCF ignores financing (EBITDA - unlevered tax - capex). "
                   "Equity FCF = what the sponsor puts in and gets back after debt service and tax.")

    # ---------- debt ----------
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("DSCR by operating year")
            fig2 = go.Figure(go.Bar(x=ops.index, y=ops["DSCR"], marker_color="#607d8b"))
            fig2.add_hline(y=dscr_hurdle, line_dash="dash", line_color="#ff8787",
                           annotation_text=f"Lender min {dscr_hurdle:.2f}x")
            fig2.update_layout(template="plotly_dark", height=400, xaxis_title="Year", yaxis_title="DSCR (x)")
            st.plotly_chart(fig2, use_container_width=True)
        with col_b:
            st.subheader("Loan balance")
            fig3 = go.Figure(go.Scatter(x=tbl.index, y=tbl["Debt balance"], fill="tozeroy", line_color="#457b9d"))
            fig3.update_layout(template="plotly_dark", height=400, xaxis_title="Year", yaxis_title="$M")
            st.plotly_chart(fig3, use_container_width=True)
        st.write(f"Debt at start of operations: **${res['debt_at_cod']:.1f}M**, "
                 f"including ${res['idc_total']:.1f}M of interest during construction.")

    # ---------- risk simulation ----------
    with tab3:
        st.write("Each draw shocks revenue, opex, capex and the interest rate, then reruns the whole model.")
        sim_key = (tuple(base.items()), n_sims, rev_sd, opex_sd, overrun_max, rate_sd, seed)
        if st.button(f"Run {n_sims:,} simulations"):
            mc = run_monte_carlo(base, n_sims=n_sims, rev_sd=rev_sd, opex_sd=opex_sd,
                                 overrun_max=overrun_max, rate_sd=rate_sd, seed=seed)
            st.session_state["mc"] = (sim_key, mc)

        saved = st.session_state.get("mc")
        if saved and saved[0] != sim_key:
            st.info("Inputs changed since the last run. Run the simulation again to update.")
        elif saved:
            mc = saved[1]
            eq = mc["equity_irr"].dropna()
            n_undefined = len(mc) - len(eq)

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Equity IRR, P10", pct(eq.quantile(0.10)) if len(eq) else "n/a")
            s2.metric("Equity IRR, P50", pct(eq.quantile(0.50)) if len(eq) else "n/a")
            s3.metric("Equity IRR, P90", pct(eq.quantile(0.90)) if len(eq) else "n/a")
            # undefined irr = equity never paid back, count it as a miss
            miss = ((mc["equity_irr"] < equity_hurdle) | mc["equity_irr"].isna()).mean()
            s4.metric("Chance equity misses target", f"{miss * 100:.0f}%")
            s5.metric("Chance min DSCR below lender min", f"{(mc['min_dscr'] < dscr_hurdle).mean() * 100:.0f}%")

            fig4 = go.Figure(go.Histogram(x=eq * 100, nbinsx=40, marker_color="#00d2ff"))
            fig4.add_vline(x=equity_hurdle * 100, line_dash="dash", line_color="#ff8787",
                           annotation_text="Target")
            fig4.update_layout(template="plotly_dark", height=400,
                               xaxis_title="Equity IRR (%)", yaxis_title="Draws")
            st.plotly_chart(fig4, use_container_width=True)
            if n_undefined:
                st.caption(f"{n_undefined} draws had no defined equity IRR (equity never paid back) "
                           "and are left out of the chart but counted as misses.")

            # which shock moves equity irr most (rank correlation)
            drivers = mc[["rev_mult", "opex_mult", "capex_overrun", "interest_rate", "equity_irr"]] \
                .corr(method="spearman")["equity_irr"].drop("equity_irr")
            drivers.index = ["Revenue level", "Opex level", "Capex overrun", "Interest rate"]
            st.subheader("What drives equity IRR")
            st.dataframe(drivers.rename("Rank correlation with equity IRR")
                         .to_frame().sort_values("Rank correlation with equity IRR", key=abs, ascending=False)
                         .style.format("{:+.2f}"))

    # ---------- checks + table ----------
    with tab4:
        st.subheader("Model checks")
        for name, ok in model_checks(base, res):
            (st.success if ok else st.error)(f"{'Pass' if ok else 'Fail'}: {name}")
        st.subheader("Annual cash flow table ($M)")
        num_cols = [c for c in tbl.columns if c != "Phase"]
        st.dataframe(tbl.style.format("{:.2f}", subset=num_cols, na_rep=""), height=450)
        st.download_button("Download table as CSV", tbl.to_csv().encode(), "project_cash_flows.csv", "text/csv")

    st.caption("Limits: annual periods, no tax loss carryforward, no reserve accounts, "
               "and debt share is an input (not sized to a target DSCR).")


if __name__ == "__main__":
    main()
