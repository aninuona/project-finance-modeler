"""project finance model, no streamlit in here so it can be tested on its own

timeline: years 0..c-1 construction, c..c+o-1 operations (no extra year at the end)
"""
import numpy as np
import pandas as pd


# ---------- finance helpers (own versions so no numpy_financial dependency) ----------

def npv(rate, cfs):
    # year 0 not discounted, same convention as excel NPV on a range starting at t0
    cfs = np.asarray(cfs, dtype=float)
    t = np.arange(len(cfs))
    return float(np.sum(cfs / (1 + rate) ** t))


def irr(cfs):
    # solve sum cf_t * x^t = 0 for x = 1/(1+r), keep real roots w r > -1
    # returns nan if no sign change (irr undefined)
    cfs = np.asarray(cfs, dtype=float)
    if not (np.any(cfs > 0) and np.any(cfs < 0)):
        return np.nan
    roots = np.roots(cfs[::-1])
    real = roots[np.isclose(roots.imag, 0, atol=1e-10)].real
    real = real[real > 0]
    if real.size == 0:
        return np.nan
    rates = 1 / real - 1
    # if more than one irr exists pick the one closest to 10%, same idea as numpy_financial
    return float(rates[np.argmin(np.abs(rates - 0.10))])


def level_payment(rate, n, principal):
    # annuity payment, rate 0 -> straight split
    if principal <= 0:
        return 0.0
    if rate == 0:
        return principal / n
    return principal * rate / (1 - (1 + rate) ** -n)


# ---------- main model ----------

def run_model(capex, construction_years, ops_years, gearing, interest_rate,
              annual_revenue, revenue_growth, annual_opex, opex_growth,
              tax_rate, discount_rate,
              rev_mult=1.0, opex_mult=1.0, capex_overrun=0.0, build_table=True):
    c, o = int(construction_years), int(ops_years)
    n = c + o
    years = np.arange(n)
    is_ops = years >= c
    t_ops = np.clip(years - c, 0, None)  # 0 in first ops yr

    # capex spread evenly over construction, overrun scales the whole budget
    capex_total = capex * (1 + capex_overrun)
    capex_s = np.where(is_ops, 0.0, capex_total / c)

    # funding, debt + equity pro rata w capex
    debt_draw = capex_s * gearing
    equity_in = capex_s * (1 - gearing)

    interest = np.zeros(n)
    principal = np.zeros(n)
    idc = np.zeros(n)  # interest during construction, capitalized into the loan
    closing = np.zeros(n)

    bal = 0.0
    for y in range(c):
        idc[y] = bal * interest_rate
        bal += idc[y] + debt_draw[y]
        closing[y] = bal
    debt_at_cod = bal

    # level debt service over ops yrs (mortgage style)
    pmt = level_payment(interest_rate, o, debt_at_cod)
    for y in range(c, n):
        interest[y] = bal * interest_rate
        principal[y] = pmt - interest[y]
        bal -= principal[y]
        closing[y] = bal
    debt_service = interest + principal

    # operations
    revenue = np.where(is_ops, annual_revenue * rev_mult * (1 + revenue_growth) ** t_ops, 0.0)
    opex = np.where(is_ops, annual_opex * opex_mult * (1 + opex_growth) ** t_ops, 0.0)
    ebitda = revenue - opex

    # straight line depreciation of capex over ops life
    # simplification: no loss carryforward, losses just pay 0 tax
    dep = np.where(is_ops, capex_total / o, 0.0)
    tax_unlevered = np.maximum(ebitda - dep, 0) * tax_rate
    tax_levered = np.maximum(ebitda - dep - interest, 0) * tax_rate

    # cfads = cash available for debt service, after tax
    cfads = ebitda - tax_levered
    has_ds = debt_service > 1e-9
    dscr = np.full(n, np.nan)
    dscr[has_ds] = cfads[has_ds] / debt_service[has_ds]

    # project (unlevered) vs equity cash flows, kept separate on purpose
    project_fcf = ebitda - tax_unlevered - capex_s
    equity_fcf = cfads - debt_service - equity_in

    ops_dscr = dscr[is_ops & has_ds]
    out = {
        "project_irr": irr(project_fcf),
        "equity_irr": irr(equity_fcf),
        "project_npv": npv(discount_rate, project_fcf),
        "min_dscr": float(np.min(ops_dscr)) if ops_dscr.size else np.nan,
        "avg_dscr": float(np.mean(ops_dscr)) if ops_dscr.size else np.nan,
        "debt_at_cod": debt_at_cod,
        "idc_total": float(idc.sum()),
        "ending_debt": float(closing[-1]),
        "capex_total": capex_total,
        "debt_drawn": float(debt_draw.sum()),
        "equity_in": float(equity_in.sum()),
    }

    if build_table:
        out["table"] = pd.DataFrame({
            "Phase": np.where(is_ops, "Ops", "Build"),
            "Capex": capex_s, "Debt draw": debt_draw, "Equity in": equity_in,
            "Revenue": revenue, "Opex": opex, "EBITDA": ebitda,
            "Depreciation": dep, "Tax (levered)": tax_levered, "CFADS": cfads,
            "Interest": interest, "IDC": idc, "Principal": principal,
            "Debt service": debt_service, "Debt balance": closing, "DSCR": dscr,
            "Project FCF": project_fcf, "Equity FCF": equity_fcf,
        }, index=pd.Index(years, name="Year"))
    return out


# ---------- monte carlo ----------

def run_monte_carlo(base, n_sims=1000, rev_sd=0.10, opex_sd=0.10,
                    overrun_max=0.20, rate_sd=0.01, seed=42):
    """re-runs the full model per draw, so every output reacts to the shocked inputs

    shocks (all persistent for the whole project life):
      revenue level   ~ normal(1, rev_sd)
      opex level      ~ normal(1, opex_sd)
      capex overrun   ~ triangular(0, 0, overrun_max), most draws small, some big
      interest rate   ~ base + normal(0, rate_sd)
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(int(n_sims)):
        kw = dict(base)
        rev_mult = max(rng.normal(1, rev_sd), 0.05)
        opex_mult = max(rng.normal(1, opex_sd), 0.05)
        overrun = rng.triangular(0, 0, overrun_max) if overrun_max > 0 else 0.0
        kw["interest_rate"] = max(base["interest_rate"] + rng.normal(0, rate_sd), 0.0)
        r = run_model(**kw, rev_mult=rev_mult, opex_mult=opex_mult,
                      capex_overrun=overrun, build_table=False)
        rows.append({
            "rev_mult": rev_mult, "opex_mult": opex_mult, "capex_overrun": overrun,
            "interest_rate": kw["interest_rate"],
            "project_irr": r["project_irr"], "equity_irr": r["equity_irr"],
            "project_npv": r["project_npv"], "min_dscr": r["min_dscr"],
        })
    return pd.DataFrame(rows)


# ---------- self checks shown in the app ----------

def model_checks(base, result):
    checks = []
    checks.append(("Debt + equity fund 100% of capex",
                   np.isclose(result["debt_drawn"] + result["equity_in"], result["capex_total"])))
    checks.append(("Loan fully repaid by final year",
                   abs(result["ending_debt"]) < 1e-6))
    checks.append(("Debt at start of ops = draws + capitalized interest",
                   np.isclose(result["debt_at_cod"], result["debt_drawn"] + result["idc_total"])))
    # at 0% debt, equity and project cash flows are the same thing
    no_debt = run_model(**{**base, "gearing": 0.0}, build_table=False)
    same = (np.isnan(no_debt["equity_irr"]) and np.isnan(no_debt["project_irr"])) or \
        np.isclose(no_debt["equity_irr"], no_debt["project_irr"])
    checks.append(("With 0% debt, equity IRR = project IRR", bool(same)))
    return checks
