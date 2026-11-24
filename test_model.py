"""run w: python test_model.py  (or pytest)"""
import numpy as np
from model import npv, irr, level_payment, run_model, run_monte_carlo, model_checks

BASE = dict(capex=150.0, construction_years=3, ops_years=25, gearing=0.70,
            interest_rate=0.055, annual_revenue=65.0, revenue_growth=0.02,
            annual_opex=15.0, opex_growth=0.02, tax_rate=0.21, discount_rate=0.08)


def bisect_irr(cfs, lo=-0.99, hi=1.0):
    # slow independent irr to check the fast one
    f = lambda r: npv(r, cfs)
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def test_helpers():
    assert np.isclose(irr([-100, 110]), 0.10)
    assert np.isclose(irr([-100, 60, 60]), bisect_irr([-100, 60, 60]), atol=1e-8)
    assert np.isnan(irr([100, 50]))  # no sign change
    assert np.isclose(npv(0.10, [-100, 110]), 0.0)
    assert np.isclose(level_payment(0.05, 10, 100), 12.950457496545, atol=1e-9)
    assert np.isclose(level_payment(0.0, 10, 100), 10.0)


def test_timeline_and_dscr():
    r = run_model(**BASE)
    tbl = r["table"]
    assert len(tbl) == 28  # 3 build + 25 ops, no extra year
    ops = tbl[tbl["Phase"] == "Ops"]
    assert (ops["Debt service"] > 0).all()  # every ops yr has debt service
    assert ops["DSCR"].notna().all() and (ops["DSCR"] > 0).all()  # no fake 0 dscr
    assert tbl.loc[tbl["Phase"] == "Build", "DSCR"].isna().all()


def test_irr_matches_independent_solver():
    r = run_model(**BASE)
    tbl = r["table"]
    assert np.isclose(r["project_irr"], bisect_irr(tbl["Project FCF"].values), atol=1e-7)
    assert np.isclose(r["equity_irr"], bisect_irr(tbl["Equity FCF"].values), atol=1e-7)


def test_checks_pass():
    r = run_model(**BASE)
    for name, ok in model_checks(BASE, r):
        assert ok, name


def test_leverage_direction():
    # positive leverage at these inputs: equity irr should beat project irr
    r = run_model(**BASE)
    assert r["equity_irr"] > r["project_irr"]


def test_monte_carlo_zero_shocks_equals_base():
    r = run_model(**BASE)
    mc = run_monte_carlo(BASE, n_sims=20, rev_sd=0, opex_sd=0, overrun_max=0, rate_sd=0)
    assert np.allclose(mc["equity_irr"], r["equity_irr"])
    assert np.allclose(mc["min_dscr"], r["min_dscr"])


def test_monte_carlo_reacts_to_inputs():
    mc = run_monte_carlo(BASE, n_sims=300, seed=1)
    # worse revenue draws should mean lower equity irr on average
    lo = mc[mc["rev_mult"] < 0.95]["equity_irr"].mean()
    hi = mc[mc["rev_mult"] > 1.05]["equity_irr"].mean()
    assert lo < hi


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("pass", name)
    r = run_model(**BASE)
    print({k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
