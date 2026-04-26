import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import sys
import os

# Page config
st.set_page_config(
    page_title="Reinsurance Portfolio Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Core functions (inline, no src/ dependency)
# ─────────────────────────────────────────────

@st.cache_data
def load_portfolio(filepath="../data/processed/claims_cleaned.csv"):
    df = pd.read_csv(filepath)
    return df["loss"].values

@st.cache_data
def simulate_annual_losses(losses, n_years=10000, avg_claims_per_year=5000, seed=42):
    np.random.seed(seed)
    totals = np.zeros(n_years)
    for i in range(n_years):
        n = np.random.poisson(avg_claims_per_year)
        if n > 0:
            totals[i] = np.random.choice(losses, size=n, replace=True).sum()
    return totals

def fit_gpd(losses, threshold_pct=95):
    threshold   = np.percentile(losses, threshold_pct)
    exceedances = losses[losses > threshold] - threshold
    params      = stats.genpareto.fit(exceedances)
    return threshold, exceedances, params

def apply_xl(losses, retention, limit):
    ceded    = np.clip(losses - retention, 0, limit)
    retained = losses - ceded
    return retained, ceded

def apply_stop_loss(annual, attachment, limit):
    ceded    = np.clip(annual - attachment, 0, limit)
    retained = annual - ceded
    return retained, ceded

def burning_cost_premium(losses, retention, limit, loading=0.15):
    _, ceded = apply_xl(losses, retention, limit)
    return ceded.sum() * (1 + loading)

def stop_loss_premium(annual, attachment, limit, loading=0.25):
    _, ceded = apply_stop_loss(annual, attachment, limit)
    return ceded.mean() * (1 + loading)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

st.sidebar.title("Reinsurance Portfolio Optimizer")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Portfolio Overview", "Extreme Value Theory",
     "Treaty Structuring", "Program Optimization", "Stress Testing"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Settings**")
n_years    = st.sidebar.slider("Simulated years (Monte Carlo)", 5000, 20000, 10000, 1000)
avg_claims = st.sidebar.slider("Avg claims per year", 1000, 10000, 5000, 500)

st.sidebar.markdown("---")
st.sidebar.markdown("**Cost of Capital**")
coc = st.sidebar.slider("Cost of Capital (%)", 4, 15, 8) / 100

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

data_path = os.path.join(os.path.dirname(__file__), "../data/processed/claims_cleaned.csv")

try:
    individual_losses = load_portfolio(data_path)
    data_loaded = True
except FileNotFoundError:
    st.error("Portfolio data not found. Run notebook 01 first to generate claims_cleaned.csv")
    data_loaded = False
    st.stop()

annual_losses = simulate_annual_losses(individual_losses, n_years, avg_claims)

# ─────────────────────────────────────────────
# PAGE 1: Portfolio Overview
# ─────────────────────────────────────────────

if page == "Portfolio Overview":
    st.title("Portfolio Overview")
    st.markdown("French Motor TPL portfolio — freMTPL2 dataset (678,013 policies)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Individual Claims",   f"{len(individual_losses):,}")
    col2.metric("Mean Severity",       f"{individual_losses.mean():,.0f}")
    col3.metric("VaR 99.5% (indiv.)",  f"{np.percentile(individual_losses, 99.5):,.0f}")
    col4.metric("Max Claim",           f"{individual_losses.max():,.0f}")

    col1b, col2b, col3b, col4b = st.columns(4)
    col1b.metric("Simulated Years",       f"{n_years:,}")
    col2b.metric("Mean Annual Loss",      f"{annual_losses.mean()/1e6:.2f}M")
    col3b.metric("VaR 99.5% (annual)",    f"{np.percentile(annual_losses, 99.5)/1e6:.2f}M")
    col4b.metric("VaR 99.9% (annual)",    f"{np.percentile(annual_losses, 99.9)/1e6:.2f}M")

    st.markdown("---")
    st.subheader("Loss Distributions")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(individual_losses, bins=80, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Claim Amount")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Individual Claim Severity")
    axes[0].set_xlim(0, np.percentile(individual_losses, 99))

    axes[1].hist(annual_losses / 1e6, bins=60, edgecolor="black", alpha=0.7, color="coral")
    axes[1].axvline(np.percentile(annual_losses, 99.5) / 1e6, color="red",
                    linestyle="--", linewidth=2, label="VaR 99.5%")
    axes[1].set_xlabel("Annual Aggregate Loss (M)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Annual Aggregate Loss ({n_years:,} simulated years)")
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Loss Exceedance Curve")
    thresholds = np.linspace(0, np.percentile(individual_losses, 99.9), 500)
    exc_prob   = np.array([np.mean(individual_losses > t) for t in thresholds])

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(thresholds, exc_prob, linewidth=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Loss Amount")
    ax2.set_ylabel("Exceedance Probability")
    ax2.set_title("Loss Exceedance Curve (log scale)")
    ax2.axhline(0.01,  color="red",    linestyle="--", label="1%")
    ax2.axhline(0.001, color="orange", linestyle="--", label="0.1%")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()

# ─────────────────────────────────────────────
# PAGE 2: EVT
# ─────────────────────────────────────────────

elif page == "Extreme Value Theory":
    st.title("Extreme Value Theory")
    st.markdown("Modeling tail losses with Generalized Pareto Distribution (GPD) — Peaks Over Threshold method.")

    threshold_pct = st.slider("Threshold percentile", 90, 99, 95)
    threshold, exceedances, gpd_params = fit_gpd(individual_losses, threshold_pct)
    shape, loc, scale = gpd_params

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Threshold",     f"{threshold:,.0f}")
    col2.metric("Exceedances",   f"{len(exceedances):,}")
    col3.metric("GPD Shape (ξ)", f"{shape:.4f}")
    col4.metric("GPD Scale (σ)", f"{scale:,.0f}")

    if shape > 0:
        st.info(f"Shape ξ = {shape:.4f} > 0: **Heavy tail (Pareto-type)**. Variance {'infinite' if shape >= 0.5 else 'finite'}.")
    else:
        st.info(f"Shape ξ = {shape:.4f} ≤ 0: Light tail.")

    st.markdown("---")
    st.subheader("Risk Metrics: GPD vs Empirical")

    n, n_ex = len(individual_losses), len(exceedances)
    rows = []
    for alpha in [0.90, 0.95, 0.99, 0.995, 0.999]:
        if abs(shape) > 1e-6:
            var_gpd = threshold + (scale / shape) * (((1 - alpha) / (n_ex / n)) ** (-shape) - 1)
        else:
            var_gpd = threshold + scale * np.log((1 - alpha) / (n_ex / n))
        var_emp  = np.percentile(individual_losses, alpha * 100)
        tvar_emp = individual_losses[individual_losses > var_emp].mean()
        rows.append({
            "Confidence": f"{alpha:.1%}",
            "VaR GPD":    f"{var_gpd:,.0f}",
            "VaR Empirical": f"{var_emp:,.0f}",
            "TVaR Empirical": f"{tvar_emp:,.0f}"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.subheader("Return Period Analysis")

    return_periods_list = [10, 20, 50, 100, 200, 500]
    rp_rows = []
    for T in return_periods_list:
        p = 1 / T
        if abs(shape) > 1e-6:
            loss = threshold + (scale / shape) * ((p * n / n_ex) ** (-shape) - 1)
        else:
            loss = threshold + scale * np.log(p * n / n_ex)
        rp_rows.append({"Return Period (years)": T, "Expected Loss": f"{loss:,.0f}"})

    col_a, col_b = st.columns([1, 2])
    col_a.dataframe(pd.DataFrame(rp_rows), use_container_width=True)

    rp_vals = [float(r["Expected Loss"].replace(",", "")) for r in rp_rows]
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(return_periods_list, rp_vals, "o-", linewidth=2)
    ax3.set_xscale("log")
    ax3.set_xlabel("Return Period (years)")
    ax3.set_ylabel("Expected Loss")
    ax3.set_title("Loss vs Return Period")
    ax3.grid(True, alpha=0.3)
    for T, v in zip(return_periods_list, rp_vals):
        ax3.annotate(f"{v:,.0f}", xy=(T, v), xytext=(5, 8),
                     textcoords="offset points", fontsize=8)
    col_b.pyplot(fig3)
    plt.close()

# ─────────────────────────────────────────────
# PAGE 3: Treaty Structuring
# ─────────────────────────────────────────────

elif page == "Treaty Structuring":
    st.title("Treaty Structuring")
    st.markdown("Compare Quota Share vs Excess of Loss structures.")

    st.subheader("Quota Share")
    qs_rate = st.slider("Cession rate (%)", 10, 50, 30) / 100
    comm    = st.slider("Commission rate (%)", 15, 35, 25) / 100

    qs_ceded    = individual_losses * qs_rate
    qs_retained = individual_losses * (1 - qs_rate)
    premiums    = individual_losses / 0.70
    qs_prem     = premiums * qs_rate
    qs_comm     = qs_prem * comm

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Ceded",    f"{qs_ceded.sum():,.0f}")
    col2.metric("Ceded Premium",  f"{qs_prem.sum():,.0f}")
    col3.metric("Net Cost",       f"{(qs_prem.sum() - qs_comm.sum()):,.0f}")

    st.markdown("---")
    st.subheader("Excess of Loss (Per-Occurrence)")

    col_r, col_l = st.columns(2)
    retention = col_r.number_input("Retention", 10000, 500000, 100000, 10000)
    limit     = col_l.number_input("Limit",     50000, 2000000, 400000, 50000)

    xl_retained, xl_ceded = apply_xl(individual_losses, retention, limit)
    xl_prem = xl_ceded.sum() * 1.15
    n_attach = (individual_losses > retention).sum()
    n_exhaust= (individual_losses > retention + limit).sum()

    col1x, col2x, col3x, col4x = st.columns(4)
    col1x.metric("Claims Attaching",   f"{n_attach:,}")
    col2x.metric("Claims Exhausting",  f"{n_exhaust:,}")
    col3x.metric("Total Ceded",        f"{xl_ceded.sum():,.0f}")
    col4x.metric("XL Premium (BC+15%)",f"{xl_prem:,.0f}")

    fig4, axes4 = plt.subplots(1, 2, figsize=(13, 4))

    pcts  = [50, 75, 90, 95, 99, 99.5]
    v_ori = [np.percentile(individual_losses, p) for p in pcts]
    v_ret = [np.percentile(xl_retained, p) for p in pcts]
    x4    = np.arange(len(pcts))
    axes4[0].bar(x4 - 0.2, v_ori, 0.4, label="Original", alpha=0.8)
    axes4[0].bar(x4 + 0.2, v_ret, 0.4, label=f"After XL {limit/1e6:.1f}M xs {retention/1e3:.0f}k", alpha=0.8)
    axes4[0].set_xticks(x4)
    axes4[0].set_xticklabels([f"P{p}" for p in pcts])
    axes4[0].set_title("VaR Comparison: Original vs After XL")
    axes4[0].legend()

    thresholds4 = np.linspace(0, individual_losses.max(), 800)
    axes4[1].plot(thresholds4, [np.mean(individual_losses > t) for t in thresholds4],
                  label="Original", linewidth=2)
    axes4[1].plot(thresholds4, [np.mean(xl_retained > t) for t in thresholds4],
                  label=f"After XL", linewidth=2)
    axes4[1].axvline(retention, color="red", linestyle="--", linewidth=1.5, label="Attachment")
    axes4[1].set_yscale("log")
    axes4[1].set_xlabel("Loss Amount")
    axes4[1].set_ylabel("Exceedance Probability")
    axes4[1].set_title("Exceedance Curve: Before vs After XL")
    axes4[1].legend()
    axes4[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

# ─────────────────────────────────────────────
# PAGE 4: Program Optimization
# ─────────────────────────────────────────────

elif page == "Program Optimization":
    st.title("Program Optimization")
    st.markdown(
        "Two-layer program: **Per-occurrence XL** (fixed) + **Aggregate Stop Loss** (adjustable).\n\n"
        "**Objective:** minimize `XL Premium + Stop Loss Premium + CoC × VaR 99.5% (annual retained)`"
    )

    mean_loss = annual_losses.mean()

    st.subheader("Layer 1: Per-Occurrence XL (fixed)")
    xl_ret_fixed = 50_000
    xl_lim_fixed = 950_000
    xl_prem_fixed = burning_cost_premium(individual_losses, xl_ret_fixed, xl_lim_fixed)
    st.info(f"Structure: {xl_lim_fixed/1e6:.1f}M xs {xl_ret_fixed/1e3:.0f}k  |  Premium: {xl_prem_fixed:,.0f}")

    st.markdown("---")
    st.subheader("Layer 2: Aggregate Stop Loss")

    col_a, col_l2 = st.columns(2)
    att_ratio = col_a.slider("Attachment (x mean annual loss)", 1.00, 1.60, 1.29, 0.01)
    lim_ratio = col_l2.slider("Limit (x mean annual loss)",     0.20, 1.00, 0.32, 0.01)

    attachment = mean_loss * att_ratio
    sl_limit   = mean_loss * lim_ratio
    sl_prem    = stop_loss_premium(annual_losses, attachment, sl_limit)

    retained_annual, _ = apply_stop_loss(annual_losses, attachment, sl_limit)
    var_original = np.percentile(annual_losses, 99.5)
    var_retained = np.percentile(retained_annual, 99.5)
    cap_cost     = var_retained * coc
    total_cost   = xl_prem_fixed + sl_prem + cap_cost
    relief_pct   = (var_original - var_retained) / var_original * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Attachment",           f"{attachment/1e6:.2f}M")
    col2.metric("Stop Loss Premium",    f"{sl_prem:,.0f}")
    col3.metric("VaR 99.5% Retained",   f"{var_retained/1e6:.2f}M",
                delta=f"{-relief_pct:.1f}%")
    col4.metric("Total Program Cost",   f"{total_cost/1e6:.2f}M")

    fig5, axes5 = plt.subplots(1, 2, figsize=(13, 5))

    axes5[0].hist(annual_losses / 1e6, bins=70, alpha=0.5, label="Original", edgecolor="black")
    axes5[0].hist(retained_annual / 1e6, bins=70, alpha=0.5,
                  label="After Program", edgecolor="black", color="coral")
    axes5[0].axvline(var_original / 1e6, color="blue",  linestyle="--", linewidth=2,
                     label=f"VaR 99.5% Original: {var_original/1e6:.1f}M")
    axes5[0].axvline(var_retained / 1e6, color="red",   linestyle="--", linewidth=2,
                     label=f"VaR 99.5% Retained: {var_retained/1e6:.1f}M")
    axes5[0].axvline(attachment / 1e6,   color="green", linestyle=":",  linewidth=2,
                     label=f"Attachment: {attachment/1e6:.1f}M")
    axes5[0].set_xlabel("Annual Aggregate Loss (M)")
    axes5[0].set_ylabel("Frequency")
    axes5[0].set_title("Annual Distribution: Before vs After")
    axes5[0].legend(fontsize=8)

    att_range  = np.linspace(mean_loss * 1.0, mean_loss * 1.6, 30)
    costs, vars_ = [], []
    for att in att_range:
        sp   = stop_loss_premium(annual_losses, att, sl_limit)
        ra, _ = apply_stop_loss(annual_losses, att, sl_limit)
        vr   = np.percentile(ra, 99.5)
        costs.append(xl_prem_fixed + sp + vr * coc)
        vars_.append(vr)

    axes5[1].plot(np.array(vars_) / 1e6, np.array(costs) / 1e6, "o-", linewidth=2)
    axes5[1].scatter(var_retained / 1e6, total_cost / 1e6,
                     s=250, c="red", marker="*", zorder=5, label="Current")
    axes5[1].set_xlabel("VaR 99.5% Annual Retained (M)")
    axes5[1].set_ylabel("Total Cost (M)")
    axes5[1].set_title("Efficient Frontier")
    axes5[1].legend()
    axes5[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

# ─────────────────────────────────────────────
# PAGE 5: Stress Testing
# ─────────────────────────────────────────────

elif page == "Stress Testing":
    st.title("Stress Testing")
    st.markdown("Program performance under five adverse scenarios applied to annual aggregate losses.")

    st.subheader("Stop Loss Parameters")
    mean_loss   = annual_losses.mean()
    att_s = st.slider("Attachment (x mean)", 1.00, 1.60, 1.29, 0.01) * mean_loss
    lim_s = st.slider("Limit (x mean)",      0.20, 1.00, 0.32, 0.01) * mean_loss

    np.random.seed(99)
    n = len(annual_losses)

    scenarios = {
        "base":               annual_losses,
        "catastrophe":        np.where(np.random.random(n) < 0.01,
                                       annual_losses * np.random.uniform(3, 8, n),
                                       annual_losses),
        "frequency_shock":    annual_losses * np.random.uniform(1.25, 1.35, n),
        "severity_inflation": annual_losses * 1.20,
        "combined_shock":     (annual_losses * 1.15 * np.random.uniform(1.1, 1.3, n) +
                               np.where(np.random.random(n) < 0.02, annual_losses * 4, 0)),
    }

    rows5 = []
    for name, stressed in scenarios.items():
        retained_s, _ = apply_stop_loss(stressed, att_s, lim_s)
        var_o = np.percentile(stressed,   99.5)
        var_r = np.percentile(retained_s, 99.5)
        rows5.append({
            "Scenario":            name,
            "Mean Annual Loss (M)": f"{stressed.mean()/1e6:.2f}",
            "VaR 99.5% Original (M)": f"{var_o/1e6:.2f}",
            "VaR 99.5% Retained (M)": f"{var_r/1e6:.2f}",
            "Capital Relief":         f"{(var_o - var_r)/var_o*100:.1f}%"
        })

    df_stress = pd.DataFrame(rows5)
    st.dataframe(df_stress, use_container_width=True)

    fig6, axes6 = plt.subplots(1, 2, figsize=(13, 5))

    scenario_names = [r["Scenario"] for r in rows5]
    var_ori_vals   = [float(r["VaR 99.5% Original (M)"].replace(",", "")) for r in rows5]
    var_ret_vals   = [float(r["VaR 99.5% Retained (M)"].replace(",", "")) for r in rows5]
    relief_vals    = [float(r["Capital Relief"].replace("%", "")) for r in rows5]

    x6 = np.arange(len(scenario_names))
    w6 = 0.35
    axes6[0].bar(x6 - w6/2, var_ori_vals, w6, label="Original",  alpha=0.8, color="lightcoral")
    axes6[0].bar(x6 + w6/2, var_ret_vals, w6, label="Retained",  alpha=0.8, color="lightgreen")
    axes6[0].set_xticks(x6)
    axes6[0].set_xticklabels(scenario_names, rotation=25, ha="right")
    axes6[0].set_ylabel("VaR 99.5% Annual (M)")
    axes6[0].set_title("Capital at Risk: Before vs After")
    axes6[0].legend()
    axes6[0].grid(True, alpha=0.3, axis="y")

    colors6 = ["green" if v > 10 else "orange" if v > 0 else "red" for v in relief_vals]
    axes6[1].bar(x6, relief_vals, color=colors6, alpha=0.85, edgecolor="black")
    axes6[1].set_xticks(x6)
    axes6[1].set_xticklabels(scenario_names, rotation=25, ha="right")
    axes6[1].set_ylabel("Capital Relief (%)")
    axes6[1].set_title("Stop Loss Capital Relief by Scenario")
    axes6[1].axhline(y=0, color="black", linewidth=1)
    axes6[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(relief_vals):
        axes6[1].text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig6)
    plt.close()

    st.markdown("---")
    st.subheader("Distribution Comparison by Scenario")
    sel_scenario = st.selectbox("Select scenario", list(scenarios.keys()))
    stressed_sel  = scenarios[sel_scenario]
    retained_sel, _ = apply_stop_loss(stressed_sel, att_s, lim_s)

    fig7, ax7 = plt.subplots(figsize=(11, 4))
    ax7.hist(stressed_sel / 1e6,  bins=70, alpha=0.5, label="Stressed", edgecolor="black")
    ax7.hist(retained_sel / 1e6,  bins=70, alpha=0.5, label="Retained", edgecolor="black", color="coral")
    ax7.axvline(np.percentile(stressed_sel,  99.5) / 1e6, color="blue", linestyle="--", linewidth=2,
                label=f"VaR 99.5% Stressed: {np.percentile(stressed_sel,99.5)/1e6:.1f}M")
    ax7.axvline(np.percentile(retained_sel,  99.5) / 1e6, color="red",  linestyle="--", linewidth=2,
                label=f"VaR 99.5% Retained: {np.percentile(retained_sel,99.5)/1e6:.1f}M")
    ax7.set_xlabel("Annual Aggregate Loss (M)")
    ax7.set_ylabel("Frequency")
    ax7.set_title(f"Scenario: {sel_scenario}")
    ax7.legend(fontsize=9)
    st.pyplot(fig7)
    plt.close()
