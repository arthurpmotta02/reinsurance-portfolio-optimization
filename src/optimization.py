import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from treaty_simulator import ExcessOfLoss, ReinsuanceProgram


def simulate_annual_losses(losses, n_years=10000, avg_claims_per_year=5000, random_state=42):
    """
    Simula perdas agregadas anuais via Monte Carlo.
    Cada ano: n_claims ~ Poisson(avg) severidades amostradas empiricamente.
    """
    np.random.seed(random_state)
    annual_totals = np.zeros(n_years)
    for i in range(n_years):
        n = np.random.poisson(avg_claims_per_year)
        if n > 0:
            annual_totals[i] = np.random.choice(losses, size=n, replace=True).sum()

    print(f"Simulated {n_years:,} annual loss years")
    print(f"Mean annual loss:    {annual_totals.mean():>15,.2f}")
    print(f"VaR 99%:             {np.percentile(annual_totals, 99):>15,.2f}")
    print(f"VaR 99.5%:           {np.percentile(annual_totals, 99.5):>15,.2f}")
    print(f"VaR 99.9%:           {np.percentile(annual_totals, 99.9):>15,.2f}")
    print(f"Max annual loss:     {annual_totals.max():>15,.2f}")
    return annual_totals


class ReinsuanceOptimizer:
    """
    Otimiza programa de resseguro em duas camadas:

    Layer 1 — Per-occurrence XL: corta sinistros individuais grandes
              Prêmio = burning cost sobre sinistros individuais
              Efeito = reduz severidade individual, não impacta muito VaR agregado

    Layer 2 — Aggregate Stop Loss (Aggregate XL): protege diretamente
              a perda anual agregada acima de um attachment anual
              Efeito = reduz VaR 99.5% anual significativamente

    Total Cost = Premium_XL + Premium_StopLoss + Cost_of_Capital x VaR_99.5_retained

    O otimizador encontra o attachment do Stop Loss que minimiza custo total.
    """

    def __init__(self, individual_losses, annual_losses, cost_of_capital=0.08):
        self.individual_losses = individual_losses
        self.annual_losses     = annual_losses
        self.cost_of_capital   = cost_of_capital
        self.optimization_results = None

        # XL per-occurrence fixo (reduz sinistros catastróficos individuais)
        self.per_occ_xl = ExcessOfLoss(retention=50_000, limit=950_000)
        per_occ_result  = self.per_occ_xl.apply(individual_losses)
        self.xl_premium = per_occ_result['total_ceded'] * 1.15
        self.retained_individual = per_occ_result['retained_losses']

    def _apply_aggregate_stop_loss(self, annual_losses, attachment, limit):
        """
        Aggregate Stop Loss: reinsurer paga perdas anuais entre attachment e attachment+limit.
        attachment e limit são valores absolutos (não ratios).
        """
        ceded    = np.clip(annual_losses - attachment, 0, limit)
        retained = annual_losses - ceded
        return retained, ceded

    def _stop_loss_premium(self, annual_losses, attachment, limit, loading=0.25):
        """
        Prêmio de stop loss = E[ceded] x (1 + loading).
        Loading mais alto (25%) porque stop loss é estrutura mais cara.
        """
        _, ceded = self._apply_aggregate_stop_loss(annual_losses, attachment, limit)
        return ceded.mean() * (1 + loading)

    def total_cost(self, params):
        attachment, limit = params

        if attachment <= 0 or limit <= 0:
            return 1e15

        # Prêmio stop loss
        sl_premium = self._stop_loss_premium(self.annual_losses, attachment, limit)

        # Capital após stop loss
        retained_annual, _ = self._apply_aggregate_stop_loss(
            self.annual_losses, attachment, limit
        )
        capital_required = np.percentile(retained_annual, 99.5)
        capital_cost     = capital_required * self.cost_of_capital

        return self.xl_premium + sl_premium + capital_cost

    def optimize_stop_loss(self, mean_loss=None):
        """
        Otimiza attachment e limit do Stop Loss.
        Attachment tipicamente entre 100% e 150% da perda média anual.
        Limit tipicamente entre 50% e 100% da perda média anual.
        """
        if mean_loss is None:
            mean_loss = self.annual_losses.mean()

        attachment_bounds = (mean_loss * 1.00, mean_loss * 1.60)
        limit_bounds      = (mean_loss * 0.30, mean_loss * 1.00)
        bounds = [attachment_bounds, limit_bounds]

        print(f"Mean annual loss: {mean_loss:,.0f}")
        print(f"Attachment bounds: ({attachment_bounds[0]:,.0f}, {attachment_bounds[1]:,.0f})")
        print(f"Limit bounds:      ({limit_bounds[0]:,.0f}, {limit_bounds[1]:,.0f})\n")
        print("Optimizing Aggregate Stop Loss...")

        result = differential_evolution(
            self.total_cost,
            bounds=bounds,
            seed=42,
            maxiter=200,
            tol=1e-7,
            disp=True,
        )

        attachment, limit = result.x

        retained_annual, ceded_annual = self._apply_aggregate_stop_loss(
            self.annual_losses, attachment, limit
        )

        sl_premium       = self._stop_loss_premium(self.annual_losses, attachment, limit)
        capital_required = np.percentile(retained_annual, 99.5)
        capital_cost     = capital_required * self.cost_of_capital

        self.optimization_results = {
            # Per-occurrence XL
            'xl_retention':          self.per_occ_xl.retention,
            'xl_limit':              self.per_occ_xl.limit,
            'xl_premium':            self.xl_premium,
            # Aggregate Stop Loss
            'sl_attachment':         attachment,
            'sl_limit':              limit,
            'sl_premium':            sl_premium,
            'sl_attachment_ratio':   attachment / mean_loss,
            # Capital
            'capital_required':      capital_required,
            'capital_cost':          capital_cost,
            'total_cost':            result.fun,
            # Métricas
            'var_99_5_original':     np.percentile(self.annual_losses, 99.5),
            'var_99_5_retained':     capital_required,
            'var_99_original':       np.percentile(self.annual_losses, 99),
            'var_99_retained':       np.percentile(retained_annual, 99),
            'retained_annual':       retained_annual,
            'ceded_annual':          ceded_annual,
            'mean_loss':             mean_loss,
        }

        relief = (self.optimization_results['var_99_5_original'] - capital_required)
        relief_pct = relief / self.optimization_results['var_99_5_original'] * 100

        print(f"\nOptimization complete.")
        print(f"\n  Per-occurrence XL: {self.per_occ_xl.limit/1e6:.1f}M xs {self.per_occ_xl.retention/1e3:.0f}k")
        print(f"  XL Premium:                    {self.xl_premium:>15,.2f}")
        print(f"\n  Aggregate Stop Loss:")
        print(f"  Attachment:                    {attachment:>15,.0f}  ({attachment/mean_loss:.2f}x mean)")
        print(f"  Limit:                         {limit:>15,.0f}  ({limit/mean_loss:.2f}x mean)")
        print(f"  Stop Loss Premium:             {sl_premium:>15,.2f}")
        print(f"\n  VaR 99.5% original:            {self.optimization_results['var_99_5_original']:>15,.2f}")
        print(f"  VaR 99.5% retained:            {capital_required:>15,.2f}")
        print(f"  Capital relief:                {relief_pct:>14.1f}%")
        print(f"\n  Total Cost:                    {result.fun:>15,.2f}")

        return self.optimization_results

    def efficient_frontier(self, attachment_range=None, limit_fixed=None):
        """Frontier variando attachment, limit fixo."""
        mean_loss = self.annual_losses.mean()

        if attachment_range is None:
            attachment_range = np.linspace(mean_loss * 1.0, mean_loss * 1.6, 20)
        if limit_fixed is None:
            limit_fixed = mean_loss * 0.5

        records = []
        for att in attachment_range:
            sl_prem = self._stop_loss_premium(self.annual_losses, att, limit_fixed)
            retained_annual, _ = self._apply_aggregate_stop_loss(
                self.annual_losses, att, limit_fixed
            )
            cap_req  = np.percentile(retained_annual, 99.5)
            cap_cost = cap_req * self.cost_of_capital
            total    = self.xl_premium + sl_prem + cap_cost

            records.append({
                'attachment':           att,
                'attachment_ratio':     att / mean_loss,
                'var_99_5':             cap_req,
                'xl_premium':           self.xl_premium,
                'sl_premium':           sl_prem,
                'capital_cost':         cap_cost,
                'total_cost':           total,
            })

        return pd.DataFrame(records)


def stress_test_program(annual_losses, optimizer, scenarios):
    """
    Stress testing usando perdas anuais e estrutura otimizada (XL + Stop Loss).
    """
    res = optimizer.optimization_results
    records = []

    for name, stress_fn in scenarios.items():
        stressed = stress_fn(annual_losses)

        # Aplicar stop loss otimizado
        retained, ceded = optimizer._apply_aggregate_stop_loss(
            stressed,
            res['sl_attachment'],
            res['sl_limit']
        )

        var_orig = np.percentile(stressed, 99.5)
        var_ret  = np.percentile(retained, 99.5)
        relief   = (var_orig - var_ret) / var_orig * 100 if var_orig > 0 else 0

        records.append({
            'scenario':             name,
            'mean_annual_loss':     stressed.mean(),
            'var_99':               np.percentile(stressed, 99),
            'var_99_5':             var_orig,
            'var_99_9':             np.percentile(stressed, 99.9),
            'var_99_5_retained':    var_ret,
            'capital_relief_pct':   relief,
        })

    return pd.DataFrame(records)


def create_stress_scenarios(annual_losses, random_state=99):
    """Cenários de stress sobre perdas anuais agregadas."""
    np.random.seed(random_state)
    n = len(annual_losses)

    scenarios = {
        'base':              lambda x: x,
        'catastrophe':       lambda x: np.where(
                                 np.random.random(len(x)) < 0.01,
                                 x * np.random.uniform(3, 8, len(x)), x),
        'frequency_shock':   lambda x: x * np.random.uniform(1.25, 1.35, len(x)),
        'severity_inflation':lambda x: x * 1.20,
        'combined_shock':    lambda x: (x * 1.15 *
                                 np.random.uniform(1.1, 1.3, len(x)) +
                                 np.where(np.random.random(len(x)) < 0.02, x * 4, 0)),
    }
    return scenarios