import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class ExtremeValueAnalysis:
    """
    Classe para modelagem de eventos extremos usando
    Generalized Pareto Distribution (GPD)
    """
    
    def __init__(self, data):
        self.data = np.array(data)
        self.threshold = None
        self.exceedances = None
        self.gpd_params = None
        
    def select_threshold(self, method='percentile', value=95):
        """
        Seleciona threshold para POT (Peaks Over Threshold)
        
        Args:
            method: 'percentile' ou 'value'
            value: valor do percentil ou threshold absoluto
        
        Returns:
            threshold selecionado
        """
        if method == 'percentile':
            self.threshold = np.percentile(self.data, value)
        elif method == 'value':
            self.threshold = value
        else:
            raise ValueError("method must be 'percentile' or 'value'")
            
        self.exceedances = self.data[self.data > self.threshold] - self.threshold
        
        n_exceed = len(self.exceedances)
        pct_exceed = n_exceed / len(self.data) * 100
        
        print(f"Threshold selected: {self.threshold:,.2f}")
        print(f"Number of exceedances: {n_exceed} ({pct_exceed:.2f}%)")
        
        return self.threshold
    
    def fit_gpd(self):
        """
        Ajusta Generalized Pareto Distribution
        
        Returns:
            tuple (shape, loc, scale)
        """
        if self.exceedances is None:
            raise ValueError("Must select threshold first using select_threshold()")
        
        self.gpd_params = stats.genpareto.fit(self.exceedances)
        shape, loc, scale = self.gpd_params
        
        print(f"\nGPD Parameters fitted:")
        print(f"  Shape (xi): {shape:.4f}")
        print(f"  Scale (sigma): {scale:,.2f}")
        print(f"  Location: {loc:,.2f}")
        
        if shape > 0:
            print(f"  Interpretation: Heavy tail (Pareto-type)")
        elif abs(shape) < 0.01:
            print(f"  Interpretation: Exponential tail")
        else:
            print(f"  Interpretation: Light tail (bounded)")
            
        return self.gpd_params
    
    def calculate_var_tvar(self, confidence_levels=[0.95, 0.99, 0.995]):
        """
        Calcula VaR e TVaR usando modelo GPD
        
        Args:
            confidence_levels: lista de níveis de confiança
        
        Returns:
            DataFrame com métricas de risco
        """
        if self.gpd_params is None:
            raise ValueError("Must fit GPD first using fit_gpd()")
        
        shape, loc, scale = self.gpd_params
        n = len(self.data)
        n_exceed = len(self.exceedances)
        
        results = []
        
        for alpha in confidence_levels:
            # VaR usando GPD
            if abs(shape) > 1e-6:
                var_gpd = self.threshold + (scale/shape) * (((1-alpha)/(n_exceed/n))**(-shape) - 1)
            else:
                var_gpd = self.threshold + scale * np.log((1-alpha)/(n_exceed/n))
            
            # TVaR (Expected Shortfall)
            if shape < 1 and shape >= 0:
                tvar_gpd = var_gpd / (1 - shape) + (scale - shape * self.threshold) / (1 - shape)
            else:
                tvar_gpd = np.nan
            
            # VaR/TVaR empírico para comparação
            var_empirical = np.percentile(self.data, alpha * 100)
            tvar_empirical = self.data[self.data > var_empirical].mean()
            
            results.append({
                'confidence': alpha,
                'VaR_GPD': var_gpd,
                'VaR_Empirical': var_empirical,
                'TVaR_GPD': tvar_gpd,
                'TVaR_Empirical': tvar_empirical
            })
        
        return pd.DataFrame(results)
    
    def goodness_of_fit(self):
        """
        Testa qualidade do ajuste GPD usando Kolmogorov-Smirnov
        
        Returns:
            tuple (theoretical_quantiles, empirical_quantiles, ks_statistic, p_value)
        """
        if self.gpd_params is None:
            raise ValueError("Must fit GPD first")
        
        shape, loc, scale = self.gpd_params
        
        # Q-Q Plot quantiles
        theoretical_quantiles = stats.genpareto.ppf(
            np.linspace(0.01, 0.99, len(self.exceedances)),
            shape, loc, scale
        )
        empirical_quantiles = np.sort(self.exceedances)
        
        # Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.kstest(
            self.exceedances,
            lambda x: stats.genpareto.cdf(x, shape, loc, scale)
        )
        
        print(f"\nKolmogorov-Smirnov Test:")
        print(f"  Statistic: {ks_statistic:.4f}")
        print(f"  P-value: {p_value:.4f}")
        
        if p_value > 0.05:
            print(f"  Result: Do not reject H0 - GPD is a good fit (p > 0.05)")
        else:
            print(f"  Result: Reject H0 - GPD may not be ideal (p < 0.05)")
        
        return theoretical_quantiles, empirical_quantiles, ks_statistic, p_value
    
    def return_period_loss(self, return_periods=[10, 50, 100, 200]):
        """
        Calcula perda esperada para diferentes períodos de retorno
        
        Args:
            return_periods: lista de períodos em anos
        
        Returns:
            DataFrame com perdas por return period
        """
        if self.gpd_params is None:
            raise ValueError("Must fit GPD first")
        
        shape, loc, scale = self.gpd_params
        n = len(self.data)
        n_exceed = len(self.exceedances)
        
        results = []
        
        for T in return_periods:
            p_annual = 1 / T
            
            if abs(shape) > 1e-6:
                loss = self.threshold + (scale/shape) * ((p_annual * n / n_exceed)**(-shape) - 1)
            else:
                loss = self.threshold + scale * np.log(p_annual * n / n_exceed)
            
            results.append({
                'return_period': T,
                'annual_prob': p_annual,
                'expected_loss': loss
            })
        
        return pd.DataFrame(results)

def hill_estimator(data, k_min=10, k_max=None):
    """
    Hill estimator para índice de cauda (tail index)
    
    Args:
        data: array de dados
        k_min: mínimo k para estimar
        k_max: máximo k (default: n/2)
    
    Returns:
        tuple (k_values, hill_estimates)
    """
    sorted_data = np.sort(data)[::-1]
    n = len(sorted_data)
    
    if k_max is None:
        k_max = min(500, n // 2)
    
    k_values = range(k_min, k_max)
    hill_estimates = []
    
    for k in k_values:
        top_k = sorted_data[:k]
        hill = np.mean(np.log(top_k)) - np.log(sorted_data[k])
        hill_estimates.append(hill)
    
    return list(k_values), hill_estimates