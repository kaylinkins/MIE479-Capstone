'''
# sharpe_ratio = 0.2111449903051779
# avg_turnover_rate = 0.664883511511584725
 '''
import numpy as np
from services.estimators import *
from services.optimization import MVO
from sklearn import linear_model, covariance
from scipy.linalg import inv

class EqualWeight:
    """Equal-weighted portfolio strategy."""
    @staticmethod
    def execute_strategy(periodReturns, *args):
        T, n = periodReturns.shape
        return np.ones(n) / n

class MultiFactorMVO:
    """Multi-Factor Model with Mean-Variance Optimization."""
    def __init__(self, NumObs=36, use_ridge=True):
        self.NumObs = NumObs
        self.use_ridge = use_ridge  # Choose model dynamically

    def execute_strategy(self, periodReturns, periodFactRet, x0):
        returns = periodReturns.iloc[-self.NumObs:, :]
        factRet = periodFactRet.iloc[-self.NumObs:, :]

        # Select OLS or Lasso based on data shape
        if self.use_ridge or factRet.shape[1] > returns.shape[0]:
            mu, Q = Ridge(returns, factRet)
        else:
            mu, Q = Lasso(returns, factRet)
        
        # Boost momentum factor influence
        if 'Mom   ' in factRet.columns:
            mom_weight = 0.65  # Increase influence of Momentum
            mu += mom_weight * factRet['Mom   '].mean()
        '''
        # Boost momentum factor influence
        if 'ST_Rev' in factRet.columns:
            ST_weight = 0.1  # Increase influence of Momentum
            mu += ST_weight * factRet['ST_Rev'].mean()
        '''
        '''
        lw = covariance.LedoitWolf()
        Q = lw.fit(returns).covariance_
        '''
        return MVO(mu, Q, x0, lambda_=0.3)

class BLMultiFactorMVO:
    """Multi-Factor Model with Black-Litterman and Mean-Variance Optimization."""
    def __init__(self, NumObs=60, tau=0.05, use_ridge=True):
        self.NumObs = NumObs
        self.tau = tau  # Confidence in prior (small tau = more market reliance)
        self.use_ridge = use_ridge  # Choose model dynamically

    def execute_strategy(self, periodReturns, periodFactRet, x0):
        returns = periodReturns.iloc[-self.NumObs:, :]
        factRet = periodFactRet.iloc[-self.NumObs:, :]

        # 1. Compute Factor-based Expected Returns (mu)
        mu_factor, _ = Ridge(returns, factRet)
        '''
        # Boost momentum factor influence
        if 'Mom   ' in factRet.columns:
            mom_weight = 0.3  # Increase influence of Momentum
            mu_factor += mom_weight * factRet['Mom   '].mean()
        '''
        # 2. Compute Market-Implied Equilibrium Returns (pi)
        market_weights = np.ones(returns.shape[1]) / returns.shape[1]  # Equal-weighted market proxy
        cov_market = np.cov(returns, rowvar=False)
        pi = self.tau * cov_market @ market_weights

        # 3. Combine with Black-Litterman Formula
        P = np.eye(len(pi))  # Assume equal confidence in all factors
        Q = mu_factor  # Factor-based views
        Omega = np.diag(np.var(returns, axis=0))  # Uncertainty in views

        inv_part = inv(inv(self.tau * cov_market) + P.T @ inv(Omega) @ P)
        mu_bl = inv_part @ (inv(self.tau * cov_market) @ pi + P.T @ inv(Omega) @ Q)
        
        # 4. Use Ledoit-Wolf for Covariance Estimation
        lw = covariance.LedoitWolf()
        Q = lw.fit(returns).covariance_
        #print(Q.shape)
        #print(mu_bl)
        mu_bl_unique = mu_bl[:, 0]
        
        #print(f"mu_bl shape: {mu_bl.flatten().shape}, Q shape: {Q.shape}, x0 shape: {x0.shape}")
        return MVO(mu_bl_unique, Q, x0, lambda_=0.09)
