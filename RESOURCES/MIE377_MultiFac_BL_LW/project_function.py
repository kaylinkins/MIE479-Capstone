'''
# sharpe_ratio = 0.2111449903051779
# avg_turnover_rate = 0.664883511511584725
'''
from services.strategies import *

def project_function(periodReturns, periodFactRet, x0):
    """
    Executes the selected portfolio strategy.
    :param periodReturns: Historical asset returns (DataFrame)
    :param periodFactRet: Factor returns (DataFrame)
    :param x0: Initial portfolio weights (array)
    :return: New portfolio allocation (array)
    """
    Strategy = BLMultiFactorMVO(use_ridge=True)  # Use LASSO by default
    x = Strategy.execute_strategy(periodReturns, periodFactRet, x0)
    '''
    # Normalize to ensure valid portfolio weights
    x = np.clip(x, 0, None)  # Ensure no short positions (long-only)
    x /= np.sum(x)  # Normalize weights to sum to 1
    '''
    return x
