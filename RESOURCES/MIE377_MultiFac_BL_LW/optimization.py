'''
# sharpe_ratio = 0.2111449903051779
# avg_turnover_rate = 0.664883511511584725
'''
import numpy as np
import cvxpy as cp

def MVO(mu, Q, x0=None, lambda_=3):
    """Performs Mean-Variance Optimization with risk aversion parameter lambda_ and turnover constraint."""
    n = len(mu)
    x = cp.Variable(n)
    targetRet = np.mean(mu)
    
    # Use lambda_ for risk-return tradeoff
    objective = cp.Minimize((lambda_/2) * cp.quad_form(x, Q) - mu.T @ x)
    
    # Define the constraints
    constraints = [
        cp.sum(x) == 1,  # Weights must sum to 1
        mu.T @ x >= targetRet  # Target return constraint
    ]
    '''
    # Add turnover constraint if initial weights are provided
    if x0 is not None:
        constraints.append(cp.norm(x - x0, 1) <= 0.6)
    '''
    # Define the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve()
    
    return x.value

