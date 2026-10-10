import numpy as np
from sklearn import linear_model, covariance
'''
# sharpe_ratio = 0.2111449903051779
# avg_turnover_rate = 0.664883511511584725

def OLS(returns, factRet):
    """Performs OLS regression to estimate expected returns and covariance."""
    T, p = factRet.shape
    X = np.concatenate([np.ones((T, 1)), factRet.values], axis=1)
    B = np.linalg.lstsq(X, returns, rcond=None)[0]
    a, V = B[0, :], B[1:, :]
    residuals = returns - X @ B
    sigma_ep = np.var(residuals, axis=0, ddof=p+1)
    D = np.diag(sigma_ep)
    mu = a.reshape(-1, 1) + V.T @ np.mean(factRet, axis=0).values.reshape(-1, 1)
    Q = V.T @ np.cov(factRet.T) @ V + D
    return mu, (Q + Q.T) / 2
    '''

def OLS(returns, factRet):
    """Performs OLS regression to estimate expected returns and covariance."""
    T, p = factRet.shape
    X = np.hstack([np.ones((T, 1)), factRet.values])
    
    # Solve for B using pseudo-inverse
    B = np.linalg.pinv(X) @ returns
    a, V = B[0, :], B[1:, :]

    # Compute residual variance
    residuals = returns - X @ B
    sigma_ep = np.var(residuals, axis=0, ddof=T-p-1)  # Adjusted ddof

    D = np.diag(sigma_ep)
    mu = a.reshape(-1, 1) + V.T @ np.mean(factRet, axis=0).values.reshape(-1, 1)
    Q = V.T @ np.cov(factRet.T) @ V + D
    return mu, (Q + Q.T) / 2


def Lasso(returns, factRet):
    # Performs LASSO regression to find the coefficients of all factors per asset
    # Uses coefficients to find expected returns
   
    # Number of observations and factors
    [T, p] = factRet.shape
    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)
    reg = linear_model.LassoCV(random_state=0)
    # Initialize coefficient matrix
    B = []
    # Loop through each asset to find coefficients and append into matrix
    for asset in returns:
        reg.fit(X, returns[asset])
        B.append(reg.coef_)
    B = np.array(B).T
    # Separate B into alpha and betas
    a = B[0, :]
    V = B[1:, :]
    # Residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)
    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values
    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D
    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2
    return mu, Q


def Ridge(returns, factRet):
    """Performs Ridge regression to estimate expected returns and covariance."""
    T, p = factRet.shape
    X = np.concatenate([np.ones((T, 1)), factRet.values], axis=1)
    reg = linear_model.RidgeCV(alphas=np.logspace(-2, 2, 150))
    
    B = np.array([reg.fit(X, returns[asset]).coef_ for asset in returns]).T
    
    a, V = B[0, :], B[1:, :]
    residuals = returns - X @ B
    sigma_ep = np.var(residuals, axis=0, ddof=p+1)
    D = np.diag(sigma_ep)
    
    mu = a.reshape(-1, 1) + V.T @ np.mean(factRet, axis=0).values.reshape(-1, 1)
    Q = V.T @ np.cov(factRet.T) @ V + D

    return mu, (Q + Q.T) / 2