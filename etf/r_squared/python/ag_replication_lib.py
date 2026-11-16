"""Regression helpers shared by ag_replication.py and ag_controls.py."""
import numpy as np
import pandas as pd


def cluster_ols(y, X, g):
    """OLS with standard errors clustered by g (CR1 small-sample factor). X includes the constant."""
    b = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ b
    bread = np.linalg.inv(X.T @ X)
    codes, uniq = pd.factorize(g)
    S = np.zeros((len(uniq), X.shape[1]))
    np.add.at(S, codes, X * e[:, None])
    G, N, K = len(uniq), len(y), X.shape[1]
    V = bread @ (S.T @ S) @ bread * (G / (G - 1)) * ((N - 1) / (N - K))
    return b, np.sqrt(np.diag(V)), 1 - (e @ e) / (((y - y.mean()) ** 2).sum())


def design(df, cols):
    parts = [np.ones((len(df), 1)), df[cols].to_numpy(float)]
    parts.append(pd.get_dummies(df["year"], drop_first=True).to_numpy(float))
    parts.append(pd.get_dummies(df["style"], drop_first=True).to_numpy(float))
    return np.hstack(parts)
