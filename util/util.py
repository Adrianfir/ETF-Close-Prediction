"""
This is the util file including the small functions
"""
__author__: str = "Pouya 'Adrian' Firouzmakan"
__all__ = [
    'pre_visualization', 'report_and_visualization', 'feat_label'
]

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def pre_visualization(df):

    plt.figure(dpi=200, figsize=(12, 8))
    sns.lineplot(data=df, x='Date', y='Low', color='b', ls='--', lw=0.5)
    sns.lineplot(data=df, x='Date', y='Close', color='k', lw=0.5)
    sns.lineplot(data=df, x='Date', y='High', color='r', ls='-.', lw=0.5)
    plt.xlabel = 'Date'
    plt.ylabel = 'Close Price'
    plt.grid()
    plt.show()


def report_and_visualization(model, xtrain, ytrain,
                             xtest, ytest, pred):
    print("Model Summary:")
    print("=" * 30)
    print(f"Best Polynomial Degree: {model['poly_feat'].degree}")
    print(f"Alpha: {model['model'].alpha}")
    print(f"L1 Ratio: {model['model'].l1_ratio}")
    print(f"Coefficients: {model['model'].coef_}")
    print(f"Intercept: {model['model'].intercept_}")
    print("\nModel Performance Metrics:")
    print("-" * 30)
    print(f"R-squared: {r2_score(ytest, pred)}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(ytest, pred)}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(ytest, pred)}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(ytest, pred))}")
    print("\nModel Details:")
    print("-" * 30)
    print("Training Data:")
    print(f"  - Number of Observations: {xtrain.shape[0]}")
    print(f"  - Number of Features: {xtrain.shape[1]}")
    print("Testing Data:")
    print(f"  - Number of Observations: {xtest.shape[0]}")
    print(f"  - Number of Features: {xtest.shape[1]}")
    print("=" * 30)

    plt.figure(dpi=100, figsize=(8, 6))
    sns.kdeplot(ytest - pred)
    plt.axhline(y=0, color='r', ls='--')
    plt.grid()
    plt.show()


def feat_label(df):
    return df.drop(['Date', 'Close'], axis=1), df['Close']
