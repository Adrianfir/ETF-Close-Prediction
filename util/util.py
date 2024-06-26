"""
This is the util file including the small functions
"""
__author__: str = "Pouya 'Adrian' Firouzmakan"
__all__ = [
    'feat_label', 'final_visualization', 'pre_visualization', 'report'
]

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config.config import config


def pre_visualization(df):
    plt.figure(dpi=200, figsize=(12, 8))
    sns.lineplot(data=df, x='Date', y='Low', color='b', ls='--', lw=0.5)
    sns.lineplot(data=df, x='Date', y='Close', color='k', lw=0.5)
    sns.lineplot(data=df, x='Date', y='High', color='r', ls='-.', lw=0.5)
    plt.xlabel = 'Date'
    plt.ylabel = 'Close Price'
    plt.grid()
    plt.show()


def report(model, xtrain, xtest, ytest, pred):
    lines = list()
    lines.append("Model Summary:")
    lines.append("=" * 30)
    lines.append(f"Best Polynomial Degree: {model['poly_feat'].degree}")
    lines.append(f"Alpha: {model['model'].alpha}")
    lines.append(f"L1 Ratio: {model['model'].l1_ratio}")
    lines.append(f"Coefficients: {model['model'].coef_}")
    lines.append(f"Intercept: {model['model'].intercept_}")
    lines.append("\nModel Performance Metrics:")
    lines.append("-" * 30)
    lines.append(f"R-squared: {r2_score(ytest, pred)}")
    lines.append(f"Mean Absolute Error (MAE): {mean_absolute_error(ytest, pred)}")
    lines.append(f"Mean Squared Error (MSE): {mean_squared_error(ytest, pred)}")
    lines.append(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(ytest, pred))}")
    lines.append("\nModel Details:")
    lines.append("-" * 30)
    lines.append("Training Data:")
    lines.append(f"  - Number of Observations: {xtrain.shape[0]}")
    lines.append(f"  - Number of Features: {xtrain.shape[1]}")
    lines.append("Testing Data:")
    lines.append(f"  - Number of Observations: {xtest.shape[0]}")
    lines.append(f"  - Number of Features: {xtest.shape[1]}")
    lines.append("=" * 30)

    with open(config['output_path'], 'w') as file:
        for l in lines:
            file.write(l + "\n\n")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), dpi=200)
    sns.kdeplot(pred - ytest, ax=axes[0])
    axes[0].set_title('Distribution of the residual error')
    axes[0].grid()

    sns.scatterplot(x=pred, y=pred - ytest, ax=axes[1])
    axes[1].axhline(y=0, ls='--', color='r')
    axes[1].set_title('prediction vs (prediction-ytest)')
    axes[1].grid()
    plt.savefig('report/res_dist.png')


def feat_label(df):
    return df.drop(['Date', 'Close'], axis=1), df['Close']


def final_visualization(model_f, df, x):
    df['y_pred'] = model_f.predict(x)
    plt.figure(dpi=200, figsize=(8, 6))
    sns.lineplot(data=df, x='Date', y='y_pred', ls='-.', color='b', label='predicted')
    sns.lineplot(data=df, x='Date', y='Close', ls='--', color='r', label='expected')
    plt.legend()
    plt.grid()
    plt.title('comparing the predicted and the expected closed-prices')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.show()


