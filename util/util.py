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
from sklearn.metrics import mean_squared_error, mean_absolute_error


def pre_visualization(df):

    plt.figure(dpi=200, figsize=(12, 8))
    sns.lineplot(data=df, x='Date', y='Low', color='b', ls='--')
    sns.lineplot(data=df, x='Date', y='Close', color='k')
    sns.lineplot(data=df, x='Date', y='High', color='r', ls='-.')
    plt.xlabel = 'Date'
    plt.ylabel = 'Close Price'
    plt.grid()
    plt.show()


def report_and_visualization(ytest, pred):

    print('\n\n\n')
    print(f'MAE: {mean_absolute_error(ytest, pred)}')
    print(f'MSE:{mean_squared_error(ytest, pred)}')
    print(f'MSE:{mean_squared_error(np.sqrt(ytest, pred))}')
    fig = plt.figure(dpi=200, figsize=(12, 8))
    sns.kdeplot(ytest - pred)
    plt.axhline(y=0, color='r', ls='--')
    plt.show()


def feat_label(df):
    return df['Close'], df.drop(['Date', 'Close'], axis=1)
