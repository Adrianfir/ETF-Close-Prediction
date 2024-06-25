"""
This is the util file including the small functions
"""
__author__: str = "Pouya 'Adrian' Firouzmakan"
__all__ = [
           'pre_visualization', 'res_visualization', 'feat_label'
           ]



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error


def pre_visualization(df):

	"""
	this function is just for visualizing the dataset befor training
	:param df:
	:param x:
	:param y_1:
	:param y_2:
	:param y_3:

	return:

	"""

	fig = plt.figure(dpi=200, figsize=(12,8))
	sns.lineplot(df['Date'], df['Low'], color='b', ls='--')
	sns.lineplot(df['Date'], df['Close'], color='k')
	sns.lineplot(df['Date'], df['High'], color='r', ls='-.')
	plt.xlabel='Date'
	plt.ylabel='Close Price'
	plt.grid()
	plt.show()


def res_visualization(y_est, pred):

	"""
	this function to see the quality of the linear model
	:param ress_error: this is the residual error 

	return:

	"""
	print('\n\n\n')
	print(f'MAE: {mean_absolute_error(ytest, pred)}')
	print(f'MSE:{mean_squared_error(ytest, pred)}')
	print(f'MSE:{mean_squared_error(np.sqrt(ytest, pred))}')
	fig = plt.figure(dpi=200, figsize=(12,8))
	sns.kdeplot(ytest - pred)
	plt.axhline(y=0, color='r', ls='--')
	plt.show()


def feat_label(df):

	"""
	this function is to determine features and the label
	:param df:

	return:
	"""
	return df['Close'], df.drop(['Date', 'Close'], axis=1)


