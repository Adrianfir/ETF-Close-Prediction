"""
This is the util file including the small functions
"""
__author__: str = "Pouya 'Adrian' Firouzmakan"
__all__ = [
           'visualization'
           ]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualization(df, , x='Date', y_1='Close', y_2='High', y_3='Low'):
	""
	this function is just for visualizing the dataset befor training
	:param df:
	:param x:
	:param y_1:
	:param y_2:
	:param y_3:

	return:

	""

	fig = plt.figure(dpi=200, figsize=(12,8))
	sns.lineplot(df['Date'], df['Low'], color='b', ls='--')
	sns.lineplot(df['Date'], df['Close'], color='k')
	sns.lineplot(df['Date'], df['High'], color='r', ls='-.')
	plt.xlabel='Date'
	plt.ylabel='Close Price'
	plt.grid()
	plt.show()

def feat_label(df):
	""
	this function is to determine features and the label
	:param df:

	return:
	""
	return df['Close'], df.drop(['Date', 'Close'], axis=1)


