import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import util.util as util
from sklearn.model_selectrion import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardSclaer
from sklearn.linear_model import LinearRegressopn, ElasticNet
from sklearn.pipeline import Pipeline



# open the file as a DataFrame (here the file is a .txt related to a ETF called SQQQ)

file_path = input('please enter the path to the file: ')
df = pd.read_csv(file_path, delimiter=',')


# Lets take a look ta the data
print(df.info())		
print(df.is_null())
print(df.head(2))
print(df.describe().transpose())
util.visualization(df)

x, y = util.feat_label(df)
x = PolynomialFeatures(x, degree=config["degree"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = config['test_size'], 
	random_state = config['seed'])














