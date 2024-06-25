import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import util.util as util
from sklearn.model_selectrion import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardSclaer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline


# open the file as a DataFrame (here the file is a .txt related to a ETF called SQQQ)

df = pd.read_csv(config['path'], delimiter=',')


# Lets take a look ta the data
print(df.info())		
print(df.is_null())
print(df.head(2))
print(df.describe().transpose())
util.visualization(df)

x, y = util.feat_label(df)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = config['test_size'], 
	random_state = config['seed'])

poly_feat = PolynomialFeatures()
scaler = StandardSclaer()
model = ElasticNet()

operations = [
			  ('poly_feat', poly_feat), 
			  ('scaler', sclaer), 
			  ('model', model)
			  ]

pipe = Pipeline(operations)

params = {'model__alpha': config['model']['lin_reg']['alpha'],
		  'model__l1_ratio': config['model']['lin_reg']['l1_ration'],
		  'poly_feat__degree': config['model']['lin_reg']['poly_deg']
		  }
grid_model = GridSearchCV(estimator=model, params=params,
						  cv=config['grid_s']['cv'], 
						  scoring=config['grid_s']['score'],
						  verbose=1)

final_model = grid_model.fit(xtrain, ytrain)

pred = final_model.predict(xtest)

res_error = ytest - pred
util.res_visualization(res_error)













