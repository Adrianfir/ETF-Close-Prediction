import pandas as pd
import util.util as util
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from config.config import config

# Open the file as a DataFrame (assuming the file is a .txt related to a ETF called SQQQ)
df = pd.read_csv(config['data']['path'], delimiter=',')
# Let's take a look at the data
print(df.info())
print(df.head(2))
print(df.describe().transpose())
util.pre_visualization(df)

# Feature and label extraction
x, y = util.feat_label(df)
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=config['test_size'],
                                                random_state=config['seed']
                                                )

# Define the pipeline
poly_feat = PolynomialFeatures()
scaler = StandardScaler()
model = ElasticNet()

operations = [
    ('poly_feat', poly_feat),
    ('scaler', scaler),
    ('model', model)
]

pipe = Pipeline(operations)

# Define the parameter grid
params = {
    'model__alpha': config['model']['lin_reg']['alpha'],
    'model__l1_ratio': config['model']['lin_reg']['l1_ratio'],
    'poly_feat__degree': config['model']['lin_reg']['poly_deg']
}

# Pass the pipeline to GridSearchCV
grid_model = GridSearchCV(estimator=pipe,
                          param_grid=params,
                          cv=config['grid_s']['cv'],
                          scoring=config['grid_s']['score'],
                          verbose=2)

# Fit the model
final_model = grid_model.fit(xtrain, ytrain)
print(f'The chosen hyperparameters are: {final_model.best_params_}')
best_estimator = final_model.best_estimator_

# Predictions
prediction = best_estimator.predict(xtest)
res_error = ytest - prediction

# Visualization and reporting
util.report_and_visualization(ytest, prediction)
