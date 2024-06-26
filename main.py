import pandas as pd
import util.util as util
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from config.config import config
import joblib
import os

# Open the file as a DataFrame (assuming the file is a .txt related to a ETF called SQQQ)
df = pd.read_csv(config['data']['path'], delimiter=',')
# Let's take a look at the data
print(df.info())
print(df.head(2))
print(df.describe().transpose())
# util.pre_visualization(df)

if __name__ == '__main__':
    x, y = util.feat_label(df)
    # Feature and label extraction
    xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                    test_size=config['test_size'],
                                                    random_state=config['seed']
                                                    )

    # Define the pipeline
    poly_feat = PolynomialFeatures()
    scaler = StandardScaler()
    model = ElasticNet()
    # in the operations, attention that the scaler must be before the poly_feat
    operations = [
        ('scaler', scaler),
        ('poly_feat', poly_feat),
        ('model', model)
    ]

    pipe = Pipeline(operations)

    # param_grid definition
    params = {
        'model__alpha': config['model']['lin_reg']['alpha'],
        'model__l1_ratio': config['model']['lin_reg']['l1_ratio'],
        'model__max_iter': config['model']['lin_reg']['max_iter'],
        'poly_feat__degree': config['model']['lin_reg']['poly_deg'],

    }

    # let's pass the pipeline to gridsearch
    grid_model = GridSearchCV(estimator=pipe,
                              param_grid=params,
                              cv=config['grid_s']['cv'],
                              scoring=config['grid_s']['score'],
                              verbose=2)

    # Fit the model
    grid_model.fit(xtrain, ytrain)
    print(f'The chosen hyperparameters are: {grid_model.best_params_}')
    best_estimator = grid_model.best_estimator_

    # Predictions
    prediction = best_estimator.predict(xtest)
    res_error = ytest - prediction
    # reporting
    util.report(best_estimator.get_params(), xtrain,
                xtest, ytest, prediction)

    # let's train a new model using the best obtained hyperparameters
    final_model = ElasticNet(alpha=best_estimator.get_params()['model'].alpha,
                             l1_ratio=best_estimator.get_params()['model'].l1_ratio,
                             max_iter=config['model']['lin_reg']['max_iter'][0],
                             )
    poly_feat_f = PolynomialFeatures(degree=1)
    scaler_f = StandardScaler()

    operations_f = [('scaler_2', scaler_f), ('poly_feat_2', poly_feat_f), ('final_model', final_model)]
    pipe_f = Pipeline(operations_f)
    pipe_f.fit(x, y)
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    model_path = os.path.join(parent_directory, 'Linear_Regression/final_model/final_model.pkl')
    joblib.dump(pipe_f, model_path)
    util.final_visualization(pipe_f, df, x)