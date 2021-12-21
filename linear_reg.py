import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from GSKpy.BasicGSK import  BasicGSK
# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
#diabetes_X = diabetes_X[:, np0]
print('Train data shape:',diabetes_X.shape)
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_pred = regr.predict(diabetes_X_test)
print('MSE=',mean_squared_error(diabetes_y_test, diabetes_y_pred))
dim = len(regr.coef_)
def evaluateFun(coef,func_args=None):
    fitness_val = []
    for co in coef:
        regr.coef_ = co
        diabetes_y_pred = regr.predict(diabetes_X_train)
        #print(mean_squared_error(diabetes_y_train, diabetes_y_pred))
        fitness_val.append(mean_squared_error(diabetes_y_train, diabetes_y_pred))
    return np.array(fitness_val)

for i in range(30):
    solver = BasicGSK(evaluateFun, dim, 200, [-1000]*dim,[1000]*dim,LPSR=True,kr=0.1,max_nfes=100000)
    g,best , best_fit, loss = solver.run(optimum=0)
    regr.coef_ = best
    diabetes_y_pred = regr.predict(diabetes_X_test)
    print()
    print('MSE=',mean_squared_error(diabetes_y_test, diabetes_y_pred))
