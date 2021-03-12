import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from GSKpy.GSK import  BasicGSK,BasicGSKLPSR
# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
#diabetes_X = diabetes_X[:, np0]
print(diabetes_X.shape)
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_pred = regr.predict(diabetes_X_test)
print(mean_squared_error(diabetes_y_test, diabetes_y_pred))
dim = len(regr.coef_)
def evaluateFun(coef,func_args=None):
    fitness_val = []
    for co in coef:
        regr.coef_ = co
        diabetes_y_pred = regr.predict(diabetes_X_train)
        #print(mean_squared_error(diabetes_y_train, diabetes_y_pred))
        fitness_val.append(mean_squared_error(diabetes_y_train, diabetes_y_pred))
    return np.array(fitness_val)
solver = BasicGSK(k=10,kf=0.5,kr=0.9,p=0.1)
for i in range(5):
    best , best_fit = solver.run(evaluateFun, dim, 100, [-1000]*dim, [1000]*dim,optimum=0, max_nfes=100000)
    regr.coef_ = best
    diabetes_y_pred = regr.predict(diabetes_X_test)
    print()
    print(mean_squared_error(diabetes_y_test, diabetes_y_pred))
