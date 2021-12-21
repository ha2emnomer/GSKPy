

This an implementation of Gaining-sharing knowledge algorithm (GSK) in python. [GSK](https://link.springer.com/article/10.1007/s13042-019-01053-x) is a nature inspired algorithm for solving real parameter optimization problems.
GSK has two main stages the junior and senior phases each has a different mutation, the dimensions (or parameters) are changed by the mutations of both the junior and senior phases.
at the same time. GSK is a  reliable and stable optimization algorithm. The repository also includes a visualization module for visualizing GSK runs.
The code have been tested on CEC 2017 benchmark functions.
The module provide a testbed optimization framework that is easy to use with GSK.


## Examples

To run GSK on CEC 2017 or CEC 2020 benchmark functions use run.py example
NOTE: the compiled binaries of functions only work in Linux environment
```
$ python run.py
```



## ❤️&nbsp; How to use GSK as a solver
An objective function of 10 dimensions with -100, 100 as lower and upper bounds
Just define your objective_function to have x (numpy array of shape (pop size, dim)) and func_args any function arguments required.
```python
def ackley(x,func_args=[20,0.2,2*pi]):
    #The objective function should take x
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    a= func_args[0]
    b= func_args[1]
    c=func_args[2]

    n = len(x[0])
    s1 = sum( x**2 ,axis=1)
    s2 = sum( cos( c * x ),axis=1)
    return -a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)
```
```js
from GSKpy.BasicGSK import BasicGSK
solver = BasicGSK(objective_function,10,100,[-100]*10,[-100]*10,max_nfes=100000)
g, best , best_fit, errors = solver.run()
```
you can also use the get_statstics functions and Viz (visualization class) after the run

```python
from GSKpy.viz import Viz
vis = Viz(ackley,-100,100,dim,func_args=[20,0.2,2*pi])
best_hist,fitness_vals, best, middle, worst,junior_dim = solver.getstatistics()

best_hist = np.array(best_hist)
best_hist = np.vstack((best_hist))
best_hist = best_hist.reshape((best_hist.shape[0],dim))
vis.set(dim,best_hist,fitness_vals,best,middle,worst)
vis.build_plot()
```
There is also an [example](https://github.com/ha2emnomer/GSKPy/blob/master/linear_reg.py) on using GSK for linear regression using scikit-learn

## 📫&nbsp; We would love to hear from you
If you have any comments or questions just email  h.nomer@nu.edu.eg
We intend to realse a pip package soon with more examples. More work is done to GSK as a solver for different optimization problems.



## ✅&nbsp; Requirements

**python 2.7 or higher
**matplotlib (for visualization)
**CSVDataFrame (or any other package for saving results)
**numpy 
