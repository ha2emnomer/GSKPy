

This an implementation of Gaining-sharing knowledge algorithm (GSK) in python. [GSK](https://link.springer.com/article/10.1007/s13042-019-01053-x) is a nature inspired algorithm for solving real parameter optimization problems. 
GSK has two main stages the junior and senior phases each has a different mutation, the dimensions (or parameters) are changed by the mutations of both the junior and senior phases
at the same time. GSK is a  reliable and stable optimization algorithm. The repository also includes a visualization module for visualizing GSK runs.
The code have been tested on CEC 2017 benchmark functions. Two version of GSK the BasicGSK and BasicGSKLSPR (with linear propulation reduction).


## Usage

just type 
```
$ python run.py
```



## ❤️&nbsp; How to use GSK as a solver

```js
solver = BasicGSKLPSR(k=10,kf=0.5,kr=0.9,p=0.1)
best , best_fit = solver.run(obj_func, dim, 100, [-100]*dim, [100]*dim)
```
you can also use the get_statstics functions and Viz after the run 

```python
vis = Viz(cec17_test_func,-100,100,dim,1)
best_hist,pop_hist = solver.getstatistics()
best_hist = np.array(best_hist)
best_hist = np.vstack((best_hist))
best_hist = best_hist.reshape((best_hist.shape[0],dim))
vis.set(dim,func+1,best_hist,pop_hist)
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

