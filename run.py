from GSKpy.GSK import  BasicGSK,BasicGSKLPSR
import numpy as np
from GSKpy.cec_functions import cec17_test_func, cec20_test_func
from CSVDataFrame import CSVDataFrame
from GSKpy.viz import Viz

obj_func = cec17_test_func
runs_no =51
optimum = [i+100.0 for i in range(0,3000,100)]
#print (optimum)
#optimum_20 = [100,1100,700,1900,1700,1600,2100,2200,2400,2500]
#vis = GSKVis(obj_func,-100,100,10,1)
funcs_num=[i for i in range(0,30) if i != 1]
max_nfes=[10000*10,10000*30,10000*50,10000*100]

solver = BasicGSKLPSR(k=10,kf=0.5,kr=0.9,p=0.1)

for d, dim in enumerate([10,30,50,100]):
    resframe = CSVDataFrame()
    resframe.setheader(['Function','Best', 'Median', 'Mean', 'Worst', 'SD'])
    frame = []
    vis = Viz(cec17_test_func,-100,100,dim,1)
    for func in funcs_num:
        runs = []
        print(func+1,dim,optimum[func])
        for run_id in range(runs_no):
            best , best_fit = solver.run(obj_func, dim, 100, [-100]*dim, [100]*dim,optimum=optimum[func], max_nfes=max_nfes[d],func_args=[dim,func+1])
            runs.append(best_fit)
            print()
            print('run {}- best fit: {}'.format(run_id+1,best_fit))
            #visualize the run
            best_hist,pop_hist = solver.getstatistics()

            best_hist = np.array(best_hist)
            best_hist = np.vstack((best_hist))
            best_hist = best_hist.reshape((best_hist.shape[0],dim))
            vis.set(dim,func+1,best_hist,pop_hist)
            #vis.build_plot()
        runs = np.array(runs)
        print('Best = {}, Worst = {} mean= {},SD = {}, Median {}'.format(np.min(runs),np.max(runs),np.mean(runs), np.std(runs),np.median(runs)))
        frame.append([func+1,np.min(runs),np.median(runs),np.mean(runs),np.max(runs),np.std(runs)])
        resframe.PassDataFrame(frame)
        resframe.save('results/results-cec-2017-{}.csv'.format(dim))
