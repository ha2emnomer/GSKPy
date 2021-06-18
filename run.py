from GSKpy.BasicGSK import  BasicGSK
import numpy as np
from GSKpy.cec_functions import cec17_test_func, cec20_test_func
from CSVDataFrame import CSVDataFrame
from GSKpy.viz import Viz
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import interactive

obj_func = cec17_test_func
runs_no =51
optimum = [i+100.0 for i in range(0,3000,100)]
#print (optimum)
#optimum_20 = [100,1100,700,1900,1700,1600,2100,2200,2400,2500]
#vis = GSKVis(obj_func,-100,100,10,1)
funcs_num=[i for i in range(5,30) if i != 1]
max_nfes=[10000*10,10000*30,10000*50,10000*100]

solver = BasicGSK(k=10,kf=0.5,kr=0.9,p=0.1)

for d, dim in enumerate([10,30,50,100]):
    resframe = CSVDataFrame()
    resframe.setheader(['Function','Best', 'Median', 'Mean', 'Worst', 'SD'])
    frame = []
    vis = Viz(cec17_test_func,-100,100,dim,1)
    for func in funcs_num:
        runs = []
        print(func+1,dim,optimum[func])
        for run_id in range(runs_no):
            g,best , best_fit, errors = solver.asynrun(obj_func, dim, 100, [-100]*dim, [100]*dim,optimum=optimum[func], max_nfes=max_nfes[d],func_args=[dim,func+1])
            runs.append(best_fit)
            print()
            print('run {}- best fit: {}'.format(run_id+1,best_fit))
            #visualize the run
            best_hist,fitness_vals, best, middle, worst,junior_dim = solver.getstatistics()

            best_hist = np.array(best_hist)
            best_hist = np.vstack((best_hist))
            best_hist = best_hist.reshape((best_hist.shape[0],dim))
            vis.set(dim,func+1,best_hist,fitness_vals,best,middle,worst)
            vis.build_plot()
            x = np.linspace(0,g,100)
            anims = []

            y1 = np.array(errors)
            f1, ax = plt.subplots(1)
            l1, = ax.plot([], [], 'o-', label='f'+str((func+1)), markevery=[-1])
            ax.legend(loc='upper right')
            ax.set_xlim(0,g)
            ax.set_ylim(0,max(errors))



            def animate(i):
                l1.set_data(x[:i], y1[:i])
                return l1,

            ani = animation.FuncAnimation(f1, animate, frames=100, interval=10)
            anims.append(ani)
            #ani.save("movie.mp4")

            plt.show()

            y1 = np.array(junior_dim)
            y2 = dim-y1

            f2, ax1 = plt.subplots()
            l1, = ax1.plot([], [], 'o-', label='junior dimensions rate', markevery=[-1])
            l2, = ax1.plot([], [], 'o-', label='senior dimensions rate', markevery=[-1])
            ax1.legend(loc='upper right')
            ax1.set_xlim(0,g)
            ax1.set_ylim(0,max(y1))


            def animatetwo(i):
                l1.set_data(x[:i], y1[:i])
                l2.set_data(x[:i], y2[:i])
                return (l1,l2)

            ani = animation.FuncAnimation(f2, animatetwo, frames=100, interval=10)
            anims.append(ani)
            #ani.save("movie2.mp4")

            plt.show()



        runs = np.array(runs)
        print('Best = {}, Worst = {} mean= {},SD = {}, Median {}'.format(np.min(runs),np.max(runs),np.mean(runs), np.std(runs),np.median(runs)))
        frame.append([func+1,np.min(runs),np.median(runs),np.mean(runs),np.max(runs),np.std(runs)])
        resframe.PassDataFrame(frame)
        resframe.save('results/results-cec-2017-{}{}.csv'.format(dim,kf))
