# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import math

class Viz():
    f = None
    X  = None
    Y = None
    Z = None
    data = None

    def __init__(self,func,lb,ub,dim,func_num):
        self.f = func
        self.dim = dim
        self.func_num = func_num
        #self.data  = data #data should be numpy array of shape(2,)
        self.lb = lb
        self.ub = ub
        #self.pop_hist = pop_hist
    def set(self,dim,func_num,data,fitness,best,middle,worst):
        self.dim = dim
        self.func_num = func_num
        self.data = data
        self.best_hist = best
        self.middle_hist = middle
        self.worst_hist = worst
        self.fitness = fitness
        x = np.linspace(self.lb, self.ub,  30)
        y = np.linspace(self.lb, self.ub, 30)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = self.evaluateFun(self.X,self.Y)
    def evaluateFun(self,X,Y):
        #print('shape',X.shape)
        Z =  np.zeros((X.shape[0],X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                hold = np.array([X[i,j],Y[i,j]]).reshape(1,2)
                #print(self.dim,self.func_num)
                Z[i,j] = self.f(hold,[2,self.func_num])
        return Z

    def update_best(self,num,data,line):

        x = self.data[num][0]
        y = self.data[num][1]
        #print(x,y)
        #print (num,x,y,self.f([x,y]))
        line.set_data(x,y)
        line.set_3d_properties(self.f(np.array(self.data[num]).reshape(1,self.dim),[self.dim,self.func_num])[0], 'z')
        return line,



    def build_plot(self,fig_size=(10,5),save=None):
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(projection='3d')
        ax.contour3D(self.X, self.Y, self.Z, 50, cmap=cm.cool,alpha=0.3)
        ax.view_init(60, 35)
        l, = plt.plot([], [], markerfacecolor='g', markeredgecolor='k', marker='o', label='best value', markersize=30, alpha=0.3)
        lines = [plt.plot([], [], markerfacecolor='b', markeredgecolor='b', marker='o', markersize=5, alpha=0.5)[0] for _ in range(self.best_hist[0].shape[0])]
        lines2 = [plt.plot([], [], markerfacecolor='m', markeredgecolor='m', marker='|', markersize=9, alpha=0.5)[0] for _ in range(self.middle_hist[0].shape[0])]
        lines3 = [plt.plot([], [], markerfacecolor='y', markeredgecolor='y', marker='x',markersize=5, alpha=0.5)[0] for _ in range(self.worst_hist[0].shape[0])]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend([l ,lines[0],lines2[0],lines3[0]], ['best_value','better pepole','middle pepole','worst pepole'])
        '''
        def update_pop(num):
            for j,line in enumerate(lines):
                #if j < self.pop_hist[num].shape[0]:
                x = self.pop_hist[num][j,0]
                y = self.pop_hist[num][j,1]
                #print(num,j,x,y)

                line.set_data(x,y)
                #print(self.f(np.array([x,y]).reshape(1,2),self.dim,self.func_num))

                line.set_3d_properties(self.f(np.array(self.pop_hist[num][j]).reshape(1,self.dim),[self.dim,self.func_num])[0], 'z')

            return lines
        '''
        def update(num,data,line):
            x = self.data[num][0]
            y = self.data[num][1]
            #print(x,y)
            #print (num,x,y,self.f([x,y]))
            line.set_data(x,y)
            line.set_3d_properties(self.f(np.array(self.data[num]).reshape(1,self.dim),[self.dim,self.func_num])[0], 'z')


            for j,l in enumerate(lines):
                #if j < self.pop_hist[num].shape[0]:
                x = self.best_hist[num][j,0]
                y = self.best_hist[num][j,1]
                #print(num,j,x,y)

                l.set_data(x,y)
                #print(self.f(np.array([x,y]).reshape(1,2),self.dim,self.func_num))

                l.set_3d_properties(self.fitness[num][0][j], 'z')
            for j,l in enumerate(lines2):
                #if j < self.pop_hist[num].shape[0]:
                x = self.middle_hist[num][j,0]
                y = self.middle_hist[num][j,1]
                #print(num,j,x,y)

                l.set_data(x,y)
                #print(self.f(np.array([x,y]).reshape(1,2),self.dim,self.func_num))

                l.set_3d_properties(self.fitness[num][1][j], 'z')
            for j,l in enumerate(lines3):
                #if j < self.pop_hist[num].shape[0]:
                x = self.worst_hist[num][j,0]
                y = self.worst_hist[num][j,1]
                #print(num,j,x,y)

                l.set_data(x,y)
                #print(self.f(np.array([x,y]).reshape(1,2),self.dim,self.func_num))

                l.set_3d_properties(self.fitness[num][2][j], 'z')


            return (line,lines,lines2,lines3)
        def init():
            #init lines
            for line in lines:
                line.set_data([], [])
            for line in lines2:
                line.set_data([], [])
            for line in lines3:
                line.set_data([], [])


            return (lines,lines2,lines3) #return everything that must be updated

        #best_indv_ani = animation.FuncAnimation(fig, self.update_best, self.data.shape[0], fargs=(self.data,l),
        #                                   interval=100, blit=False)
        ani3 = animation.FuncAnimation(fig, update, len(self.best_hist),init_func=init ,fargs=(self.data,l),
                                                                           interval=100, blit=False)
        if save != None:
            ani3.save(save+'.mp4')
        #lines_ani = animation.FuncAnimation(fig, update_pop, frames=len(self.pop_hist), init_func=init,
        #                                   interval=100, blit=False)

        plt.show()
        return plt
    def plot_losses(self,losses):
        fig = plt.figure()
        ax = plt.axes(xlim=(0, 100), ylim=(-100, 0))
        N = 2
        color = ['g', 'b']
        lines = [plt.plot([], [],color[c])[0] for c in range(N)]
        def init():
            #init lines
            for line in lines:
                line.set_data([], [])

            return lines

        def animate(i):

            #animate lines
            for j,line in enumerate(lines):
                line.set_data([i,i],[i,losses[j][i]])



            return lines

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(losses[0]), interval=100, blit=True)

        plt.show()
