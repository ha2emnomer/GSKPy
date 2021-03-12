# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    def set(self,dim,func_num,data,pop_hist):
        self.dim = dim
        self.func_num = func_num
        self.data = data
        self.pop_hist = pop_hist
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


    def build_plot(self,fig_size=(10,5)):
        fig = plt.figure(figsize=fig_size)
        ax = plt.axes(projection='3d')
        ax.contour3D(self.X, self.Y, self.Z, 200, cmap='binary')
        l, = plt.plot([], [], markerfacecolor='g', markeredgecolor='g', marker='o', markersize=10, alpha=0.3)
        lines = [plt.plot([], [], markerfacecolor='b', markeredgecolor='b', marker='o', markersize=5, alpha=0.5)[0] for _ in range(self.pop_hist[0].shape[0])]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
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
        def init():
            #init lines
            for line in lines:
                line.set_data([], [])

            return lines #return everything that must be updated

        line_ani = animation.FuncAnimation(fig, self.update_best, self.data.shape[0], fargs=(self.data,l),
                                           interval=100, blit=False)
        lines_ani = animation.FuncAnimation(fig, update_pop, frames=len(self.pop_hist), init_func=init,
                                           interval=100, blit=False)

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
