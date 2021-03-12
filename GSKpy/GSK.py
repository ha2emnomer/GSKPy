import numpy as np
import copy
import sys
from GSKpy.Gained_Shared_Junior_R1R2R3 import Gained_Shared_Junior_R1R2R3
from GSKpy.Gained_Shared_Senior_R1R2R3 import Gained_Shared_Senior_R1R2R3
from GSKpy.boundConstraint import boundConstraint


class GSK():
    def __init__(self,k=10,kf=0.5,kr=0.9,p=0.1):
        self.K = k
        self.Kf = kf
        self.Kr = kr
        self.p = p
    def run(self):
        pass
    def getstatistics(self):
        pass
class BasicGSK(GSK):
    def __init__(self,k=10,kf=0.5,kr=0.9,p=0.1):
        GSK.__init__(self,k,kf,kr,p)
        self.best_hist=[]
        self.pop_hist=[]
    def reset(self,k=10,kf=0.5,kr=0.9,p=0.1):
        GSK.__init__(self,k,kf,kr,p)
    def getstatistics(self):
        return self.best_hist,self.pop_hist

    def run(self,evaluation_func, problem_size, pop_size, low, high,optimum=100.0,val_2_reach=10 ** (- 8),func_args=None,max_nfes=1000,verbose= True):

        pop = np.concatenate([np.random.uniform(low[i], high[i], size=(pop_size,1)) for i in range(problem_size)], axis=1)
        nfes = 0
        g=0

        fitness = evaluation_func(pop,func_args)

        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        for i in range(pop_size):
            nfes = nfes + 1
        K = np.full((pop_size, 1), self.K, dtype=int)
        Kf = np.array([self.Kf]*pop_size).reshape(pop_size,1)
        Kr = np.array([self.Kr]*pop_size).reshape(pop_size,1)
        self.pop_hist = [pop]
        self.best_hist = [best_solution]
        while nfes < max_nfes:

            g = g + 1
            D_Junior = np.ceil(problem_size*((1 - nfes/max_nfes)**K))
            D_Senior = problem_size-D_Junior
            best_ind = np.argsort(fitness)
            best_idx = np.argmin(fitness)
            RJ1,RJ2,RJ3 = Gained_Shared_Junior_R1R2R3(best_ind)
            RS1,RS2,RS3 = Gained_Shared_Senior_R1R2R3(best_ind,self.p)
            Gained_Shared_Junior = np.zeros((pop_size,problem_size))
            Gained_Shared_Senior = np.zeros((pop_size,problem_size))

            for i in best_ind:
                if fitness[RJ3[i]] > fitness[i]:
                    Gained_Shared_Junior[i] = pop[i] + Kf[i]  * (pop[RJ1[i]] - pop[RJ2[i]] + pop[i] - pop[RJ3[i]])
                else:
                    Gained_Shared_Junior[i] = pop[i] + Kf[i]  * (pop[RJ1[i]] - pop[RJ2[i]] +  pop[RJ3[i]]- pop[i])
                if fitness[RS2[i]] > fitness[i]:
                    Gained_Shared_Senior[i] = pop[i] + Kf[i] * (pop[RS1[i]] - pop[RS3[i]] +  pop[i]- pop[RS2[i]])
                else:
                    Gained_Shared_Senior[i] = pop[i] + Kf[i] * (pop[RS1[i]] - pop[RS3[i]] +  pop[RS2[i]]- pop[i])
            boundConstraint(Gained_Shared_Junior,pop,low[0],high[0])
            boundConstraint(Gained_Shared_Senior,pop,low[0],high[0])
            mutant = np.zeros((1,problem_size))
            for i in range(pop_size):

                junior_dim = np.random.rand(problem_size) <= (D_Junior[i] / problem_size)
                senior_dim = np.invert(junior_dim)
                mutant[0,junior_dim] = Gained_Shared_Junior[i,junior_dim]
                mutant[0,senior_dim] = Gained_Shared_Senior[i,senior_dim]
                cross_points = np.random.rand(problem_size) <= Kr[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, problem_size)] = True
                trial = np.where(cross_points, mutant[0], pop[i])
                trial = np.array(trial).reshape((1,problem_size))
                f = evaluation_func(trial,func_args)
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    if f < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial
                nfes+=1
            new_pop = copy.deepcopy(pop)
            new_best = copy.deepcopy(best_solution)
            self.pop_hist.append(new_pop)
            self.best_hist.append(new_best)
            best_fitness = fitness[best_idx]-optimum
            if best_fitness < val_2_reach:
                best_fitness=0
                break
            if verbose:
                #pass
                sys.stdout.write('generation {} - pop_size {} - fittness {} - nfes {} \r'.format(g,pop_size,fitness[best_idx],nfes))
                sys.stdout.flush()
        return best_solution , best_fitness
class BasicGSKLPSR(GSK):
    def __init__(self,k=10,kf=0.5,kr=0.9,p=0.1):
        GSK.__init__(self,k,kf,kr,p)
    def reset(self,k=10,kf=0.5,kr=0.9,p=0.1):
        GSK.__init__(self,k,kf,kr,p)
    def getstatistics(self):
        return self.best_hist,self.pop_hist

    def run(self,evaluation_func, problem_size, pop_size, low, high,optimum=100.0,val_2_reach=10 ** (- 8),func_args=None,max_nfes=1000,verbose= True):

        pop = np.concatenate([np.random.uniform(low[i], high[i], size=(pop_size,1)) for i in range(problem_size)], axis=1)
        nfes = 0
        g=0
        max_pop_size = pop_size
        min_pop_size = 12

        fitness = evaluation_func(pop,func_args)

        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        for i in range(pop_size):
            nfes = nfes + 1
        K = np.full((pop_size, 1), self.K, dtype=int)
        Kf = np.array([self.Kf]*pop_size).reshape(pop_size,1)
        Kr = np.array([self.Kr]*pop_size).reshape(pop_size,1)
        self.pop_hist = [pop]
        self.best_hist = [best_solution]
        while nfes < max_nfes:

            g = g + 1
            D_Junior = np.ceil(problem_size*((1 - nfes/max_nfes)**K))
            D_Senior = problem_size-D_Junior
            best_ind = np.argsort(fitness)
            best_idx = np.argmin(fitness)
            RJ1,RJ2,RJ3 = Gained_Shared_Junior_R1R2R3(best_ind)
            RS1,RS2,RS3 = Gained_Shared_Senior_R1R2R3(best_ind,self.p)
            Gained_Shared_Junior = np.zeros((pop_size,problem_size))
            Gained_Shared_Senior = np.zeros((pop_size,problem_size))

            for i in best_ind:
                if fitness[RJ3[i]] > fitness[i]:
                    Gained_Shared_Junior[i] = pop[i] + Kf[i]  * (pop[RJ1[i]] - pop[RJ2[i]] + pop[i] - pop[RJ3[i]])
                else:
                    Gained_Shared_Junior[i] = pop[i] + Kf[i]  * (pop[RJ1[i]] - pop[RJ2[i]] +  pop[RJ3[i]]- pop[i])
                if fitness[RS2[i]] > fitness[i]:
                    Gained_Shared_Senior[i] = pop[i] + Kf[i] * (pop[RS1[i]] - pop[RS3[i]] +  pop[i]- pop[RS2[i]])
                else:
                    Gained_Shared_Senior[i] = pop[i] + Kf[i] * (pop[RS1[i]] - pop[RS3[i]] +  pop[RS2[i]]- pop[i])
            boundConstraint(Gained_Shared_Junior,pop,low[0],high[0])
            boundConstraint(Gained_Shared_Senior,pop,low[0],high[0])
            mutant = np.zeros((1,problem_size))
            for i in range(pop_size):

                junior_dim = np.random.rand(problem_size) <= (D_Junior[i] / problem_size)
                senior_dim = np.invert(junior_dim)
                #apply junior and senior mutation
                mutant[0,junior_dim] = Gained_Shared_Junior[i,junior_dim]
                mutant[0,senior_dim] = Gained_Shared_Senior[i,senior_dim]
                #cross over
                cross_points = np.random.rand(problem_size) <= Kr[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, problem_size)] = True
                trial = np.where(cross_points, mutant[0], pop[i])
                trial = np.array(trial).reshape((1,problem_size))
                f = evaluation_func(trial,func_args)
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    if f < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial
                nfes+=1
            plan_pop_size = round((min_pop_size - max_pop_size) * ((nfes / max_nfes) ** (1)) + max_pop_size)

            if pop_size > plan_pop_size:
                reduction_ind_num = pop_size - plan_pop_size
                if pop_size - reduction_ind_num < min_pop_size:
                    reduction_ind_num = pop_size - min_pop_size
                pop_size = pop_size - reduction_ind_num
                for r in range(reduction_ind_num):
                    indBest = np.argsort(fitness)
                    worst_ind = indBest[-1]
                    #popold = np.delete(popold, worst_ind, 0).
                    if best_idx != worst_ind:
                        pop = np.delete(pop, worst_ind, 0)
                        fitness = np.delete(fitness, worst_ind, 0)
                        K = np.delete(K, worst_ind, 0)
                        Kf = np.delete(Kf,worst_ind,0)
                        Kr = np.delete(Kr,worst_ind,0)
                best_idx = np.argsort(fitness)[0]
            new_pop = copy.deepcopy(pop)
            new_best = copy.deepcopy(best_solution)
            self.pop_hist.append(new_pop)
            self.best_hist.append(new_best)
            best_fitness = fitness[best_idx]-optimum
            if best_fitness < val_2_reach:
                best_fitness=0
                break
            if verbose:
                #pass
                sys.stdout.write('generation {} - pop_size {} - fittness {} - nfes {} \r'.format(g,pop_size,fitness[best_idx],nfes))
                sys.stdout.flush()
        return best_solution , fitness[best_idx]-optimum
