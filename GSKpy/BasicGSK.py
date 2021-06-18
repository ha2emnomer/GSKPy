import numpy as np
import copy
import sys
from GSKpy.Gained_Shared_Junior_R1R2R3 import Gained_Shared_Junior_R1R2R3
from GSKpy.Gained_Shared_Senior_R1R2R3 import Gained_Shared_Senior_R1R2R3
from GSKpy.Gained_Shared_Middle_R1R2R3 import Gained_Shared_Middle_R1R2R3
from GSKpy.boundConstraint import boundConstraint
from GSKpy.GSK import GSK
class BasicGSK(GSK):
    def __init__(self,k=10,kf=0.5,kr=0.9,p=0.1):
        GSK.__init__(self,k,kf,kr,p)
        self.best_hist=[]
        self.pop_hist=[]
        self.best_pop_hist = []
        self.middle_pop_hist = []
        self.worst_pop_hist = []
        self.errors = []
        self.junior_dim = []
        self.fitness_vals = []
    def reset(self,k=10,kf=0.5,kr=0.9,p=0.1):
        GSK.__init__(self,k,kf,kr,p)
    def getstatistics(self):
        return self.best_hist,self.fitness_vals, self.best_pop_hist, self.middle_pop_hist, self.worst_pop_hist, self.junior_dim

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
                junior_dim_rand = np.random.rand(problem_size) <= Kr[i]
                junior_dim = np.logical_and(junior_dim,junior_dim_rand)
                senior_dim_rand = np.random.rand(problem_size) <= Kr[i]
                senior_dim = np.logical_and(senior_dim,senior_dim_rand)


                mutant[0,junior_dim] = Gained_Shared_Junior[i,junior_dim]
                mutant[0,senior_dim] = Gained_Shared_Senior[i,senior_dim]
                trial = mutant
                ''''
                cross_points = np.random.rand(problem_size) <= Kr[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, problem_size)] = True
                trial = np.where(cross_points, mutant[0], pop[i])
                trial = np.array(trial).reshape((1,problem_size))
                '''
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
                sys.stdout.write('generation {} - pop_size {} - var {} - fittness {} - nfes {} \r'.format(g,pop_size,np.var(pop),fitness[best_idx],nfes))
                sys.stdout.flush()
        return best_solution , best_fitness
    def asynrun(self,evaluation_func, problem_size, pop_size, low, high,optimum=100.0,val_2_reach=10 ** (- 8),func_args=None,max_nfes=1000,verbose= True):
        max_nfes= max_nfes
        #max_nfes =0
        max_pop_size = pop_size
        min_pop_size = 12
        popold = np.concatenate([np.random.uniform(low[i], high[i], size=(pop_size,1)) for i in range(problem_size)], axis=1)
        pop = popold



        fitness = evaluation_func(pop,func_args)

        nfes = 0

        bsf_fit_var = 1e+300
        bsf_solution = popold[0]
        val_2_reach=10 ** (- 8)
        for i in range(pop_size):
            nfes = nfes + 1

            if nfes > max_nfes:
                break
            if fitness[i] < bsf_fit_var:
                bsf_fit_var = fitness[i]
        self.pop_hist = [pop]
        self.best_hist = [bsf_solution]

        K = np.full((pop_size, 1), self.K, dtype=int)
        Kf = np.array([self.Kf]*pop_size).reshape(pop_size,1)
        Kr = np.array([self.Kr]*pop_size).reshape(pop_size,1)
        g = 0
        while nfes < max_nfes:

            g = g + 1
            D_Gained_Shared_Junior = np.ceil(problem_size*((1 - nfes/max_nfes)**K))
            self.junior_dim.append(D_Gained_Shared_Junior[0])
            pop = popold

            indBest = np.argsort(fitness)	# if fitness not np array use x = np.array([3, 1, 2]) to convert it

            Rg1,Rg2,Rg3 = Gained_Shared_Junior_R1R2R3(indBest)

            R1,R2,R3 = Gained_Shared_Senior_R1R2R3(indBest,self.p)

            R01 = range(pop_size)

            Gained_Shared_Junior = np.zeros((pop_size,problem_size))


            ind1 = fitness[R01] > fitness[Rg3] # fitness must be np.array
            if np.sum(ind1) > 0:
                Gained_Shared_Junior[ind1,:] = pop[ind1,:] + Kf[ind1,:] * np.ones((np.sum(ind1),problem_size)) * (pop[Rg1[ind1],:] - pop[Rg2[ind1],:] + pop[Rg3[ind1],:] - pop[ind1,:])

            ind1 = np.invert(ind1)
            if np.sum(ind1) > 0:
                Gained_Shared_Junior[ind1,:] = pop[ind1,:] + Kf[ind1,:] * np.ones((np.sum(ind1),problem_size)) * (pop[Rg1[ind1],:] - pop[Rg2[ind1],:] + pop[ind1,:] - pop[Rg3[ind1],:])
            R0 = range(pop_size)

            Gained_Shared_Senior = np.zeros((pop_size,problem_size))

            ind = fitness[R0] > fitness[R2]
            if np.sum(ind) > 0:
                Gained_Shared_Senior[ind,:] = pop[ind,:] + Kf[ind,:] * np.ones((np.sum(ind),problem_size)) * (pop[R1[ind],:] - pop[ind,:] + pop[R2[ind],:] - pop[R3[ind],:])

            ind = np.invert(ind)
            if np.sum(ind) > 0:
                Gained_Shared_Senior[ind,:] = pop[ind,:] + Kf[ind,:] * np.ones((np.sum(ind),problem_size)) * (pop[R1[ind],:] - pop[R2[ind],:] + pop[ind,:] - pop[R3[ind],:])

            boundConstraint(Gained_Shared_Junior,pop,low,high)
            boundConstraint(Gained_Shared_Senior,pop,low,high)

            D_Gained_Shared_Junior_mask = np.random.rand(pop_size,problem_size) <= (D_Gained_Shared_Junior[:] / problem_size)
            D_Gained_Shared_Senior_mask = np.invert(D_Gained_Shared_Junior_mask)
            D_Gained_Shared_Junior_rand_mask = np.random.rand(pop_size,problem_size) <= Kr
            D_Gained_Shared_Junior_mask = np.logical_and(D_Gained_Shared_Junior_mask,D_Gained_Shared_Junior_rand_mask)
            D_Gained_Shared_Senior_rand_mask = np.random.rand(pop_size,problem_size) <= Kr
            D_Gained_Shared_Senior_mask = np.logical_and(D_Gained_Shared_Senior_mask,D_Gained_Shared_Senior_rand_mask)

            ui = copy.deepcopy(pop)

            ui[D_Gained_Shared_Junior_mask] = Gained_Shared_Junior[D_Gained_Shared_Junior_mask]
            ui[D_Gained_Shared_Senior_mask] = Gained_Shared_Senior[D_Gained_Shared_Senior_mask]

            children_fitness = evaluation_func(ui,func_args)

            for i in range(pop_size):
                nfes = nfes + 1
                if nfes > max_nfes:
                    break
                if children_fitness[i] < bsf_fit_var:
                    bsf_fit_var = children_fitness[i]
                    bsf_solution = ui[i,:]

            bsf_error_val=bsf_fit_var - optimum
            if verbose:
                sys.stdout.write('generation {} - pop_size {} - var {} -fittness {}- nfes {}\r'.format(g,pop_size,np.var(pop),bsf_error_val,nfes))
                sys.stdout.flush()


            conc = np.concatenate((fitness.reshape(-1,1),children_fitness.reshape(-1,1)), axis=1)
            Child_is_better_index = conc.argmin(axis=1)

            fitness = conc[range(conc.shape[0]),Child_is_better_index]#.reshape(-1, 1)
            popold = pop
            popold[Child_is_better_index == 1,:] = ui[Child_is_better_index == 1,:]


            new_pop = copy.deepcopy(pop)
            new_best = copy.deepcopy(bsf_solution)
            self.pop_hist.append(new_pop)
            self.best_hist.append(new_best)
            self.best_pop_hist.append(pop[R1])
            self.middle_pop_hist.append(pop[R2])
            self.worst_pop_hist.append(pop[R3])
            self.fitness_vals.append([fitness[R1],fitness[R2],fitness[R3]])
            self.errors.append(bsf_error_val)
            if bsf_error_val < val_2_reach:
                bsf_error_val=0
                break



        return g,bsf_solution,bsf_error_val,self.errors
