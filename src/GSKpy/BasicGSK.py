import numpy as np
import copy
import sys
from GSKpy.Gained_Shared_Junior_R1R2R3 import Gained_Shared_Junior_R1R2R3
from GSKpy.Gained_Shared_Senior_R1R2R3 import Gained_Shared_Senior_R1R2R3
from GSKpy.Gained_Shared_Middle_R1R2R3 import Gained_Shared_Middle_R1R2R3
from GSKpy.boundConstraint import boundConstraint
from GSKpy.GSK import GSK
class BasicGSK(GSK):
    def __init__(self,evaluation_func,problem_size,pop_size,low,high,max_nfes=1000,func_args=None,bound_constraint="default",LPSR=False,k=10,kf=0.5,kr=0.9,p=0.1):
        """
        Args:
            evaluation_func: the function to be evaluated.
            problem_size: the number of dimensions of solution
            pop_size: the size of population
            low: list containing lower bounds for each dimension
            high: list containing higher bounds for each dimension
            max_nfes: maximum number of function evaluations
            func_args: a list containing args to be passed to evaluation_func
            bound_constraint: function to limit the values of solutions "default" uses the default boundConstraint function
            LPSR: if True Linear population size reduction is used
            k:factor for experience equation that determines the number of dimensions for junior phase
            kf: knowledge factor
            kr: knowledge rate
            p= perecentage of population to be selected for best and worst class in senior and junior phases
        """
        GSK.__init__(self,evaluation_func,func_args,k,kf,kr,p)
        self.best_hist=[]
        self.pop_hist=[]
        self.best_pop_hist = []
        self.middle_pop_hist = []
        self.worst_pop_hist = []
        self.errors = []
        self.junior_dim = []
        self.fitness_vals = []
        self.pop = []
        self.problem_size = problem_size
        self.pop_size = pop_size
        if len(low) != self.problem_size or len(high) != self.problem_size:
            raise('Low and high bounadries should be as the same size of problem_size')
        self.low = low
        self.high = high
        self.bound_constraint = bound_constraint
        self.max_nfes = max_nfes
        self.nfes  = 0
        self.fitness = None
        self.popold = None
        self.LPSR =LPSR
    def reset(self,k=10,kf=0.5,kr=0.9,p=0.1):
        #rest only rest parameters statistic are appended
        GSK.__init__(self,k,kf,kr,p)
    def getstatistics(self):
        return self.best_hist,self.fitness_vals, self.best_pop_hist, self.middle_pop_hist, self.worst_pop_hist, self.junior_dim

    def run(self, optimum=0.0,val_2_reach=10 ** (- 8),verbose= True, track=False):
        """run method that optimize the function
        Args:
        optimum: optimum value of the function used for CEC testbench mark
        val_2_reach: value to reach the algorithm stops when this value is reached
        verbose: each generation output is displayed
        track: if True stores store the values of pop, fitness_vals, best, worst, the change of junior dimensions
        """
        max_pop_size = self.pop_size
        #min pop size is set for population reduction.
        min_pop_size = 12


        if len(self.pop) == 0:
            #create a new population if no population exists
            self.popold = np.concatenate([np.random.uniform(self.low[i], self.high[i], size=(self.pop_size,1)) for i in range(self.problem_size)], axis=1)
            self.pop = self.popold
            self.fitness = self.evaluation_func(self.pop,self.func_args)

            self.nfes = 0

            self.bsf_fit_var = 1e+300
            self.bsf_solution = self.popold[0]
            for i in range(self.pop_size):
                self.nfes = self.nfes + 1

                if self.nfes > self.max_nfes:
                    break
                if self.fitness[i] < self.bsf_fit_var:
                    self.bsf_fit_var = self.fitness[i]
            self.bsf_error_val=self.bsf_fit_var - optimum
            self.pop_hist = [self.pop]
            self.best_hist = [self.bsf_solution]
        K = np.full((self.pop_size, 1), self.K, dtype=int)
        Kf = np.array([self.Kf]*self.pop_size).reshape(self.pop_size,1)
        Kr = np.array([self.Kr]*self.pop_size).reshape(self.pop_size,1)
        g = 0
        while self.nfes < self.max_nfes:
            g = g + 1
            D_Gained_Shared_Junior = np.ceil(self.problem_size*((1 - self.nfes/self.max_nfes)**K))

            self.pop = self.popold
            pop = self.pop #avoid meesy code with self.pop

            indBest = np.argsort(self.fitness)	# if fitness not np array use x = np.array([3, 1, 2]) to convert it

            Rg1,Rg2,Rg3 = Gained_Shared_Junior_R1R2R3(indBest)

            R1,R2,R3 = Gained_Shared_Senior_R1R2R3(indBest,self.p)

            R01 = range(self.pop_size)

            Gained_Shared_Junior = np.zeros((self.pop_size,self.problem_size))


            ind1 = self.fitness[R01] > self.fitness[Rg3] # fitness must be np.array
            if np.sum(ind1) > 0:
                Gained_Shared_Junior[ind1,:] = pop[ind1,:] + Kf[ind1,:] * np.ones((np.sum(ind1),self.problem_size)) * (pop[Rg1[ind1],:] - pop[Rg2[ind1],:] + pop[Rg3[ind1],:] - pop[ind1,:])

            ind1 = np.invert(ind1)
            if np.sum(ind1) > 0:
                Gained_Shared_Junior[ind1,:] = pop[ind1,:] + Kf[ind1,:] * np.ones((np.sum(ind1),self.problem_size)) * (pop[Rg1[ind1],:] - pop[Rg2[ind1],:] + pop[ind1,:] - pop[Rg3[ind1],:])
            R0 = range(self.pop_size)

            Gained_Shared_Senior = np.zeros((self.pop_size,self.problem_size))

            ind = self.fitness[R0] > self.fitness[R2]
            if np.sum(ind) > 0:
                Gained_Shared_Senior[ind,:] = pop[ind,:] + Kf[ind,:] * np.ones((np.sum(ind),self.problem_size)) * (pop[R1[ind],:] - pop[ind,:] + pop[R2[ind],:] - pop[R3[ind],:])

            ind = np.invert(ind)
            if np.sum(ind) > 0:
                Gained_Shared_Senior[ind,:] = pop[ind,:] + Kf[ind,:] * np.ones((np.sum(ind),self.problem_size)) * (pop[R1[ind],:] - pop[R2[ind],:] + pop[ind,:] - pop[R3[ind],:])
            if self.bound_constraint == "default":
                boundConstraint(Gained_Shared_Junior,pop,self.low,self.high)
                boundConstraint(Gained_Shared_Senior,pop,self.low,self.high)
            else:
                #called twice on the junior mutants and senior mutants before getting mixed
                self.bound_constraint(Gained_Shared_Junior)
                self.bound_constraint(Gained_Shared_Senior)


            D_Gained_Shared_Junior_mask = np.random.rand(self.pop_size,self.problem_size) <= (D_Gained_Shared_Junior[:] / self.problem_size)
            D_Gained_Shared_Senior_mask = np.invert(D_Gained_Shared_Junior_mask)
            D_Gained_Shared_Junior_rand_mask = np.random.rand(self.pop_size,self.problem_size) <= Kr
            D_Gained_Shared_Junior_mask = np.logical_and(D_Gained_Shared_Junior_mask,D_Gained_Shared_Junior_rand_mask)
            D_Gained_Shared_Senior_rand_mask = np.random.rand(self.pop_size,self.problem_size) <= Kr
            D_Gained_Shared_Senior_mask = np.logical_and(D_Gained_Shared_Senior_mask,D_Gained_Shared_Senior_rand_mask)

            ui = copy.deepcopy(pop)

            ui[D_Gained_Shared_Junior_mask] = Gained_Shared_Junior[D_Gained_Shared_Junior_mask]
            ui[D_Gained_Shared_Senior_mask] = Gained_Shared_Senior[D_Gained_Shared_Senior_mask]

            children_fitness = self.evaluation_func(ui,self.func_args)

            for i in range(self.pop_size):
                self.nfes = self.nfes + 1
                if self.nfes > self.max_nfes:
                    break
                if children_fitness[i] < self.bsf_fit_var:
                    self.bsf_fit_var = children_fitness[i]
                    self.bsf_solution = ui[i,:]

            self.bsf_error_val=self.bsf_fit_var - optimum
            if verbose:
                print('\rgeneration {} - pop_size {} - fittness {} - nfes {}'.format(g,self.pop_size,self.bsf_error_val,self.nfes),end='')
                sys.stdout.flush()


            conc = np.concatenate((self.fitness.reshape(-1,1),children_fitness.reshape(-1,1)), axis=1)
            Child_is_better_index = conc.argmin(axis=1)

            self.fitness = conc[range(conc.shape[0]),Child_is_better_index]#.reshape(-1, 1)
            self.popold = pop
            self.popold[Child_is_better_index == 1,:] = ui[Child_is_better_index == 1,:]
            #Linear population size reduction
            if self.LPSR:
                plan_pop_size = round((min_pop_size - max_pop_size) * ((self.nfes / self.max_nfes) ** (1)) + max_pop_size)
                best_idx = np.argsort(self.fitness)[0]
                if self.pop_size > plan_pop_size:
                    reduction_ind_num = self.pop_size - plan_pop_size
                    if self.pop_size - reduction_ind_num < min_pop_size:
                        reduction_ind_num = self.pop_size - min_pop_size
                    self.pop_size = self.pop_size - reduction_ind_num
                    for r in range(reduction_ind_num):
                        indBest = np.argsort(self.fitness)
                        worst_ind = indBest[-1]
                        self.popold = np.delete(self.popold, worst_ind, 0)
                        if best_idx != worst_ind:
                            self.pop = np.delete(self.pop, worst_ind, 0)
                            self.fitness = np.delete(self.fitness, worst_ind, 0)
                            K = np.delete(K, worst_ind, 0)
                            Kf = np.delete(Kf,worst_ind,0)
                            Kr = np.delete(Kr,worst_ind,0)

            if track:
                new_pop = copy.deepcopy(self.pop)
                new_best = copy.deepcopy(self.bsf_solution)
                self.pop_hist.append(new_pop)
                self.best_hist.append(new_best)
                self.best_pop_hist.append(self.pop[R1])
                self.middle_pop_hist.append(self.pop[R2])
                self.worst_pop_hist.append(self.pop[R3])
                self.fitness_vals.append([self.fitness[R1],self.fitness[R2],self.fitness[R3]])
                self.errors.append(self.bsf_error_val)
                self.junior_dim.append(D_Gained_Shared_Junior[0])
            if self.bsf_error_val < val_2_reach:
                self.bsf_error_val=0
                break


        return g,self.bsf_solution,self.bsf_error_val,self.errors
