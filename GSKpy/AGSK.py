import numpy as np
import copy
import sys
from GSKpy.Gained_Shared_Junior_R1R2R3 import Gained_Shared_Junior_R1R2R3
from GSKpy.Gained_Shared_Senior_R1R2R3 import Gained_Shared_Senior_R1R2R3
from GSKpy.Gained_Shared_Middle_R1R2R3 import Gained_Shared_Middle_R1R2R3
from GSKpy.boundConstraint import boundConstraint
from GSKpy.GSK import GSK
class AGSK(GSK):
    def __init__(self,k=10,kf=0.5,kr=0.9,p=0.1):
        GSK.__init__(self,k,kf,kr,p)
        self.best_hist=[]
        self.pop_hist=[]
        self.mem_size = 10
        self.KF_Memory = np.full((self.mem_size, 1), 0.5)
        self.KR_Memory = np.full((self.mem_size, 1), 0.9)

    def reset(self,k=10,kf=0.5,kr=0.9,p=0.1):
        GSK.__init__(self,k,kf,kr,p)
    def getstatistics(self):
        return self.best_hist,self.pop_hist

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
        self.KF_Memory = np.full((self.mem_size, 1), 0.5)
        self.KR_Memory = np.full((self.mem_size, 1), 0.9)

        Kf = np.array([self.Kf]*pop_size).reshape(pop_size,1)
        Kr = np.array([self.Kr]*pop_size).reshape(pop_size,1)
        K = np.full((pop_size, 1), 10.0, dtype='float32')
        '''
        K_ind = np.random.rand(pop_size,1)

        K[K_ind  < 0.5] = np.random.rand(np.sum(K_ind<0.5))
        K[K_ind  >= 0.5] = np.ceil(20 *np.random.rand(np.sum(K_ind>=0.5)))3
        '''
        old_kf = Kf

        g = 0
        update_index = 0
        while nfes < max_nfes:

            g = g + 1
            indx = np.random.choice(range(self.mem_size),(pop_size,1))
            mean_kf = [self.KF_Memory[ind,0] for ind in indx]
            mean_kr = [self.KR_Memory[ind,0] for ind in indx]


            Kr = np.random.normal(mean_kr, 0.1 , (pop_size,1))
            pos = np.where(Kf <= 0)[0]
            #print(pos)
            while len(pos) !=0:
                Kf[pos] = np.random.normal(mean_kf[pos], 0.1 , (len(pos),1))
                pos = np.where(Kf <= 0)[0]
            Kf = np.minimum(Kf, 1)
            Kr[Kr  == -1 ] = 0
            Kr= np.minimum(Kr,1)
            Kr= np.maximum(Kr,0)


            D_Gained_Shared_Junior = np.ceil(problem_size*((1 - nfes/max_nfes)**K))
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

            fitness_imp = np.abs(fitness-children_fitness)
            I = fitness > children_fitness

            #sum_fitness_imp = np.sum(fitness_imp[fitness > children_fitness])

            new_Kf = Kf[I==0]
            new_Kr = Kr[I==0]
            fitness_imp_val = fitness_imp[I==0]

            if len(new_Kr) > 0:
                sum_fitness_imp = np.sum(fitness_imp_val)
                fitness_imp_val = fitness_imp_val / sum_fitness_imp

            conc = np.concatenate((fitness.reshape(-1,1),children_fitness.reshape(-1,1)), axis=1)
            Child_is_better_index = conc.argmin(axis=1)

            fitness = conc[range(conc.shape[0]),Child_is_better_index]#.reshape(-1, 1)
            popold = pop
            popold[Child_is_better_index == 1,:] = ui[Child_is_better_index == 1,:]


            update_kf =  np.sum(fitness_imp_val *new_Kf**2) / np.sum(fitness_imp_val*new_Kf)
            if np.max(new_Kr) == 0 or self.KR_Memory[update_index%self.mem_size]==-1:
                update_kr = -1
            else:
                update_kr =  np.sum(fitness_imp_val *new_Kr**2) / np.sum(fitness_imp_val*new_Kr)
            self.KF_Memory[update_index%self.mem_size] = update_kf
            self.KR_Memory[update_index%self.mem_size] = update_kr
            #print(self.KR_Memory)
            update_index+=1





            new_pop = copy.deepcopy(pop)
            new_best = copy.deepcopy(bsf_solution)
            self.pop_hist.append(new_pop)
            self.best_hist.append(new_best)
            if bsf_error_val < val_2_reach:
                bsf_error_val=0
                break



        return bsf_solution,bsf_error_val
