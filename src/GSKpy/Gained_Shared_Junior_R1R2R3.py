import random
import numpy as np

def Gained_Shared_Junior_R1R2R3(indBest=None):

    pop_size=len(indBest)
# Gained_Shared_Junior_R1R2R3.m:4
    #R0=range(0,pop_size)
# Gained_Shared_Junior_R1R2R3.m:5
    R1=[None for i in range(pop_size)]
# Gained_Shared_Junior_R1R2R3.m:6
    R2=[None for i in range(pop_size)]
# Gained_Shared_Junior_R1R2R3.m:7
    R3=[None for i in range(pop_size)]
# Gained_Shared_Junior_R1R2R3.m:8
    for ind, i in enumerate(indBest):
        #ind=find(indBest == i)
# Gained_Shared_Junior_R1R2R3.m:11
        if ind == 0:
            R1[i]=indBest[1]
# Gained_Shared_Junior_R1R2R3.m:13
            R2[i]=indBest[2]
# Gained_Shared_Junior_R1R2R3.m:14
        else:
            if ind == (pop_size-1):
                R1[i]=indBest[pop_size - 3]
# Gained_Shared_Junior_R1R2R3.m:16
                R2[i]=indBest[pop_size - 2]
# Gained_Shared_Junior_R1R2R3.m:17
            else:
                R1[i]=indBest[ind - 1]
# Gained_Shared_Junior_R1R2R3.m:19
                R2[i]=indBest[ind + 1]
# Gained_Shared_Junior_R1R2R3.m:20

        R3[i]= random.choice([x for x in range(pop_size) if x != R1[i] and x != R2[i] and x != i]) #floor(dot(rand(1,pop_size),pop_size)) + 1
# Gained_Shared_Junior_R1R2R3.m:24
    #for i in arange(1,99999999).reshape(-1):
        #pos=(logical_or(logical_or((R3 == R2),(R3 == R1)),(R3 == R0)))
# Gained_Shared_Junior_R1R2R3.m:27
        #if sum(pos) == 0:
            #break
        #else:
            #R3[pos]=floor(dot(rand(1,sum(pos)),pop_size)) + 1
# Gained_Shared_Junior_R1R2R3.m:31
        #if i > 1000:
            #error('Can not genrate R3 in 1000 iterations')
    return np.array(R1),np.array(R2),np.array(R3)
