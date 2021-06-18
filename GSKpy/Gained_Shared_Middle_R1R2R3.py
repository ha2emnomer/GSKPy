import random
import numpy as np

def Gained_Shared_Middle_R1R2R3(pop_size):


# Gained_Shared_Junior_R1R2R3.m:4
    #R0=range(0,pop_size)
# Gained_Shared_Junior_R1R2R3.m:5
    R1=[None for i in range(pop_size)]
# Gained_Shared_Junior_R1R2R3.m:6
    R2=[None for i in range(pop_size)]
# Gained_Shared_Junior_R1R2R3.m:7
    R3=[None for i in range(pop_size)]
    R4=[None for i in range(pop_size)]
    R5=[None for i in range(pop_size)]
    R6=[None for i in range(pop_size)]
# Gained_Shared_Junior_R1R2R3.m:8
    range_idx = pop_size
    for  i in range(pop_size):
        #R1[i] = ind

        R1[i]= random.choice([x for x in range(range_idx) if x != i])
        R2[i]= random.choice([x for x in range(range_idx) if x != R1[i] and x != i])
        R3[i]= random.choice([x for x in range(range_idx) if x != R2[i] and x != R1[i] and x != i])
        R4[i]= random.choice([x for x in range(range_idx) if x != R2[i] and x != R1[i] and x != R3[i] and x != i])
        R5[i]= random.choice([x for x in range(range_idx) if x != R2[i] and x != R1[i] and x != R3[i] and x != R4[i] and x != i])
        R6[i]= random.choice([x for x in range(range_idx) if x != R2[i] and x != R1[i] and x != R3[i] and x != R4[i] and x != R5[i] and x != i])
        #R4[i] = ind

         #floor(dot(rand(1,pop_size),pop_size)) + 1
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
    return np.array(R1),np.array(R2),np.array(R3),np.array(R4),np.array(R5),np.array(R6)
