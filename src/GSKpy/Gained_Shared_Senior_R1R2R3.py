import random
import numpy as np

# Hazem used 0.2 and 0.8 instead of 0.1 and 0.9 to enable population size of 5
def Gained_Shared_Senior_R1R2R3(indBest=None,p=0.1):

    pop_size=len(indBest)
# Gained_Shared_Senior_R1R2R3.m:4
    #R1=indBest(arange(1,round(dot(pop_size,0.1))))
# Gained_Shared_Senior_R1R2R3.m:6
    #R1rand=ceil(dot(length(R1),rand(pop_size,1)))
# Gained_Shared_Senior_R1R2R3.m:7
    #R1=R1(R1rand)
    R1=[indBest[random.randint(0, round(pop_size*p)-1)] for x in range(pop_size)]
# Gained_Shared_Senior_R1R2R3.m:8
    #R2=indBest(arange(round(dot(pop_size,0.1)) + 1,round(dot(pop_size,0.9))))
# Gained_Shared_Senior_R1R2R3.m:10
    #R2rand=ceil(dot(length(R2),rand(pop_size,1)))
# Gained_Shared_Senior_R1R2R3.m:11
    #R2=R2(R2rand)
    R2=[indBest[random.randint(round(pop_size*p), round(pop_size*(1-p))-1)] for x in range(pop_size)]

# Gained_Shared_Senior_R1R2R3.m:12
    #R3=indBest(arange(round(dot(pop_size,0.9)) + 1,end()))
# Gained_Shared_Senior_R1R2R3.m:14
    #R3rand=ceil(dot(length(R3),rand(pop_size,1)))
# Gained_Shared_Senior_R1R2R3.m:15
    #R3=R3(R3rand)
    R3=[indBest[random.randint(round(pop_size*(1-p)), pop_size-1)] for x in range(pop_size)]
# Gained_Shared_Senior_R1R2R3.m:16
    return np.array(R1),np.array(R2),np.array(R3)
