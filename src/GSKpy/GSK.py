import numpy as np
import copy
import sys
from GSKpy.Gained_Shared_Junior_R1R2R3 import Gained_Shared_Junior_R1R2R3
from GSKpy.Gained_Shared_Senior_R1R2R3 import Gained_Shared_Senior_R1R2R3
from GSKpy.Gained_Shared_Middle_R1R2R3 import Gained_Shared_Middle_R1R2R3
from GSKpy.boundConstraint import boundConstraint


class GSK():
    def __init__(self,evaluation_func,func_args=None,k=10,kf=0.5,kr=0.9,p=0.1):
        """
        Args:
            evaluation_func: the function to be evaluated.
            func_args: a list containing args to be passed to evaluation_func
            k:factor for experience equation that determines the number of dimensions for junior phase
            kf: knowledge factor
            kr: knowledge rate
            p= perecentage of population to be selected for best and worst class in senior and junior phases
        """
        self.K = k
        self.Kf = kf
        self.Kr = kr
        self.p = p
        self.evaluation_func = evaluation_func
        self.func_args = func_args
    def reset():
        #you can reset the parameters but not the evaluation_func
        #reset makes you able to start from the last point you were at.
        pass
    def run(self):
        pass
    def getstatistics(self):
        pass
