import numpy as np
import copy
import sys
from GSKpy.Gained_Shared_Junior_R1R2R3 import Gained_Shared_Junior_R1R2R3
from GSKpy.Gained_Shared_Senior_R1R2R3 import Gained_Shared_Senior_R1R2R3
from GSKpy.Gained_Shared_Middle_R1R2R3 import Gained_Shared_Middle_R1R2R3
from GSKpy.boundConstraint import boundConstraint


class GSK():
    def __init__(self,k=10,kf=0.5,kr=0.9,p=0.9):
        self.K = k
        self.Kf = kf
        self.Kr = kr
        self.p = p
    def run(self):
        pass
    def getstatistics(self):
        pass
