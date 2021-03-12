from ctypes import CDLL, POINTER, c_int, c_double
import os
import numpy as np


def cec17_test_func(x,args):
    dll_path=CDLL(os.path.abspath('lib/cec17_test_func.so'))
    f=[]
    #f_pop = []

    nx=args[0]
    mx=1

    func_num=args[1]
    functions = dll_path
    x_pointer_type = POINTER(c_double * nx)
    f_pointer_type = POINTER(c_double * mx)
    nx_type = c_int
    mx_type = c_int
    func_num_type = c_int
    functions.cec17_test_func.argtypes = [x_pointer_type, f_pointer_type,
                                          nx_type, mx_type, func_num_type]
    functions.cec17_test_func.restype = None
    x_ctype = (c_double * nx)()
    for _, indv in enumerate(x):
        for i, value in enumerate(indv):
            if len(x.shape) > 1:
                x_ctype[i] = value
            else:
                x_ctype[i] = value
        f_ctype = (c_double * mx)()
        for i in range(mx):
            f_ctype[i] = 0
        functions.cec17_test_func(x_pointer_type(x_ctype), f_pointer_type(f_ctype),
                                  nx, mx, func_num)
        #for i in range(len(f)):
        f.append(f_ctype[0])
        #f_pop.append(f)
    return np.array(f)
def cec20_test_func(x,nx=10,func_num=1):
    dll_path=CDLL(os.path.abspath('cec20_test_func.so'))
    f=[]
    mx=1
    nx=nx
    functions = dll_path
    x_pointer_type = POINTER(c_double * nx)
    f_pointer_type = POINTER(c_double * mx)
    nx_type = c_int
    mx_type = c_int
    func_num_type = c_int
    functions.cec20_test_func.argtypes = [x_pointer_type, f_pointer_type,
                                          nx_type, mx_type, func_num_type]
    functions.cec20_test_func.restype = None

    x_ctype = (c_double * nx)()
    for _, indv in enumerate(x):
        for i, value in enumerate(indv):
            if len(x.shape) > 1:
                x_ctype[i] = value
            else:
                x_ctype[i] = value
        f_ctype = (c_double * mx)()
        for i in range(mx):
            f_ctype[i] = 0
        functions.cec20_test_func(x_pointer_type(x_ctype), f_pointer_type(f_ctype),
                                  nx, mx, func_num)
        #for i in range(len(f)):
        f.append(f_ctype[0])
        #f_pop.append(f)
    return np.array(f)
