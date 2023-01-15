import numpy as np
import scipy.optimize as optimize

#OPTIMISATION USING SCIPY
def Obj_func(x):
    m,y,d=x
    dmax = 400
    tOF = np.pi-2*np.arctan2(2*d,m)
    if tOF> np.pi:
        tOF = np.pi
    f= -1*((tOF-y)/(tOF+y))#*((dmax-d)/dmax)* np.sin(tOF*y)
    return f

initial_guess=[10,1,1]
bnds = ((0, 40), (0, np.pi),(400, None)) #all four variables are positive and greater than zero
#Always t1 and t2 should always be lesser than bo and ho
#res=optimize.minimize(Obj_func, method='SLSQP',initial_guess, bounds=bnds,constraints=cons)
res=optimize.minimize(Obj_func,initial_guess, bounds=bnds)
print ("Result",res)