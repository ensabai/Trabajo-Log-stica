from constructives import cgrasp
from localsearch import lsbestimp
import time

def execute(inst, tiempo, alpha):
    best = None
    conjunto_soluciones = []
    time_start = time.time()
    time_final = 0
    i = 0
    while time_final < tiempo:
        print("Iter "+str(i+1)+": ", end="")
        sol = cgrasp.construct(inst, alpha)
        print("C -> "+str(round(sol['of'], 2)), end=", ")
        lsbestimp.improve(sol)
        print("LS -> "+str(round(sol['of'], 2)))
        conjunto_soluciones.append(sol)
        if best is None or best['of'] < sol['of']:
            best = sol
        time_final = time.time()-time_start
        i += 1
    return best, conjunto_soluciones
