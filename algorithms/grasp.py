from constructives import cgrasp
from localsearch import lsbestimp

def execute(inst, iters, alpha):
    best = None
    for i in range(iters):
        print("Iter "+str(i+1)+": ", end="")
        sol = cgrasp.construct(inst, alpha)
        print("C -> "+str(round(sol['of'], 2)), end=", ")
        lsbestimp.improve(sol)
        print("LS -> "+str(round(sol['of'], 2)))
        if best is None or best['of'] < sol['of']:
            best = sol
    return best
