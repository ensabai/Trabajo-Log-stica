from structure import instance, solution
from algorithms import grasp
import random

def executeInstance():
    path = "instances/MDG-a_2_n500_m50.txt"
    inst = instance.readInstance(path)
    sol, _, _ = grasp.execute(inst, 1, -1)
    print("\nBEST SOLUTION:")
    solution.printSolution(sol)

if __name__ == '__main__':
    random.seed(1)
    executeInstance()

