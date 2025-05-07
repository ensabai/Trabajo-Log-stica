from structure import instance, solution
from algorithms import grasp
import random
import time
import os

def executeInstance():

    #path = "instances/MDG-a_2_n500_m50.txt"
    registro = "resultados.csv"
    f = open(registro, "a")
    f.write("archivo;alpha;valor;tiempo\n")
    for nombre_archivo in os.listdir("instances"):
        path = "instances/" + nombre_archivo
        inst = instance.readInstance(path)
        for cont in range(0,101,5):
            alpha = cont/100
            time_start = time.time()
            sol, _ = grasp.execute(inst, 1, alpha)
            time_final = time.time()-time_start
            #print("\nBEST SOLUTION:")
            #solution.printSolution(sol)
            f.write(f"{nombre_archivo[:-4]};{alpha};{sol["of"]:.2f};{time_final:.2f}\n")
    f.close()


if __name__ == '__main__':
    random.seed(1)
    executeInstance()

