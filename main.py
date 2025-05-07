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
        print(f"\nArchivo: {nombre_archivo}")
        print("-"*50,"\n")
        path = "instances/" + nombre_archivo
        inst = instance.readInstance(path)
        for cont in range(0,101,5):
            alpha = cont/100
            sol, _, time_final = grasp.execute(inst, 1, alpha)
            #print("\nBEST SOLUTION:")
            #solution.printSolution(sol)
            f.write(f"{nombre_archivo[:-4]};{alpha};{sol["of"]:.4f};{time_final:.4f}\n")
            print(f"alpha={alpha} -> Puntuación: {sol["of"]:.2f} Tiempo: {time_final:.2f}")
    f.close()


if __name__ == '__main__':
    random.seed(1)
    executeInstance()

