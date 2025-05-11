from structure import instance, solution
from algorithms import grasp
import random

def pathRelinking(path, alpha, tiempo):
    inst = instance.readInstance(path)
    sol, conjunto_soluciones, tiempo_final_grasp = grasp.execute(inst, tiempo / 2, alpha)
    return