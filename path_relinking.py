from structure import instance, solution
from algorithms import grasp
from copy import deepcopy
import pandas as pd

def relink(start, guide):
    current = deepcopy(start)
    best = deepcopy(start)
    to_add = guide["sol"] - current["sol"]
    to_remove = current["sol"] - guide["sol"]

    while to_add:
        vecinos = []
        for a in to_add:
            for r in to_remove:
                nuevo = deepcopy(current)
                nuevo["sol"].remove(r)
                nuevo["sol"].add(a)
                nuevo["of"] = solution.evaluate(nuevo)
                vecinos.append(nuevo)
        if not vecinos:
            break
        current = max(vecinos, key=lambda s: s["of"])
        if current["of"] > best["of"]:
            best = deepcopy(current)
        to_add = guide["sol"] - current["sol"]
        to_remove = current["sol"] - guide["sol"]

    return best

def pathRelinking(path, alpha=0.1, tiempo=60, elite_size=5):
    """
    Ejecuta GRASP + Path Relinking sobre una instancia.

    Parámetros:
        path: ruta del archivo de instancia.
        alpha: valor alfa para GRASP.
        tiempo: tiempo total (se divide entre GRASP y PR).
        elite_size: número de soluciones élite para PR.

    Devuelve:
        mejor_pr: mejor solución encontrada por PR.
        df_pr: DataFrame con todas las soluciones PR.
    """
    # Leer la instancia
    inst = instance.readInstance(path)

    # Ejecutar GRASP con la mitad del tiempo asignado
    _, soluciones, _ = grasp.execute(inst, tiempo / 2, alpha)

    # Ordenar soluciones y construir conjunto élite
    soluciones.sort(key=lambda s: -s['of'])  # orden descendente
    elite = []
    seen = set()
    for sol in soluciones:
        frozen = frozenset(sol['sol'])
        if frozen not in seen:
            elite.append(sol)
            seen.add(frozen)
        if len(elite) >= elite_size:
            break

    # Aplicar PR entre pares de soluciones élite
    soluciones_pr = []
    mejor_pr = None
    for i in range(len(elite)):
        for j in range(i + 1, len(elite)):
            pr1 = relink(elite[i], elite[j])
            pr2 = relink(elite[j], elite[i])
            soluciones_pr.extend([pr1, pr2])
            for sol in [pr1, pr2]:
                if mejor_pr is None or sol["of"] > mejor_pr["of"]:
                    mejor_pr = sol

    df_pr = pd.DataFrame(soluciones_pr)
    return mejor_pr, df_pr
