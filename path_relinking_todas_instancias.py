import os
import time
import copy
import random
import pandas as pd
from structure.instance import readInstance
from structure import solution
from localsearch import lsbestimp
from constructives import cgrasp
from path_relinking import path_relinking

def solution_difference(sol1, sol2):
    return len(sol1['sol'] ^ sol2['sol'])


def run_grasp(inst, tiempo, alpha, num_elite, diversity_weight):
    best = None
    all_solutions_dict = {}
    time_start = time.time()
    while time.time() - time_start < tiempo:
        sol = cgrasp.construct(inst, alpha)
        lsbestimp.improve(sol)

        sol_key = frozenset(sol['sol'])
        if sol_key not in all_solutions_dict or sol['of'] > all_solutions_dict[sol_key]['of']:
            all_solutions_dict[sol_key] = copy.deepcopy(sol)

        if best is None or sol['of'] > best['of']:
            best = copy.deepcopy(sol)

    all_unique_solutions = list(all_solutions_dict.values())
    elite_set = []
    if not all_unique_solutions:
        return best, [], 0.0, elite_set

    overall_best_unique = max(all_unique_solutions, key=lambda s: s['of'])
    elite_set.append(overall_best_unique)
    remaining = [s for s in all_unique_solutions if s != overall_best_unique]

    min_div_to_elite = {frozenset(s['sol']): solution_difference(s, overall_best_unique) for s in remaining}

    while len(elite_set) < num_elite and remaining:
        best_score = -float('inf')
        selected = None

        for sol in remaining:
            key = frozenset(sol['sol'])
            div = min_div_to_elite.get(key, 0)
            score = sol['of'] + diversity_weight * div
            if score > best_score:
                best_score = score
                selected = sol

        if selected:
            elite_set.append(selected)
            remaining = [s for s in remaining if s != selected]

            for s in remaining:
                key = frozenset(s['sol'])
                new_div = solution_difference(s, selected)
                min_div_to_elite[key] = min(min_div_to_elite.get(key, float('inf')), new_div)
            min_div_to_elite.pop(frozenset(selected['sol']), None)
        else:
            break

    return best, all_unique_solutions, time.time() - time_start, elite_set



def ejecutar_grasp_pr(total_time_limit=60, carpeta="instances"):
    alpha = 0.1
    elite_size = 10
    beta = 0.2
    diversity_weight = 1.0

    resultados = []
    for archivo in sorted(os.listdir(carpeta)):
        if not archivo.endswith(".txt"):
            continue
        path = os.path.join(carpeta, archivo)
        print(f"\nProcesando: {archivo}")

        try:
            inst = readInstance(path)
        except Exception as e:
            print(f"Error leyendo {archivo}: {e}")
            continue

        start_global = time.time()

        best_grasp, _, t_grasp, elite = run_grasp(inst, total_time_limit / 2, alpha, elite_size, diversity_weight)
        best_global = copy.deepcopy(best_grasp)

        start_pr = time.time()
        t_limit = total_time_limit / 2
        pairs = [(i, j) for i in range(len(elite)) for j in range(i+1, len(elite))]

        for i, j in pairs:
            if time.time() - start_pr > t_limit:
                break

            pr1 = path_relinking(elite[i], elite[j], beta)
            if pr1['of'] > best_global['of']:
                best_global = copy.deepcopy(pr1)

            if time.time() - start_pr > t_limit:
                break

            pr2 = path_relinking(elite[j], elite[i], beta)
            if pr2['of'] > best_global['of']:
                best_global = copy.deepcopy(pr2)
        
        tiempo_final = time.time() - start_global

        print(f"\nMejor solución para {archivo} (OF: {best_global['of']}):")
        print(f"   Elementos: {sorted(best_global['sol'])}")

        resultados.append({
            "archivo": archivo[:-4],
            "alpha": alpha,
            "valor": best_global['of'],
            "tiempo": tiempo_final
        })

    df = pd.DataFrame(resultados)
    df.to_csv(f"resultados/resultados_pr_{float(total_time_limit)}s.csv", index=False, sep=";")
    print("\nResultados guardados en 'resultados_grasp_pr.csv'")


if __name__ == "__main__":
    random.seed(1)
    ejecutar_grasp_pr(total_time_limit=0.5)