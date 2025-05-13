from structure.instance import readInstance
import time
import copy
import pandas as pd
from structure import solution
import random
import os
from localsearch import lsbestimp
from constructives import cgrasp

def solution_difference(sol1, sol2):
    return len(sol1['sol'] ^ sol2['sol'])
   
def run_grasp(inst, tiempo, alpha, num_elite, diversity_weight):
    best = None
    all_solutions_dict = {}
    time_start = time.time()
    time_final = 0
    i = 0
    while time_final < tiempo:
        sol = cgrasp.construct(inst, alpha)
        lsbestimp.improve(sol)

        sol_key = frozenset(sol['sol'])
        if sol_key not in all_solutions_dict or sol['of'] > all_solutions_dict[sol_key]['of']:
            all_solutions_dict[sol_key] = copy.deepcopy(sol) 

        
        if best is None or sol['of'] > best['of']: 
            best = copy.deepcopy(sol) 

        time_final = time.time() - time_start
        i += 1

    all_unique_solutions = list(all_solutions_dict.values())

    elite_set = []
    if not all_unique_solutions:
        return best, [], time_final, elite_set

    overall_best_unique = None
    for sol in all_unique_solutions:
        if overall_best_unique is None or sol['of'] > overall_best_unique['of']:
            overall_best_unique = sol

    elite_set.append(overall_best_unique)

    remaining_candidates = [sol for sol in all_unique_solutions if sol != overall_best_unique]

    min_diversity_to_elite = {}
    if remaining_candidates:
        for sol in remaining_candidates:
            min_diversity_to_elite[frozenset(sol['sol'])] = solution_difference(sol, overall_best_unique)

    while len(elite_set) < num_elite and remaining_candidates:
        best_score = -float('inf')
        selected_sol = None
        selected_sol_key = None

        for sol in remaining_candidates:
            sol_key = frozenset(sol['sol'])
            current_min_diversity = min_diversity_to_elite.get(sol_key, 0)
            score = sol['of'] + diversity_weight * current_min_diversity

            if score > best_score:
                best_score = score
                selected_sol = sol
                selected_sol_key = sol_key

        if selected_sol:
            elite_set.append(selected_sol)

            remaining_candidates = [sol for sol in remaining_candidates if sol != selected_sol]

            for sol_rem in remaining_candidates:
                 sol_rem_key = frozenset(sol_rem['sol'])
                 new_div_to_added = solution_difference(sol_rem, selected_sol)
                 current_min_div = min_diversity_to_elite.get(sol_rem_key, float('inf'))
                 min_diversity_to_elite[sol_rem_key] = min(current_min_div, new_div_to_added)

            if selected_sol_key in min_diversity_to_elite:
                 del min_diversity_to_elite[selected_sol_key]
        else:
            break 

    return best, all_unique_solutions, time_final, elite_set

def calculate_sum_dist_to_sol(instance, sol_set, n):
    sum_dist = [0.0] * n
    for i in range(n):
        for s in sol_set:
            sum_dist[i] += instance['d'][i][s]
        sum_dist[i] = round(sum_dist[i], 2)
    return sum_dist


def path_relinking(sol_start, sol_guiding, beta):
    instance = sol_start['instance']
    n = instance['n']

    to_remove = list(sol_start['sol'] - sol_guiding['sol'])
    to_add = list(sol_guiding['sol'] - sol_start['sol'])

    current_sol = copy.deepcopy(sol_start)
    best_sol = copy.deepcopy(current_sol)

    sum_dist_to_current_sol = calculate_sum_dist_to_sol(instance, current_sol['sol'], n)

    while to_add:
        add_vals = [(u, sum_dist_to_current_sol[u]) for u in to_add]
        rem_vals = [(v, sum_dist_to_current_sol[v]) for v in to_remove]

        if not add_vals or not rem_vals:
            break

        min_a, max_a = min(add_vals, key=lambda x: x[1])[1], max(add_vals, key=lambda x: x[1])[1]
        add_thresh = max_a - beta * (max_a - min_a)
        add_rcl = [u for u, val in add_vals if val >= add_thresh]

        min_r, max_r = min(rem_vals, key=lambda x: x[1])[1], max(rem_vals, key=lambda x: x[1])[1]
        rem_thresh = min_r + beta * (max_r - min_r)
        rem_rcl = [v for v, val in rem_vals if val <= rem_thresh]

        if not add_rcl or not rem_rcl:
            break

        u = random.choice(add_rcl)
        v = random.choice(rem_rcl)

        delta = (sum_dist_to_current_sol[u] - instance['d'][u][v]) - sum_dist_to_current_sol[v]
        delta = round(delta, 2)

        current_sol['sol'].remove(v)
        current_sol['sol'].add(u)
        current_sol['of'] += delta
        current_sol['of'] = round(current_sol['of'], 2)

        for x in range(n):
            sum_dist_to_current_sol[x] = sum_dist_to_current_sol[x] - instance['d'][x][v] + instance['d'][x][u]
            sum_dist_to_current_sol[x] = round(sum_dist_to_current_sol[x], 2)

        to_remove.remove(v)
        to_add.remove(u)

        if current_sol['of'] > best_sol['of']:
            best_sol = copy.deepcopy(current_sol)

    return best_sol


def ejecutar_grasp_pr_sobre_instancias(total_time_limit=60, carpeta_instancias="instances"):
    alpha_value = 0.1
    elite_set_size = 10
    diversity_weight = 1.0
    beta_value = 0.2

    resultados = []

    archivos = sorted([f for f in os.listdir(carpeta_instancias) if f.endswith(".txt")])
    for archivo in archivos:
        path = os.path.join(carpeta_instancias, archivo)
        print(f"\n--- INSTANCIA: {archivo} ---")
        try:
            inst = readInstance(path)
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")
            continue

        # GRASP
        best_grasp, all_sols, t_grasp, elite = run_grasp(
            inst, total_time_limit / 2, alpha_value, elite_set_size, diversity_weight
        )

        best_global = copy.deepcopy(best_grasp)

        # PR
        start_pr = time.time()
        t_pr_limit = total_time_limit / 2
        idxs = [(i, j) for i in range(len(elite)) for j in range(i+1, len(elite))]

        for i, j in idxs:
            if time.time() - start_pr > t_pr_limit:
                break

            sol1 = elite[i]
            sol2 = elite[j]

            pr_fwd = path_relinking(sol1, sol2, beta_value)
            if pr_fwd['of'] > best_global['of']:
                best_global = copy.deepcopy(pr_fwd)

            if time.time() - start_pr > t_pr_limit:
                break

            pr_bwd = path_relinking(sol2, sol1, beta_value)
            if pr_bwd['of'] > best_global['of']:
                best_global = copy.deepcopy(pr_bwd)

        # 🔹 Imprimir detalles de la mejor solución para esta instancia
        print(f"\nMejor solución global para {archivo}:")
        print(f"\n - Valor objetivo: {best_global['of']}")
        print(f"\n - Elementos seleccionados: {sorted(best_global['sol'])}\n")

        resultados.append({
            "instancia": archivo,
            "grasp_mejor": best_grasp["of"],
            "grasp_elite": len(elite),
            "pr_mejor": best_global["of"],
            "mejora_abs": round(best_global["of"] - best_grasp["of"], 2),
            "mejor_solucion_pr": best_global["sol"]
        })

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("resultados_grasp_pr.csv", index=False)
    print("\nTodos los resultados guardados en 'resultados_grasp_pr.csv'")

if __name__ == "__main__":
    random.seed(1)
    ejecutar_grasp_pr_sobre_instancias(total_time_limit=60)