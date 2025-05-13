from structure.instance import readInstance
import time
import copy
from structure import solution
import random
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
    p = instance['p']

    to_remove = list(sol_start['sol'] - sol_guiding['sol'])
    to_add = list(sol_guiding['sol'] - sol_start['sol'])

    current_sol = copy.deepcopy(sol_start)
    best_sol = copy.deepcopy(current_sol)

    sum_dist_to_current_sol = calculate_sum_dist_to_sol(instance, current_sol['sol'], n)

    path_length = len(to_add)

    while len(to_add) > 0:
        add_candidates_values = [(u, sum_dist_to_current_sol[u]) for u in to_add]

        remove_candidates_values = [(v, sum_dist_to_current_sol[v]) for v in to_remove]

        min_add_val = min(add_candidates_values, key=lambda item: item[1])[1] if add_candidates_values else 0
        max_add_val = max(add_candidates_values, key=lambda item: item[1])[1] if add_candidates_values else 0
        add_threshold = max_add_val - beta * (max_add_val - min_add_val)
        add_rcl = [u for (u, val) in add_candidates_values if val >= add_threshold]

        min_remove_val = min(remove_candidates_values, key=lambda item: item[1])[1] if remove_candidates_values else 0
        max_remove_val = max(remove_candidates_values, key=lambda item: item[1])[1] if remove_candidates_values else 0
        remove_threshold = min_remove_val + beta * (max_remove_val - min_remove_val)
        remove_rcl = [v for (v, val) in remove_candidates_values if val <= remove_threshold]

        if not add_rcl or not remove_rcl:
             break 

        chosen_add_elem = random.choice(add_rcl)
        chosen_remove_elem = random.choice(remove_rcl)

        delta_of = (sum_dist_to_current_sol[chosen_add_elem] - instance['d'][chosen_add_elem][chosen_remove_elem]) - sum_dist_to_current_sol[chosen_remove_elem]
        delta_of = round(delta_of, 2)

        current_sol['sol'].remove(chosen_remove_elem)
        current_sol['sol'].add(chosen_add_elem)

        current_sol['of'] += delta_of
        current_sol['of'] = round(current_sol['of'], 2)

        for x in range(n):
             sum_dist_to_current_sol[x] = sum_dist_to_current_sol[x] - instance['d'][x][chosen_remove_elem] + instance['d'][x][chosen_add_elem]
             sum_dist_to_current_sol[x] = round(sum_dist_to_current_sol[x], 2)

        to_remove.remove(chosen_remove_elem)
        to_add.remove(chosen_add_elem)

        if current_sol['of'] > best_sol['of']:
            best_sol = copy.deepcopy(current_sol)

    return best_sol

def main(total_time_limit=120):
    instance_file = "instances/MDG-a_2_n500_m50.txt"
    alpha_value = 0.1       
    elite_set_size = 10     
    diversity_weight = 1.0

    beta_value = 0.2

    time_limit_grasp = total_time_limit / 2.0
    time_limit_pr = total_time_limit / 2.0

    try:
        instance = readInstance(instance_file)
        print(f"Instancia leída: n={instance['n']}, p={instance['p']}")
    except FileNotFoundError:
        print(f"Error: El archivo de instancia '{instance_file}' no fue encontrado.")
        return
    except Exception as e:
        print(f"Error al leer la instancia: {e}")
        return

    print("\n--- Ejecutando GRASP para generar el conjunto de élite diverso ---")
    start_time_grasp = time.time()

    best_grasp_sol, all_unique_solutions, grasp_exec_time, elite_set = run_grasp(
        instance, time_limit_grasp, alpha_value, elite_set_size, diversity_weight
    )

    end_time_grasp = time.time()
    print(f"Fase GRASP finalizada en {grasp_exec_time:.2f} segundos (de {time_limit_grasp:.2f} asignados).")
    print(f"Mejor solución encontrada por GRASP (OF): {best_grasp_sol['of']:.2f}")
    print(f"Número total de soluciones únicas encontradas por GRASP: {len(all_unique_solutions)}")
    print(f"Tamaño del conjunto de élite generado: {len(elite_set)}")

    print("Valores OF del conjunto de élite:", [round(sol['of'], 2) for sol in elite_set])
    min_pairwise_diff = float('inf')
    if len(elite_set) > 1:
        for i in range(len(elite_set)):
            for j in range(i + 1, len(elite_set)):
                diff = solution_difference(elite_set[i], elite_set[j])
                min_pairwise_diff = min(min_pairwise_diff, diff)
        print(f"Mínima diferencia simétrica por pares en el conjunto de élite: {min_pairwise_diff}")
    elif len(elite_set) == 1:
        print("Conjunto de élite con 1 solución, no aplica diferencia por pares.")
    else:
        print("Conjunto de élite vacío.")

    print("\n--- Ejecutando Path Relinking en el conjunto de élite ---")
    start_time_pr = time.time()
    elapsed_time_pr = 0

    best_solution_overall = copy.deepcopy(best_grasp_sol)

    num_elite = len(elite_set)

    pair_indices = [(i, j) for i in range(num_elite) for j in range(i + 1, num_elite)]

    pair_idx = 0
    while pair_idx < len(pair_indices) and elapsed_time_pr < time_limit_pr:
        i, j = pair_indices[pair_idx]

        sol1 = elite_set[i]
        sol2 = elite_set[j]

        print(f"  Relinking entre solución {i+1} (OF: {sol1['of']:.2f}) y solución {j+1} (OF: {sol2['of']:.2f})")

        current_time = time.time()
        elapsed_time_pr = current_time - start_time_pr
        if elapsed_time_pr >= time_limit_pr:
            print("Tiempo límite de Path Relinking alcanzado. Deteniendo.")
            break

        pr_fwd = path_relinking(sol1, sol2, beta_value)
        if pr_fwd:
            if pr_fwd['of'] > best_solution_overall['of']:
                best_solution_overall = copy.deepcopy(pr_fwd)
                print(f"    -> Nueva mejor solución encontrada (OF: {best_solution_overall['of']:.2f}) durante relinking hacia adelante.")

        current_time = time.time()
        elapsed_time_pr = current_time - start_time_pr
        if elapsed_time_pr >= time_limit_pr:
            print("Tiempo límite de Path Relinking alcanzado. Deteniendo.")
            break

        pr_bwd = path_relinking(sol2, sol1, beta_value)
        if pr_bwd:
            if pr_bwd['of'] > best_solution_overall['of']:
                best_solution_overall = copy.deepcopy(pr_bwd)
                print(f"    -> Nueva mejor solución encontrada (OF: {best_solution_overall['of']:.2f}) durante relinking hacia atrás.")

        current_time = time.time()
        elapsed_time_pr = current_time - start_time_pr
        if elapsed_time_pr >= time_limit_pr:
            print("Tiempo límite de Path Relinking alcanzado. Deteniendo.")
            break

        pair_idx += 1


    end_time_pr = time.time()
    print(f"Fase Path Relinking finalizada. Tiempo transcurrido: {elapsed_time_pr:.2f} segundos (de {time_limit_pr:.2f} asignados).")

    print("\n--- Resultado Global Final ---")
    print(f"Mejor solución global encontrada (OF): {best_solution_overall['of']:.2f}")
    print("Detalles de la mejor solución global:")
    solution.printSolution(best_solution_overall)


if __name__ == "__main__":
    random.seed(1)
    main(total_time_limit=10)