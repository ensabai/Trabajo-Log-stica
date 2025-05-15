import copy
import random

def calculate_sum_dist_to_sol(instance, sol_set, n):
    sum_dist = [0.0] * n
    for i in range(n):
        for s in sol_set:
            sum_dist[i] += instance['d'][i][s]
        sum_dist[i] = round(sum_dist[i], 2)
    return sum_dist

def path_relinking(sol_start, sol_guide, beta):
    instance = sol_start['instance']
    n = instance['n']

    to_remove = list(sol_start['sol'] - sol_guide['sol'])
    to_add = list(sol_guide['sol'] - sol_start['sol'])

    current = copy.deepcopy(sol_start)
    best = copy.deepcopy(current)

    sum_dist = calculate_sum_dist_to_sol(instance, current['sol'], n)

    while to_add:
        add_vals = [(u, sum_dist[u]) for u in to_add]
        rem_vals = [(v, sum_dist[v]) for v in to_remove]

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

        delta = (sum_dist[u] - instance['d'][u][v]) - sum_dist[v]
        delta = round(delta, 2)

        current['sol'].remove(v)
        current['sol'].add(u)
        current['of'] += delta
        current['of'] = round(current['of'], 2)

        for x in range(n):
            sum_dist[x] = sum_dist[x] - instance['d'][x][v] + instance['d'][x][u]
            sum_dist[x] = round(sum_dist[x], 2)

        to_remove.remove(v)
        to_add.remove(u)

        if current['of'] > best['of']:
            best = copy.deepcopy(current)

    return best