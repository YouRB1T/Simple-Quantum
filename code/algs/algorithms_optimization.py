import math
import random
import time
from collections import deque

import numpy as np

def simulated_annealing(objective_function, initial_solution, generate_neighbor_solution,
                        num_vertices, edges, t=1000, t_min=0.01, alpha=0.95, max_iterations=1000, min_delta=1e-6, window_size=5):
    s = initial_solution(num_vertices)
    best_solution = s.copy()
    best_value = objective_function(edges, s)
    values = [best_value]
    evals = 1
    iters = 0

    improvement_window = deque(maxlen=window_size)
    improvement_window.append(0)

    start_time = time.time()
    best_probability = 0
    while t > t_min and iters < max_iterations:
        s_new = generate_neighbor_solution(s)
        v_new = objective_function(edges, s_new); evals += 1
        delta = v_new - objective_function(edges, s); evals += 1

        if delta > 0 or random.random() < math.exp(delta / t):
            s = s_new.copy()

        current_value = objective_function(edges, s); evals += 1
        if current_value > best_value:
            best_value = current_value
            best_solution = s.copy()

        best_probability = math.exp((best_value - current_value) / t)

        values.append(best_value)
        improvement_window.append(abs(values[-1] - values[-2]))

        if len(improvement_window) == window_size and all(x < min_delta for x in improvement_window):
            break

        t *= alpha
        iters += 1

    total_time = time.time() - start_time

    return {
        "best_solution": best_solution,
        "best_value": best_value,
        "total_time": total_time,
        "iterations": iters,
        "values_curve": values,
        "n_evals": evals,
        "best_measurement_probability": best_probability
    }


def genetic_algorithm(objective_function, initial_solution_fn, mutate, crossover,
                      num_vertices, edges, population_size=50, generations=100, mutation_rate=0.1, min_delta=1e-6, window_size=5):
    start_time = time.time()
    evals = 0
    history = []
    best_probability = 0

    population = [initial_solution_fn(num_vertices) for _ in range(population_size)]
    def fitness(sol):
        nonlocal evals
        evals += 1
        return objective_function(edges, sol)

    improvement_window = deque(maxlen=window_size)
    improvement_window.append(0)

    for gen in range(generations):
        population.sort(key=fitness, reverse=True)
        best = population[0]
        best_val = fitness(best)
        history.append(best_val)

        if len(history) > 1:
            improvement_window.append(abs(history[-1] - history[-2]))

        if len(improvement_window) == window_size and all(x < min_delta for x in improvement_window):
            break

        best_probability = 1 / (1 + best_val)

        new_pop = population[:population_size//2]
        while len(new_pop) < population_size:
            p1, p2 = random.sample(population[:population_size//2], 2)
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_pop.append(child)
        population = new_pop

    best_solution = population[0]
    best_value = fitness(best_solution)
    total_time = time.time() - start_time

    return {
        "best_solution": best_solution,
        "best_value": best_value,
        "total_time": total_time,
        "generations": generations,
        "values_curve": history,
        "n_evals": evals,
        "best_measurement_probability": best_probability
    }


def tabu_search(objective_function, generate_neighbors, initial_solution,
                edges, num_vertices, max_iter=100, tabu_size=10, min_delta=1e-6, window_size=5):
    start_time = time.time()
    evals = 0
    history = []
    best_probability = 0

    current = initial_solution(num_vertices)
    best = current.copy()
    best_score = objective_function(edges, current); evals += 1
    history.append(best_score)

    improvement_window = deque(maxlen=window_size)
    improvement_window.append(0)

    tabu_list = []

    for it in range(max_iter):
        neighs = [n for n in generate_neighbors(current) if n not in tabu_list]
        if not neighs: break

        scores = [objective_function(edges, n) for n in neighs]; evals += len(neighs)
        current = neighs[np.argmax(scores)]
        tabu_list.append(current)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        cur_score = objective_function(edges, current); evals += 1
        if cur_score > best_score:
            best_score = cur_score
            best = current.copy()

        best_probability = 1 / (1 + best_score)

        history.append(best_score)

        if len(history) > 1:
            improvement_window.append(abs(history[-1] - history[-2]))

        if len(improvement_window) == window_size and all(x < min_delta for x in improvement_window):
            break

    total_time = time.time() - start_time

    return {
        "best_solution": best,
        "best_value": best_score,
        "total_time": total_time,
        "iterations": it+1,
        "values_curve": history,
        "n_evals": evals,
        "best_measurement_probability": best_probability
    }


def ant_colony_optimization(distance_matrix, num_ants=10, num_iterations=100,
                            alpha=1, beta=2, evaporation=0.5):
    start_time = time.time()
    evals = 0
    history = []

    num_nodes = len(distance_matrix)
    pheromones = np.ones((num_nodes, num_nodes))
    best_path, best_dist = None, float("inf")

    for it in range(num_iterations):
        all_paths, all_dists = [], []
        for _ in range(num_ants):
            path = np.random.permutation(num_nodes)

            dist = sum(distance_matrix[path[i]][path[i+1]]
                       for i in range(num_nodes-1))
            dist += distance_matrix[path[-1]][path[0]]
            evals += 1

            all_paths.append(path)
            all_dists.append(dist)
            if dist < best_dist:
                best_dist = dist
                best_path = path.copy()
        history.append(best_dist)

        pheromones *= (1-evaporation)
        for path, dist in zip(all_paths, all_dists):
            for i in range(num_nodes-1):
                pheromones[path[i], path[i+1]] += 1/dist

    total_time = time.time() - start_time

    return {
        "best_solution": best_path,
        "best_value": best_dist,
        "total_time": total_time,
        "iterations": num_iterations,
        "values_curve": history,
        "n_evals": evals
    }
