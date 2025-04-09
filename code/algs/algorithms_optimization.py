import math
import random
import time
import numpy as np


def simulated_annealing(objective_function, initial_solution, generate_neighbor_solution, num_vertices, edges,
                               t=1000, t_min=0.01, alpha=0.95, max_iterations=1000):
    """
    Алгоритм симуляции отжига
    :param objective_function: целевая функция задачи
    :param initial_solution: начальное условие задачи
    :param generate_neighbor_solution: генерация близкой задачи
    :param num_vertices: кол-во вершин в графе
    :param t: начальная температура
    :param t_min: минимальная температура
    :param alpha: уменьшение температуры
    :param max_iterations: предельное кол-во итераций
    :return: tuple: (best_solution, best_value, times, values) - лучшее разделение, размер разреза,
               времена и значения целевой функции на каждой итерации.
    """
    s = initial_solution(num_vertices)
    best_solution = s.copy()
    best_value = objective_function(edges, s)

    values = []
    start_time = time.time()
    iteration = 0

    while t > t_min and iteration < max_iterations:
        s_ = generate_neighbor_solution(s)
        delta = objective_function(edges, s_) - objective_function(edges, s)

        if delta > 0:
            s = s_.copy()
        else:
            p = math.exp(delta / t)
            if random.uniform(0, 1) < p:
                s = s_.copy()

        current_value = objective_function(edges, s)
        if current_value > best_value:
            best_solution = s.copy()
            best_value = current_value

        t = t * alpha
        iteration += 1

        values.append(best_value)

    ans_time = time.time() - start_time
    return best_solution, best_value, ans_time


def genetic_algorithm(objective_function, initial_solution_fn, mutate, crossover, num_vertices, edges,
                      population_size=50, generations=100, mutation_rate=0.1):
    """
    Реализация стандартоного генетического алгоритма
    Args:
        objective_function: целевая функция задачи
        initial_solution_fn: функция генерации начальных данных задачи
        mutate: функция мутации
        crossover: функция смешивания
        num_vertices: кол-во вершин
        edges: ребра
        population_size: ращмер популяции
        generations: кол-во поколений
        mutation_rate: коэффициент мутации

    Returns:
        best_solution: лучшее решение
        elapsed_time: время работы алгоритма

    """
    start_time = time.time()

    population = [initial_solution_fn(num_vertices) for _ in range(population_size)]

    def wrapped_fitness(partition):
        return objective_function(edges, partition)

    for _ in range(generations):
        population = sorted(population, key=wrapped_fitness, reverse=True)
        new_population = population[: population_size // 2]

        while len(new_population) < population_size:
            p1, p2 = random.sample(population[: population_size // 2], 2)
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)

        population = new_population

    best_solution = max(population, key=wrapped_fitness)

    elapsed_time = time.time() - start_time

    return best_solution, elapsed_time, wrapped_fitness(best_solution)



def tabu_search(objective_function, generate_neighbors, initial_solution, edges,
                num_vertices, max_iter=100, tabu_size=10):
    """
    Алгоритм табу поиска для задачи Max-Cut

    :param objective_function: функция оценки (cut_max.objective_function)
    :param generate_neighbors: функция генерации соседей (cut_max.generate_neighbors)
    :param initial_solution: функция генерации начального решения (cut_max.initial_solution)
    :param edges: список рёбер графа
    :param num_vertices: количество вершин графа
    :param max_iter: число итераций
    :param tabu_size: размер табу-списка
    :return: (лучшая особь, время работы)
    """
    start_time = time.time()

    current_solution = initial_solution(num_vertices)
    best_solution = current_solution
    best_score = objective_function(edges, best_solution)

    tabu_list = []

    for _ in range(max_iter):
        neighbors = generate_neighbors(current_solution)

        neighbors = [n for n in neighbors if n not in tabu_list]

        if not neighbors:
            break

        current_solution = max(neighbors, key=lambda sol: objective_function(edges, sol))

        tabu_list.append(current_solution)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        current_score = objective_function(edges, current_solution)
        if current_score > best_score:
            best_solution = current_solution
            best_score = current_score

    return best_solution, time.time() - start_time, best_score


def ant_colony_optimization(distance_matrix,  num_ants=10, num_iterations=100, alpha=1, beta=2, evaporation=0.5):
    """
    Муравьиный алгоритм для поиска кратчайшего пути.
    :param distance_matrix: Матрица растояний
    :param num_ants: кол-во муравьев
    :param num_iterations: кол-во итераций
    :param alpha: влияние феромона
    :param beta: влияние расстояния
    :param evaporation: коэффициент испарения феромона
    :return: taple(best_path, best_distance, working_time) вернем лучший путь, и лучшу дистанцию,
                а также время выполнения алгоритма
    """
    start_time = time.time()
    num_nodes = len(distance_matrix)
    pheromones = np.ones((num_nodes, num_nodes))
    best_path = None
    best_distance = float('inf')

    for _ in range(num_iterations):
        all_paths = []
        all_distances = []

        for _ in range(num_ants):
            path = np.random.permutation(num_nodes)
            distance = sum(distance_matrix[path[i]][path[i+1]] for i in range(num_nodes - 1))
            distance += distance_matrix[path[-1]][path[0]]

            all_paths.append(path)
            all_distances.append(distance)

            if distance < best_distance:
                best_distance = distance
                best_path = path

        pheromones *= (1 - evaporation)
        for path, dist in zip(all_paths, all_distances):
            for i in range(num_nodes - 1):
                pheromones[path[i], path[i+1]] += 1 / dist

    return best_path, best_distance, time.time() - start_time