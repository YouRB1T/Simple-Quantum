import math
import random
import time
import numpy as np


def simulated_annealing(objective_function, initial_solution, generate_neighbor_solution, num_vertices,
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
    best_value = objective_function(s)

    times = []
    values = []
    start_time = time.time()
    iteration = 0

    while t > t_min and iteration < max_iterations:
        s_ = generate_neighbor_solution(s)
        delta = objective_function(s_) - objective_function(s)

        if delta > 0:
            s = s_.copy()
        else:
            p = math.exp(delta / t)
            if random.uniform(0, 1) < p:
                s = s_.copy()

        current_value = objective_function(s)
        if current_value > best_value:
            best_solution = s.copy()
            best_value = current_value

        t = t * alpha
        iteration += 1

        values.append(best_value)

    ans_value = min(values)
    ans_time = time.time() - start_time
    return best_solution, best_value, ans_time, ans_value


def genetic_algorithm(objective_function, generate_solution, mutate, crossover,
                      population_size=50, generations=100, mutation_rate=0.1):
    """
    Простая реализация генетического алгоритма

    :param objective_function: целевая функция задачи
    :param generate_solution: функция генерации случайного решения
    :param mutate: функция мутации
    :param crossover: функция скрещивания
    :param population_size: размер популяции
    :param generations: кол-во поколений
    :param mutation_rate: вероятность мутации
    :return: taple(person_of_population, working_time)
                возвращаем лучшего потомка, который будет лучшим по целевой функцииБ а также ремя работы метода
    """
    start_time = time.time()

    population = [generate_solution() for _ in range(population_size)]

    for _ in range(generations):
        population = sorted(population, key=objective_function, reverse=True)

        new_population = population[: population_size // 2]

        while len(new_population) < population_size:
            p1, p2 = random.sample(population[: population_size // 2], 2)
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)

        population = new_population

    return max(population, key=objective_function), time.time() - start_time


def tabu_search(objective_function, generate_neighbors, initial_solution, max_iter=100, tabu_size=10):
    """
    Алгоритм табу поиска
    :param objective_function: целевая функция задачи
    :param generate_neighbors: функция генерации соседей
    :param initial_solution: начальное решение задачи
    :param max_iter: кол-во максимальных итераций
    :param tabu_size: размер табу списка
    :return: taple(best_solution< working_time) вернем лучшее решение и время работы
    """
    stat_time = time.time()

    current_solution = initial_solution
    best_solution = current_solution
    best_score = objective_function(best_solution)

    tabu_list = []

    for _ in range(max_iter):
        neighbors = generate_neighbors(current_solution)

        neighbors = [n for n in neighbors if n not in tabu_list]

        if not  neighbors:
            break

        current_solution = max(neighbors, key=objective_function)

        tabu_list.append(current_solution)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        current_score = objective_function(current_solution)
        if current_score > best_score:
            best_solution = current_solution
            best_score = current_score

    return best_solution, time.time() - stat_time


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
            distance = sum(distance_matrix[path[i], path[i+1]] for i in range(num_nodes - 1))
            distance += distance_matrix[path[-1], path[0]]

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