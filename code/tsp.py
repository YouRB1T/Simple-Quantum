import math
import random
import time


def objective_function(solution, distance_matrix):
    """
    Вычисляет длину маршрута коммивояжера по заданной матрице расстояний.
    :param solution: маршрут (перестановка городов)
    :param distance_matrix: матрица расстояний между городами
    :return: общая длина маршрута
    """
    return sum(distance_matrix[solution[i - 1]][solution[i]] for i in range(len(solution)))


def generate_initial_solution(num_vertices):
    """
    Генерирует случайное начальное решение (перестановку городов).
    :param num_vertices: количество городов
    :return: случайный маршрут
    """
    solution = list(range(num_vertices))
    random.shuffle(solution)
    return solution


def generate_neighbor_solution(solution):
    """
    Генерирует соседнее решение путем реверса случайного отрезка маршрута.
    :param solution: текущее решение
    :return: измененное соседнее решение
    """
    i, j = sorted(random.sample(range(len(solution)), 2))
    neighbor = solution[:]
    neighbor[i:j] = reversed(neighbor[i:j])
    return neighbor


def mutate(solution):
    """
    Мутирует маршрут, реверсируя случайный отрезок.
    :param solution: маршрут
    :return: мутированное решение
    """
    i, j = sorted(random.sample(range(len(solution)), 2))
    solution[i:j] = reversed(solution[i:j])
    return solution


def crossover(parent1, parent2):
    """
    Реализует кроссовер (скрещивание) двух родительских маршрутов.
    :param parent1: первый родитель
    :param parent2: второй родитель
    :return: потомок
    """
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    pointer = end
    for i in range(size):
        if parent2[i] not in child:
            if pointer >= size:
                pointer = 0
            child[pointer] = parent2[i]
            pointer += 1
    return child
