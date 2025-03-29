import random


def create_objective_function(edges, partition):
    """
    Создаёт целевую функцию для задачи Max-Cut на основе графа.

    Args:
        graph (list of tuples): Список рёбер графа.
        partitions (list): списак принадлежности вершин к компонентам связанности
    Returns:
        function: Функция, которая вычисляет размер разреза для заданного разделения.
    """
    return sum(w for i, j, w in edges if partition[i] != partition[j])


def initial_solution(num_vertices):
    """
    Генерирует начальное случайное разделение.

    Args:
        num_vertices (int): Число вершин в графе.

    Returns:
        list: Список из 0 и 1, представляющий начальное разделение.
    """
    return [random.randint(0, 1) for _ in range(num_vertices)]


def generate_neighbor_solution(partition):
    """
    Генерирует соседнее решение, изменяя одну вершину.

    Args:
        partition (list): Текущее разделение.

    Returns:
        list: Новое разделение.
    """
    new_partition = partition.copy()
    vertex_to_change = random.randint(0, len(partition) - 1)
    new_partition[vertex_to_change] = 1 - new_partition[vertex_to_change]
    return new_partition