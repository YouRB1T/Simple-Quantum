import random


def create_objective_function(edges, partition, penalty_weight=1.0):
    """
    Создаёт целевую функцию для задачи Max-Cut на основе графа.

    Args:
        graph (list of tuples): Список рёбер графа.
        partitions (list): списак принадлежности вершин к компонентам связанности
    Returns:
        function: Функция, которая вычисляет размер разреза для заданного разделения.
    """
    cut_size = sum(w for i, j, w in edges if partition[i] != partition[j])

    penalty = sum(w * penalty_weight for i, j, w in edges if partition[i] == partition[j])

    return cut_size - penalty


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

def crossover(parent1, parent2):
    """
    Выполняет одноточечное скрещивание между двумя решениями.

    Args:
        parent1 (list): Первое родительское решение.
        parent2 (list): Второе родительское решение.

    Returns:
        list: Потомок после кроссовера.
    """
    crossover_point = random.randint(1, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]
