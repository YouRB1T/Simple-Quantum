import random


def objective_function(edges, partition,
                       penalty_weight=1.0,
                       balance_weight=0.1,
                       important_weight=2.0,
                       important_edges=None):
    """

    Args:
        edges:
        partition:
        penalty_weight:
        balance_weight:
        important_weight:
        important_edges:

    Returns:

    """
    cut_size = sum(w for i, j, w in edges if partition[i] != partition[j])


    return cut_size


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


def generate_neighbors(solution):
    """

    Args:
        solution:

    Returns:

    """
    neighbors = []
    for i in range(len(solution)):
        neighbor = solution.copy()
        neighbor[i] = 1 - neighbor[i]
        neighbors.append(neighbor)
    return neighbors


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


def mutate(child):
    """

    Args:
        child:

    Returns:

    """
    index = random.randint(0, len(child) - 1)
    child[index] = 1 - child[index]
    return child