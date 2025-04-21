import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from code.tasks.cut_max import objective_function

n = 5
# Загрузка результатов из JSON
filename = f'benchmark_results_{n}.json'
with open(filename, 'r') as f:
    data = json.load(f)

# Предполагаем, что результаты лежат в виде словарей по алгоритмам
# Для примера возьмем QAOA, Simulated Annealing, Genetic Algorithm, Tabu Search
algorithms = ['qaoa', 'genetic_algorithm', 'tabu_search']

# Загрузка графа (предположим, что граф генерируется здесь)  # примерный размер графа
m = 3
G = nx.gnp_random_graph(n, 0.5)  # Сгенерируем случайный граф с n вершинами

# Преобразуем граф в формат, используемый в функции objective_function
edges = [(u, v, 1.0) for u, v in G.edges()]  # Предполагаем, что веса рёбер равны 1.0

# Список для хранения результатов целевой функции для разных алгоритмов
efficiency_values = {}

# Проходим по каждому алгоритму и рассчитываем значение целевой функции для каждой итерации
for algo in algorithms:
    print(f"Обработка {algo}...")

    efficiency_value = []

    for idx, result in enumerate(data[algo]):

        if algo == 'qaoa':
            bitstring = result.get('best_measurement', {}).get('bitstring', '')
            total_time = result.get('total_time', None)
            if bitstring and total_time is not None and total_time > 0:
                partition = [int(bit) for bit in bitstring]  # Преобразуем битстроку в разбиение
                objective_value = objective_function(edges, partition)  # Рассчитываем целевую функцию
                efficiency = objective_value / total_time
                efficiency_value.append(efficiency)

        else:
            best_value = result.get('best_value', None)
            total_time = result.get('total_time', None)
            if best_value is not None and total_time is not None and total_time > 0:
                efficiency = best_value / total_time
                efficiency_value.append(efficiency)

    efficiency_values[algo] = np.mean(efficiency_value)

plt.figure(figsize=(10, 6))

# Столбцы для каждого алгоритма
plt.bar(efficiency_values.keys(), efficiency_values.values(), color=['blue', 'red', 'green', 'purple'])

# Оформление графика
plt.xlabel('Алгоритмы', fontsize=14)
plt.ylabel('Среднее значение целевой функции', fontsize=14)
plt.title(f'Среднее значение целевой функции {n} вершинных графов для различных алгоритмов', fontsize=16)

# Добавление значений на вершине столбцов для лучшего восприятия
for i, v in enumerate(efficiency_values.values()):
    plt.text(i, v + 0.05, round(v, 2), ha='center', va='bottom', fontsize=12)

# Показываем график
plt.tight_layout()
plt.show()
