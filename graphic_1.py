import json
import matplotlib.pyplot as plt

filename = 'benchmark_results_5.json'
with open(filename, 'r') as f:
    data = json.load(f)

algorithms = ['qaoa', 'simulated_annealing', 'genetic_algorithm', 'tabu_search', 'random_baseline']
results = {algo: [] for algo in algorithms}

for algo in algorithms:
    print(f"Обработка {algo}...")
    for result in data[algo]:
        if algo == 'qaoa':
            value = result.get('best_objective')
        elif algo == 'random_baseline':
            value = result.get('mean_value')
        else:
            value = result.get('best_value')

        if value is not None:

            if isinstance(value, dict) and "__complex__" in value:
                value = value["__complex__"][0]
            results[algo].append(value)

plt.figure(figsize=(10, 6))
styles = {
    'qaoa': {'color': 'blue', 'linestyle': 'dashed', 'marker': 'o', 'label': 'QAOA'},
    'simulated_annealing': {'color': 'red', 'linestyle': '--', 'marker': 'x', 'label': 'Simulated Annealing'},
    'genetic_algorithm': {'color': 'green', 'linestyle': '-.', 'marker': 's', 'label': 'Genetic Algorithm'},
    'tabu_search': {'color': 'purple', 'linestyle': ':', 'marker': 'D', 'label': 'Tabu Search'},
    'random_baseline': {'color': 'yellow', 'linestyle': '-', 'marker': 'D', 'label': 'Random'}
}

print(results.get('random_baseline'))

for algo in algorithms:
    plt.plot(results[algo],
             color=styles[algo]['color'],
             linestyle=styles[algo]['linestyle'],
             marker=styles[algo]['marker'],
             label=styles[algo]['label'],
             markersize=6, linewidth=2)

plt.xlabel('Номер графа')
plt.ylabel('Значение целевой функции')
plt.title('Качество решения для различных алгоритмов')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
