import optuna
import numpy as np
from code.utils import max_cut_generator_graph
from code.tasks import cut_max
from code.algs import algorithms_optimization
from tqdm import tqdm
import json

results = {}

def make_objective(n, graph_list):
    def objective(trial):
        population_size = trial.suggest_int("population_size", 10, 200)
        mutation_rate = trial.suggest_float("mutation_rate", 0.01, 0.3)
        generations = trial.suggest_int("generations", 30, 300)

        values = []

        for G in graph_list:
            edges = [(u, v, int(float(data))) for u, v, data in G.edges(data="weight")]

            result = algorithms_optimization.genetic_algorithm(
                objective_function=cut_max.objective_function,
                initial_solution_fn=cut_max.initial_solution,
                mutate=cut_max.mutate,
                crossover=cut_max.crossover,
                num_vertices=n,
                edges=edges,
                population_size=population_size,
                generations=generations,
                mutation_rate=mutation_rate
            )

            values.append(result["best_value"])

        return -np.mean(values)
    return objective

for n in tqdm(range(5, 21), desc="Размеры графа"):
    graph_list = [max_cut_generator_graph.create_weighted_graph(n, int(n * 1.5)) for _ in range(50)]

    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(n, graph_list), n_trials=30)

    results[n] = {
        "best_params": study.best_params,
        "best_value": -study.best_value,
    }

with open("genetic_optuna_results.json", "w") as f:
    json.dump(results, f, indent=4)

for n in sorted(results):
    print(f"\nГраф размером n={n}")
    print("Лучшие параметры:", results[n]["best_params"])
    print("Среднее значение целевой функции:", results[n]["best_value"])
