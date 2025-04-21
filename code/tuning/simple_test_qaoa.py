import optuna
import numpy as np

from code.quantum.algorythms import qaoa_solver
from code.utils import max_cut_generator_graph
from code.tasks import cut_max
from tqdm import tqdm
import json

results = {}

SUPPORTED_OPTIMIZERS = [
    "COBYLA", "SPSA", "ADAM", "L_BFGS_B",
    "NELDER_MEAD", "POWELL", "SLSQP", "TNC"
]

def make_objective(n, graph_list):
    def objective(trial):
        optimizer_type = trial.suggest_categorical("optimizer", ["COBYLA", "SPSA", "ADAM", "L_BFGS_B", "NELDER_MEAD", "POWELL", "SLSQP", "TNC"])
        reps = trial.suggest_int("reps", 1, 2)
        maxiter = trial.suggest_int("maxiter", 100, 1000)

        values = []

        for G in graph_list:
            edges = [(u, v, int(float(data))) for u, v, data in G.edges(data="weight")]

            try:
                result = qaoa_solver(
                    n=n,
                    G=G,
                    elist=edges,
                    reps=reps,
                    optimizer_type=optimizer_type,
                    maxiter=maxiter
                )

                if "optimal_value" in result:
                    values.append(-result["optimal_value"])
                else:
                    raise ValueError("QAOA result missing 'optimal_value'")
            except Exception as e:
                continue

        if values:
            return np.mean(values)
        else:
            raise ValueError("Нет допустимых результатов для оценки")

    return objective



for n in tqdm(range(11, 16), desc="Размеры графа"):
    graph_list = [max_cut_generator_graph.create_weighted_graph(n, int(n * 1.5)) for _ in range(10)]

    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(n, graph_list), n_trials=5)

    results[n] = {
        "best_params": study.best_params,
        "best_value": -study.best_value,
    }

with open("../results/qaoa_optuna_results_more_10.json", "w") as f:
    json.dump(results, f, indent=4)

for n in sorted(results):
    print(f"\nГраф размером n={n}")
    print("Лучшие параметры:", results[n]["best_params"])
    print("Среднее значение целевой функции:", results[n]["best_value"])
