import numpy as np
import networkx as nx
from qiskit_ibm_runtime import QAOA

from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from qiskit_optimization.applications import Maxcut

from code.tasks import cut_max
from code.utils import max_cut_generator_graph


algorithm_globals.random_seed = 42
# Генерация графа
n = 4
m = 3
G = max_cut_generator_graph.create_weighted_graph(n, m)
edge_labels = {k: f'{float(v):.3f}' for k, v in nx.get_edge_attributes(G, 'weight').items()}
elist = [(*key, int(float(value))) for key, value in edge_labels.items()]

# Формирование матрицы весов
w = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        temp = G.get_edge_data(i, j, default=0)
        if temp != 0:
            w[i, j] = temp["weight"]

# Создание задачи Max-Cut
max_cut = Maxcut(w)
qp = max_cut.to_quadratic_program()

# Преобразование в Ising Hamiltonian
qubitOp, offset = qp.to_ising()

# Загрузка учетных данных IBM Quantum
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)
session = Session(backend=backend)
sampler = Sampler(mode=session)

# Создание решателя QAOA
optimizer = COBYLA(maxiter=300)

qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=2)

# Вычисление минимального собственного значения
result = qaoa.compute_minimum_eigenvalue(qubitOp)

# Получение решения
x = max_cut.sample_most_likely(result.eigenstate)

# Вывод результатов
print("🧮 Energy:", result.eigenvalue.real)
print("⏱ Optimization time:", result.optimizer_time)
print("📈 Max-Cut objective (adjusted):", result.eigenvalue.real + offset)
print("🧩 Bitstring solution:", x)
print("🎯 Cut value (custom):", cut_max.objective_function(elist, x))
print("🎯 Cut value (Qiskit):", qp.objective.evaluate(x))

# --- 🔍 Название используемого бэкенда ---
print("🖥️ Executed on backend:", backend.name)
print(backend.configuration)
print(backend.provider)
print(backend.properties)