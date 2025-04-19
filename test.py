import numpy as np
import time
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session
from scipy.optimize import minimize
from code.quantum import hemiltonians
from qiskit.circuit.library import QAOAAnsatz

from code.utils import max_cut_generator_graph

# 1. Построение MaxCut‑графа на 5 вершинах
n = 5
edge_list = [
    (0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0),
    (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)
]

G = max_cut_generator_graph.create_weighted_graph(127, 10)
n = len(G)
print(G.edges)
edge_list = G.edges

# 2. Гамильтониан и shift
hamiltonian, shift = hemiltonians.max_cut_hemiltonian(edge_list, n)

# 3. QAOA‑Ansatz и начальные параметры
reps = 2
init_params = [np.pi, np.pi/2] * reps
ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=reps)

# 4. Подключаемся к IBM Runtime и берём самый быстрый симулятор на ≥127 кубитах
service = QiskitRuntimeService()
backend = service.least_busy(min_num_qubits=127)
print(backend)

# 5. Транспилируем Ansatz под целевой backend (чтобы не было ошибки ISA)
ansatz = transpile(ansatz, backend=backend, optimization_level=1)

# 6. Функция стоимости (EstimatorV2 требует mode=session)
def cost(params, ansatz, hamiltonian, estimator):
    pubs = [(ansatz, hamiltonian, params)]
    job = estimator.run(pubs)
    res  = job.result()
    return res[0].data.evs[0] + shift

# 7. Запускаем Qiskit‑сессию и оптимизацию
start = time.time()
with Session(backend=backend) as session:
    print(f"start session: {session.details}")

    # создаём EstimatorV2 через mode=session
    estimator = Estimator(mode=session)
    estimator.options.default_shots = 512

    result = minimize(
        cost,
        init_params,
        args=(ansatz, hamiltonian, estimator),
        method="COBYLA",
        tol=1e-2
    )
end = time.time()

# 8. Выводим результаты
print("=== QAOA (Runtime) ===")
print("Params:         ", result.x)
print("Min energy+shift:", result.fun)
print("Success:        ", result.success)
print("Iterations:     ", getattr(result, "nit", None))
print(f"Elapsed time:   {end - start:.2f} s")
