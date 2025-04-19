import numpy as np
import time
from qiskit import transpile
from scipy.optimize import minimize
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Sampler, Session, EstimatorOptions, SamplerOptions
from qiskit.circuit.library import QAOAAnsatz

from code.utils import max_cut_generator_graph
from code.quantum import hemiltonians

G = max_cut_generator_graph.create_weighted_graph(127, 1)
n = len(G.nodes)
edge_list = list(G.edges(data=True))
print("Рёбра графа:", edge_list)

hamiltonian, shift = hemiltonians.max_cut_hemiltonian(edge_list, n)

reps = 2
init_params = [np.pi, np.pi/2] * reps
ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=reps)

service = QiskitRuntimeService()
backend = service.least_busy(simulator=False, operational=True, min_num_qubits=n)
print("Используем бэкенд:", backend.name)

ansatz = transpile(ansatz, backend=backend, optimization_level=1)

def cost(params, ansatz, hamiltonian, estimator):
    pubs = [(ansatz, hamiltonian, params)]
    job = estimator.run(pubs)
    pub_result = job.result()[0]
    ev = pub_result.data.evs
    return ev + shift

start = time.time()
with Session(backend=backend) as session:
    est_opts = EstimatorOptions()
    est_opts.default_shots = 100
    estimator = Estimator(mode=session, options=est_opts)

    result = minimize(
        cost,
        init_params,
        args=(ansatz, hamiltonian, estimator),
        method="COBYLA",
        tol=1e-2
    )
    elapsed = time.time() - start

    optimal_params = result.x
    print("Оптимальные параметры:", optimal_params)
    print("Время оптимизации:", elapsed)

    bound_circuit = ansatz.assign_parameters(optimal_params)
    bound_circuit.measure_all()

    sampler_opts = SamplerOptions(shots=100)
    sampler = Sampler(mode=session, options=sampler_opts)
    samp_job = sampler.run([bound_circuit])
    quasi_dist = samp_job.result().quasi_dists[0]

best_bitstring = max(quasi_dist, key=quasi_dist.get)
prob = quasi_dist[best_bitstring]
bit_str = best_bitstring if isinstance(best_bitstring, str) else format(best_bitstring, f'0{n}b')
bits = np.array([int(b) for b in bit_str])

print("Лучшая строка:", bit_str)
print("Массив битов:", bits)
print("Вероятность:", prob)
