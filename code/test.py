import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from qiskit.primitives import Sampler
from qiskit.result import QuasiDistribution
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

import utils_graph
from hemiltonians import max_cut_hemiltonian

w = np.array(
    [[0.0, 1.0, 1.0, 0.0],
     [1.0, 0.0, 1.0, 1.0],
     [1.0, 1.0, 0.0, 1.0],
     [0.0, 1.0, 1.0, 0.0]]
)
ages, n = utils_graph.convert_form_matrix(w)

G = nx.from_numpy_array(w)

qubit_op, offset = max_cut_hemiltonian(ages, n)
sampler = Sampler()

def objective_value(x, w):
    X = np.outer(x, (1 - x))
    w_01 = np.where(w != 0, 1, 0)
    return np.sum(w_01 * X)

def bitfield(n, L):
    result = np.binary_repr(n, L)
    return [int(digit) for digit in result]

def sample_most_likely(state_vector):
    if isinstance(state_vector, QuasiDistribution):
        values = list(state_vector.values())
    else:
        values = state_vector
    n = int(np.log2(len(values)))
    k = np.argmax(np.abs(values))
    x = bitfield(k, n)
    x.reverse()
    return np.asarray(x)

algorithm_globals.random_seed = 10598
optimizer = COBYLA()
qaoa = QAOA(sampler, optimizer, reps=2)
result = qaoa.compute_minimum_eigenvalue(qubit_op)

x = sample_most_likely(result.eigenstate)
print(f"Разбиение вершин: {x}")
print(f"Objective value: {objective_value(x, w)}")

layout = nx.spring_layout(G, seed=10)
node_colors = ['red' if xi == 0 else 'blue' for xi in x]

nx.draw(G, layout, node_color=node_colors, with_labels=True, edge_color="gray")

labels = {(i, j): w[i, j] for i, j in G.edges()}
nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels)

plt.show()
