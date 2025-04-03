import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals


def max_cut_cost_hamiltonian(edges, num_vertices):
    pauli_terms = []
    coeffs = []

    for i, j in edges:
        z_terms = ['I'] * num_vertices
        z_terms[i], z_terms[j] = 'Z', 'Z'
        pauli_terms.append("".join(z_terms))
        coeffs.append(-0.5)

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))


def max_cut_mixing_hamiltonian(num_vertices):
    pauli_terms = []
    coeffs = []

    for i in range(num_vertices):
        term = ["I"] * num_vertices
        term[i] = "X"
        pauli_terms.append("".join(term))
        coeffs.append(-1.0)

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))


w = np.array([[0.0, 1.0, 1.0, 0.0],
              [1.0, 0.0, 1.0, 1.0],
              [1.0, 1.0, 0.0, 1.0],
              [0.0, 1.0, 1.0, 0.0]])

G = nx.from_numpy_array(w)
edges = list(G.edges())
num_vertices = len(G.nodes)

cost_hamiltonian = max_cut_cost_hamiltonian(edges, num_vertices)
mixing_hamiltonian = max_cut_mixing_hamiltonian(num_vertices)

algorithm_globals.random_seed = 10598
sampler = Sampler()
optimizer = COBYLA()

qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=2, mixer=mixing_hamiltonian)

result = qaoa.compute_minimum_eigenvalue(cost_hamiltonian)

def bitfield(n, L):
    return [int(digit) for digit in np.binary_repr(n, L)]


best_cut = result
print(f"Лучший разрез: {best_cut}")


