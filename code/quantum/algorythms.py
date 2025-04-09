import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.applications import Maxcut

from code.tasks import cut_max


def qaoa_solver(n, G, elist):
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = temp["weight"]

    max_cut = Maxcut(w)
    qp = max_cut.to_quadratic_program()

    qubitOp, offset = qp.to_ising()

    algorithm_globals.random_seed = 123

    optimizer = COBYLA(maxiter=300)
    qaoa = QAOA(sampler=Sampler(), optimizer=optimizer, reps=2)

    result = qaoa.compute_minimum_eigenvalue(qubitOp)

    x = max_cut.sample_most_likely(result.eigenstate)

    return cut_max.objective_function(elist, x), result.optimizer_time