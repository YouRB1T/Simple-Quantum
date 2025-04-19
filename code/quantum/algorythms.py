from json import dumps

import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA, COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.applications import Maxcut
from qiskit.qasm2 import dumps as qasm_dumps

from code.tasks import cut_max


def qaoa_solver(n, G, elist, reps=2):
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
    qaoa = QAOA(sampler=Sampler(), optimizer=optimizer, reps=reps)

    result = qaoa.compute_minimum_eigenvalue(qubitOp)

    return serialize_sampling_vqe_result(result)


def serialize_sampling_vqe_result(result):
    aux = result.aux_operators_evaluated
    best_meas = result.best_measurement
    cost_evals = result.cost_function_evals

    # Преобразуем QuasiDistribution в обычный словарь
    eigen_dist = dict(result.eigenstate)

    eig = result.eigenvalue

    # 🔧 ПРИВЯЗЫВАЕМ ПАРАМЕТРЫ перед сериализацией в QASM
    bound_circuit = result.optimal_circuit.assign_parameters(result.optimal_parameters)
    circ = qasm_dumps(bound_circuit)

    params = dict(result.optimal_parameters)
    point = list(result.optimal_point)
    opt_val = result.optimal_value
    opt_evals = result.optimizer_evals

    opt_res = result.optimizer_result
    optimizer_result = {
        "x": list(opt_res.x),
        "fun": opt_res.fun,
        "nit": getattr(opt_res, "nit", None),
        "message": getattr(opt_res, "message", None)
    }

    opt_time = result.optimizer_time

    return {
        "aux_operators_evaluated": aux,
        "best_measurement": best_meas,
        "cost_function_evals": cost_evals,
        "eigenstate": eigen_dist,
        "eigenvalue": eig,
        "optimal_circuit_qasm": circ,
        "optimal_parameters": params,
        "optimal_point": point,
        "optimal_value": opt_val,
        "optimizer_evals": opt_evals,
        "optimizer_result": optimizer_result,
        "total_time": opt_time
    }
