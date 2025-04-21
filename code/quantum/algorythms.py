from json import dumps

import numpy as np
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA, COBYLA, ADAM, L_BFGS_B, NELDER_MEAD, POWELL, SLSQP, TNC
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.applications import Maxcut
from qiskit.qasm2 import dumps as qasm_dumps
from code.tasks.cut_max import objective_function


def qaoa_solver(n, G, elist, reps=2, optimizer_type="COBYLA", maxiter=300):
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = temp["weight"]
                w[j, i] = temp["weight"]

    max_cut = Maxcut(w)
    qp = max_cut.to_quadratic_program()

    qubitOp, offset = qp.to_ising()

    algorithm_globals.random_seed = 123

    optimizer_type = optimizer_type.upper()
    if optimizer_type == "COBYLA":
        optimizer = COBYLA(maxiter=maxiter)
    elif optimizer_type == "SPSA":
        optimizer = SPSA(maxiter=maxiter)
    elif optimizer_type == "ADAM":
        optimizer = ADAM(maxiter=maxiter)
    elif optimizer_type == "L_BFGS_B":
        optimizer = L_BFGS_B(maxiter=maxiter)
    elif optimizer_type == "NELDER_MEAD":
        optimizer = NELDER_MEAD(maxiter=maxiter)
    elif optimizer_type == "POWELL":
        optimizer = POWELL(maxiter=maxiter)
    elif optimizer_type == "SLSQP":
        optimizer = SLSQP(maxiter=maxiter)
    elif optimizer_type == "TNC":
        optimizer = TNC(maxiter=maxiter)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {optimizer_type}")

    qaoa = QAOA(sampler=Sampler(), optimizer=optimizer, reps=reps)

    result = qaoa.compute_minimum_eigenvalue(qubitOp)

    bitstring = result.best_measurement.get("bitstring")

    partition = [int(bit) for bit in bitstring]

    return serialize_sampling_vqe_result(result, objective_function(elist, partition))


def serialize_sampling_vqe_result(result, objective):
    aux = result.aux_operators_evaluated
    best_meas = result.best_measurement
    cost_evals = result.cost_function_evals

    eigen_dist = dict(result.eigenstate)

    eig = result.eigenvalue

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
        "best_objective": objective,
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
