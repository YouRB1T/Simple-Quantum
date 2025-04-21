import random

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit.library import QAOAAnsatz

weight_range = 1

def max_cut_hemiltonian(ages, n):
    pauli_list = []
    shift = 0

    for i, j, data in ages:
        pauli_list = []
        for edge in ages:
            paulis = ["I"] * n
            paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

            weight = data.get('weight', weight_range)

            pauli_list.append(("".join(paulis)[::-1], weight))
        shift += 0.5 * weight


    return SparsePauliOp.from_list(pauli_list), shift


def yan_max_cut_hemiltonian(ages, n):
    pauli_list = []
    coeffs = []
    shift = 0

    for i, j in ages:
        w = 1

        x_p = np.zeros(n, dtype=bool)
        z_p = np.zeros(n, dtype=bool)
        z_p[i] = True
        z_p[j] = True
        pauli_list.append(Pauli((z_p, x_p)))
        coeffs.append(-0.5 * w)  # Умножаем на вес
        shift += 0.5 * w  # Также с учётом веса

    for i, j in ages:
        w = 1

        x_p = np.zeros(n, dtype=bool)
        z_p = np.zeros(n, dtype=bool)
        z_p[i] = True
        z_p[j] = True
        pauli_list.append(Pauli((z_p, x_p)))
        coeffs.append(1.0 * w)  # Умножаем на вес

    shift += n

    return SparsePauliOp(pauli_list, coeffs=coeffs), shift


# https://qiskit-community.github.io/qiskit-optimization/tutorials/06_examples_max_cut_and_tsp.html
def tsp_hamiltonian(adj_matrix, N, A, edges, partition, penalty_weight=1.0):
    """
    Создаёт гамильтонианы стоимости и смешивания для задачи коммивояжёра (TSP)
    с дополнительным штрафомt.

    Аргументы:
        adj_matrix (np.ndarray): Матрица смежности (расстояния между вершинами).
        N (int): Количество вершин в графе.
        A (float): Параметр штрафа для ограничений TSP.
        edges (list of tuples): Список рёбер (i, j, w) для определения запрещённых переходов.
        partition (list): Список, указывающий партицию каждой вершины (0 или 1).
        penalty_weight (float): Вес штрафа, вдохновлённого Max-Cut.

    Возвращает:
        cost_hamiltonian (SparsePauliOp): Гамильтониан стоимости.
        cost_shift (float): Постоянное смещение гамильтониана стоимости.
        mixing_hamiltonian (SparsePauliOp): Гамильтониан смешивания (сумма операторов X).
        mixing_shift (float): Постоянное смещение гамильтониана смешивания (обычно 0).
    """
    #TODO глянуть формулу внимательнее
    num_qubits = N * N

    cost_pauli_list = []
    cost_coeffs = []
    cost_shift = 0

    def qubit_index(i, p, N):
        return i * N + p

    for i in range(N):
        for j in range(N):
            if i != j:
                w_ij = adj_matrix[i, j]
                for p in range(N):
                    p_next = (p + 1) % N
                    qubit_i = qubit_index(i, p, N)
                    qubit_j = qubit_index(j, p_next, N)

                    cost_shift += w_ij / 4
                    x_p = np.zeros(num_qubits, dtype=bool)
                    z_p = np.zeros(num_qubits, dtype=bool)
                    z_p[qubit_i] = True
                    cost_pauli_list.append(Pauli((z_p, x_p)))
                    cost_coeffs.append(-w_ij / 4)
                    x_p = np.zeros(num_qubits, dtype=bool)
                    z_p = np.zeros(num_qubits, dtype=bool)
                    z_p[qubit_j] = True
                    cost_pauli_list.append(Pauli((z_p, x_p)))
                    cost_coeffs.append(-w_ij / 4)
                    x_p = np.zeros(num_qubits, dtype=bool)
                    z_p = np.zeros(num_qubits, dtype=bool)
                    z_p[qubit_i] = True
                    z_p[qubit_j] = True
                    cost_pauli_list.append(Pauli((z_p, x_p)))
                    cost_coeffs.append(w_ij / 4)

    for p in range(N):
        cost_shift += A
        for i in range(N):
            qubit_i = qubit_index(i, p, N)
            cost_shift += -A
            x_p = np.zeros(num_qubits, dtype=bool)
            z_p = np.zeros(num_qubits, dtype=bool)
            z_p[qubit_i] = True
            cost_pauli_list.append(Pauli((z_p, x_p)))
            cost_coeffs.append(A)

        for i in range(N):
            for i_prime in range(i + 1, N):
                qubit_i = qubit_index(i, p, N)
                qubit_i_prime = qubit_index(i_prime, p, N)
                cost_shift += A / 4
                x_p = np.zeros(num_qubits, dtype=bool)
                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[qubit_i] = True
                cost_pauli_list.append(Pauli((z_p, x_p)))
                cost_coeffs.append(-A / 4)
                x_p = np.zeros(num_qubits, dtype=bool)
                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[qubit_i_prime] = True
                cost_pauli_list.append(Pauli((z_p, x_p)))
                cost_coeffs.append(-A / 4)
                x_p = np.zeros(num_qubits, dtype=bool)
                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[qubit_i] = True
                z_p[qubit_i_prime] = True
                cost_pauli_list.append(Pauli((z_p, x_p)))
                cost_coeffs.append(A / 4)

    for i in range(N):
        cost_shift += A
        for p in range(N):
            qubit_p = qubit_index(i, p, N)
            cost_shift += -A
            x_p = np.zeros(num_qubits, dtype=bool)
            z_p = np.zeros(num_qubits, dtype=bool)
            z_p[qubit_p] = True
            cost_pauli_list.append(Pauli((z_p, x_p)))
            cost_coeffs.append(A)

        for p in range(N):
            for p_prime in range(p + 1, N):
                qubit_p = qubit_index(i, p, N)
                qubit_p_prime = qubit_index(i, p_prime, N)
                cost_shift += A / 4
                x_p = np.zeros(num_qubits, dtype=bool)
                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[qubit_p] = True
                cost_pauli_list.append(Pauli((z_p, x_p)))
                cost_coeffs.append(-A / 4)
                x_p = np.zeros(num_qubits, dtype=bool)
                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[qubit_p_prime] = True
                cost_pauli_list.append(Pauli((z_p, x_p)))
                cost_coeffs.append(-A / 4)
                x_p = np.zeros(num_qubits, dtype=bool)
                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[qubit_p] = True
                z_p[qubit_p_prime] = True
                cost_pauli_list.append(Pauli((z_p, x_p)))
                cost_coeffs.append(A / 4)

    for i, j, w in edges:
        if partition[i] == partition[j]:
            for p in range(N):
                p_next = (p + 1) % N
                qubit_i = qubit_index(i, p, N)
                qubit_j = qubit_index(j, p_next, N)

                cost_shift += (w * penalty_weight) / 4
                x_p = np.zeros(num_qubits, dtype=bool)
                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[qubit_i] = True
                cost_pauli_list.append(Pauli((z_p, x_p)))
                cost_coeffs.append(-(w * penalty_weight) / 4)
                x_p = np.zeros(num_qubits, dtype=bool)
                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[qubit_j] = True
                cost_pauli_list.append(Pauli((z_p, x_p)))
                cost_coeffs.append(-(w * penalty_weight) / 4)
                x_p = np.zeros(num_qubits, dtype=bool)
                z_p = np.zeros(num_qubits, dtype=bool)
                z_p[qubit_i] = True
                z_p[qubit_j] = True
                cost_pauli_list.append(Pauli((z_p, x_p)))
                cost_coeffs.append((w * penalty_weight) / 4)

    cost_hamiltonian = SparsePauliOp(cost_pauli_list, coeffs=cost_coeffs)


    mixing_pauli_list = []
    mixing_coeffs = []
    mixing_shift = 0

    for qubit in range(num_qubits):
        x_p = np.zeros(num_qubits, dtype=bool)
        z_p = np.zeros(num_qubits, dtype=bool)
        x_p[qubit] = True
        mixing_pauli_list.append(Pauli((z_p, x_p)))
        mixing_coeffs.append(1.0)


    mixing_hamiltonian = SparsePauliOp(mixing_pauli_list, coeffs=mixing_coeffs)

    return cost_hamiltonian, cost_shift, mixing_hamiltonian, mixing_shift


def graph_coloring_cost_hemiltonian(n, edges, k):
    """
    Функция для создания гемильтониана стоимости для задачи раскраски графа
    Args:
        n: кол-во вершин
        edges: ребра графа
        k: кол-во цветов для графа

    Returns:
        SparsePauliOp гемильтониан стоимости в форме оператора Паули
    """
    pauli_terms = []
    coeffs = []

    for (i, j) in edges:
        for c in range(k):
            term = ["I"] * (n * k)
            term[i * k + c] = "Z"
            term[j * k + c] = "Z"
            pauli_terms.append("".join(term))
            coeffs.append(1.0)

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))


def graph_coloring_mixing_hemiltonian(n, k):
    """
    Функция для создания гемильтониана смешивания для задачи раскраски графа
    Args:
        n: кол-во вершин
        k: кол-во цветов для графа

    Returns:
        SparsePauliOp гемильтониан смешивания в форме оператора Паули
    """

    pauli_terms = []
    coeffs = []

    for i in range(n):
        for c in range(k):
            term = ["I"] * (n * k)
            term[i * k + c] = "X"
            pauli_terms.append("".join(term))
            coeffs.append(-1.0)

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))


def qaoa_ansatz_from_hemiltonians(cost_hamiltonian_op, mixing_hamiltonian_op, reps):
    """
    Функция, для составления общего гемильтониана задачи для алгоритма QAOA
    Args:
        cost_hamiltonian_op: гемильтониан стоимости для алгоритма
        mixing_hamiltonian_op: гемильтониан смешивания для алгоритма
        reps: кол-во слоев

    Returns:
        QAOAAnsatz - квантовая вариационная схема
    """
    return QAOAAnsatz(cost_operator=cost_hamiltonian_op, mixer_operator=mixing_hamiltonian_op, reps=reps)
