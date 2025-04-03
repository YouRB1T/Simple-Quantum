import numpy as np
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.circuit.library import QAOAAnsatz

def max_cut_hemiltonian(ages, n):
    pauli_list = []
    coeffs = []
    shift = 0

    for i, j, w in ages:
        x_p = np.zeros(n, dtype=bool)
        z_p = np.zeros(n, dtype=bool)
        z_p[i] = True
        z_p[j] = True
        pauli_list.append(Pauli((z_p, x_p)))
        coeffs.append(-0.5)
        shift += 0.5

    for i, j, w in ages:
        x_p = np.zeros(n, dtype=bool)
        z_p = np.zeros(n, dtype=bool)
        z_p[i] = True
        z_p[j] = True
        pauli_list.append(Pauli((z_p, x_p)))
        coeffs.append(1.0)

    shift += n

    return SparsePauliOp(pauli_list, coeffs=coeffs), shift




def tsp_cost_hemiltonian(n, distance_matrix):
    """
    Функция нахождения гемильтониана стоимости для задачи комивояджера
    Args:
        n: кол-во вершин в графе
        distance_matrix: матрица дистанций графа

    Returns:
        SparsePauliOp гемильтониан стоимости в форме оператора Паули
    """
    pauli_terms = []
    coeffs = []

    for i in range(n):
        for j in range(i + 1, n):
            term = ["I"] * (n * n)
            term[i * n + j] = "Z"
            term[j * n + i] = "Z"
            pauli_terms.append("".join(term))
            coeffs.append(-distance_matrix[i][j])

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))


def tsp_mixing_hemiltonian(n):
    """
    Функция нахождения гемильтониана смешивания для задачи комивояджера
    Args:
        n: кол-во вершин в графе

    Returns:
        SparsePauliOp гемильтониан смешивания в форме оператора Паули
    """
    pauli_terms = []
    coeffs = []

    for i in range(n):
        for j in range(n):
            if i != j:
                term = ["I"] * (n * n)
                term[i * n + j] = "X"
                pauli_terms.append("".join(term))
                coeffs.append(-1.0)

    return SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))


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
