from qiskit.quantum_info import Pauli, SparsePauliOp
import numpy as np

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

if __name__ == "__main__":
    import numpy as np
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler

    N = 3
    adj_matrix = np.array([
        [0, 48, 91],
        [48, 0, 63],
        [91, 63, 0]
    ])
    A = 100

    edges = [(i, j, adj_matrix[i, j]) for i in range(N) for j in range(i + 1, N)]
    partition = [0, 1, 0]
    penalty_weight = 10.0

    cost_hamiltonian, cost_shift, mixing_hamiltonian, mixing_shift = tsp_hamiltonian(
        adj_matrix, N, A, edges, partition, penalty_weight
    )

    sampler = Sampler()
    optimizer = COBYLA(maxiter=100)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1, mixer=mixing_hamiltonian)

    result = qaoa.compute_minimum_eigenvalue(operator=cost_hamiltonian)

    optimal_state = result.best_measurement['bitstring']
    print("Оптимальное состояние (битовая строка):", optimal_state)

    x = np.array([int(bit) for bit in optimal_state])
    z = tsp.interpret(x)
    print("Решение TSP (порядок посещения городов):", z)
    print("Целевое значение решения (общее расстояние):", tsp.tsp_value(z, adj_matrix))

    print("\nГамильтониан стоимости (H_C):")
    print(cost_hamiltonian)
    print("\nСмещение стоимости:", cost_shift)
    print("\nГамильтониан смешивания (H_M):")
    print(mixing_hamiltonian)
    print("\nСмещение смешивания:", mixing_shift)