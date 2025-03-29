# quantum_annealing_utils.py
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import time


def create_graph_and_hamiltonian(edges, num_nodes):
    """
    Создаёт гамильтониан для задачи Max-Cut на основе графа.

    Args:
        edges (list of tuples): Список рёбер графа.
        num_nodes (int): Число вершин в графе.

    Returns:
        SparsePauliOp: Гамильтониан задачи.
    """
    hamiltonian_terms = []
    for (i, j) in edges:
        pauli_str = ["I"] * num_nodes
        pauli_str[i] = "Z"
        pauli_str[j] = "Z"
        pauli_str = "".join(pauli_str)
        hamiltonian_terms.append((pauli_str, -0.5))
    hamiltonian = SparsePauliOp.from_list(hamiltonian_terms)
    return hamiltonian


def setup_simulator(with_noise=False, shots=2048):
    """
    Настраивает симулятор и примитив Sampler.

    Args:
        with_noise (bool): Если True, добавляет шум в симуляцию (в данном случае игнорируется).
        shots (int): Число выборок для симуляции (игнорируется, так как используется statevector).

    Returns:
        tuple: (AerSimulator, AerSampler)
    """
    backend = AerSimulator(method="statevector")
    sampler = AerSampler()
    sampler.set_options(backend=backend)
    return backend, sampler


def run_warmup(hamiltonian, sampler, warmup_reps=1, warmup_maxiter=20):
    """
    Выполняет прогрев для нахождения начальных параметров.

    Args:
        hamiltonian (SparsePauliOp): Гамильтониан задачи.
        sampler (AerSampler): Примитив для симуляции.
        warmup_reps (int): Число слоёв для прогрева.
        warmup_maxiter (int): Число итераций для прогрева.

    Returns:
        list: Начальные параметры (числовое значение).
    """
    print("Прогрев...")
    optimizer_warmup = COBYLA(maxiter=warmup_maxiter)
    qaoa_warmup = QAOA(optimizer=optimizer_warmup, reps=warmup_reps, sampler=sampler)
    result_warmup = qaoa_warmup.compute_minimum_eigenvalue(operator=hamiltonian)
    initial_point_base = list(result_warmup.optimal_parameters.values())
    print("Начальные параметры из прогрева:", initial_point_base)
    return initial_point_base


def run_qaoa_experiment(hamiltonian, sampler, initial_point_base, reps_values, maxiter_values):
    """
    Выполняет эксперимент с QAOA, варьируя reps и maxiter.

    Args:
        hamiltonian (SparsePauliOp): Гамильтониан задачи.
        sampler (AerSampler): Примитив для симуляции.
        initial_point_base (list): Начальные параметры из прогрева.
        reps_values (list): Список значений reps для эксперимента.
        maxiter_values (list): Список значений maxiter для эксперимента.

    Returns:
        tuple: (energies, times, result_last, times_history, values_history) - энергии, времена,
               результат последнего запуска, история времени и значений энергии.
    """
    energies = np.zeros((len(reps_values), len(maxiter_values)))
    times = np.zeros((len(reps_values), len(maxiter_values)))
    result_last = None
    times_history = []
    values_history = []

    for i, reps in enumerate(reps_values):
        for j, maxiter in enumerate(maxiter_values):
            print(f"Запуск QAOA с reps={reps}, maxiter={maxiter}...")
            optimizer = COBYLA(maxiter=maxiter)
            initial_point = initial_point_base * reps
            initial_point = initial_point[:2 * reps]

            history = {'times': [], 'values': []}
            start_time = time.time()

            def callback(eval_count, parameters, mean, metadata):
                elapsed_time = time.time() - start_time
                history['times'].append(elapsed_time)
                history['values'].append(mean)

            qaoa = QAOA(optimizer=optimizer, reps=reps, sampler=sampler, initial_point=initial_point, callback=callback)
            result = qaoa.compute_minimum_eigenvalue(operator=hamiltonian)
            end_time = time.time()

            energies[i, j] = result.eigenvalue.real
            times[i, j] = end_time - start_time
            print(f"Энергия: {energies[i, j]}, Время: {times[i, j]:.2f} сек")

            # Добавляем историю для текущего запуска
            times_history.extend(history['times'])
            values_history.extend(history['values'])

            if reps == reps_values[-1] and maxiter == maxiter_values[-1]:
                result_last = result

    return energies, times, result_last, times_history, values_history


def plot_results(reps_values, maxiter_values, energies_ideal, times_ideal, energies_noisy, times_noisy, result_ideal,
                 result_noisy):
    """
    Строит графики и гистограммы для результатов эксперимента.

    Args:
        reps_values (list): Список значений reps.
        maxiter_values (list): Список значений maxiter.
        energies_ideal (ndarray): Энергии для идеальной симуляции.
        times_ideal (ndarray): Времена для идеальной симуляции.
        energies_noisy (ndarray): Энергии для симуляции с шумом (в данном случае игнорируется).
        times_noisy (ndarray): Времена для симуляции с шумом (в данном случае игнорируется).
        result_ideal (SamplingVQEResult): Результат последнего запуска (без шума).
        result_noisy (SamplingVQEResult): Результат последнего запуска (с шумом, в данном случае игнорируется).
        optimal_energy (float): Оптимальная энергия для задачи.
    """

    plt.figure(figsize=(8, 6))
    plot_histogram(result_ideal.eigenstate).show()
    plt.title("Распределение решений (reps=5, maxiter=200)")
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for i, reps in enumerate(reps_values):
        axs[0, 0].plot(maxiter_values, energies_ideal[i, :], marker='o', label=f'reps={reps}')
    axs[0, 0].set_xlabel('Число итераций (maxiter)')
    axs[0, 0].set_ylabel('Энергия')
    axs[0, 0].set_title('Энергия от maxiter')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    for i, reps in enumerate(reps_values):
        axs[0, 1].plot(maxiter_values, times_ideal[i, :], marker='o', label=f'reps={reps}')
    axs[0, 1].set_xlabel('Число итераций (maxiter)')
    axs[0, 1].set_ylabel('Время (сек)')
    axs[0, 1].set_title('Время от maxiter')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    for j, maxiter in enumerate(maxiter_values):
        axs[1, 0].plot(reps_values, energies_ideal[:, j], marker='o', label=f'maxiter={maxiter}')
    axs[1, 0].set_xlabel('Число слоёв (reps)')
    axs[1, 0].set_ylabel('Энергия')
    axs[1, 0].set_title('Энергия от reps')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    for j, maxiter in enumerate(maxiter_values):
        axs[1, 1].plot(reps_values, times_ideal[:, j], marker='o', label=f'maxiter={maxiter}')
    axs[1, 1].set_xlabel('Число слоёв (reps)')
    axs[1, 1].set_ylabel('Время (сек)')
    axs[1, 1].set_title('Время от reps')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()