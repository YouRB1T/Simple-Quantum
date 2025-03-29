from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

qr = QuantumRegister(2, 'q')
cr = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qr, cr)

circuit.h(qr[0])
circuit.cx(qr[0], qr[1])

circuit.measure(qr, cr)

simulator = AerSimulator()

job = simulator.run(circuit, shots=1024)
result = job.result()

counts = result.get_counts(circuit)
print("Результаты измерений:", counts)

plot_histogram(counts)
plt.show(block=True)