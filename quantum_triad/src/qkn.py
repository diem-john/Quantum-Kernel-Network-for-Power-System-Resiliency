import pennylane as qml
import numpy as np
from sklearn.svm import SVC


class QuantumKernelNetwork:
    """Quantum Kernel Network for microgrid failure classification."""

    def __init__(self, n_qubits: int, layers: int = 2):
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

    def feature_map(self, x):
        """Hardware-efficient angle embedding with entangling layers."""
        qml.AngleEmbedding(x[:self.n_qubits], wires=range(self.n_qubits), rotation='Y')
        for _ in range(self.layers):
            qml.StronglyEntanglingLayers(weights=np.random.uniform(0, 2 * np.pi, (1, self.n_qubits, 3)),
                                         wires=range(self.n_qubits))

    def kernel_circuit(self, x1, x2):
        """Computes the transition amplitude (fidelity) between two encoded states."""
        self.feature_map(x1)
        qml.adjoint(self.feature_map)(x2)
        return qml.probs(wires=range(self.n_qubits))

    def qnode_kernel(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x1, x2):
            return self.kernel_circuit(x1, x2)

        return circuit

    def compute_kernel_matrix(self, X1, X2, progress_callback=None):
        """Generates the Gram matrix for the quantum kernel with progress tracking."""
        q_kernel = self.qnode_kernel()
        matrix = np.zeros((len(X1), len(X2)))

        total_steps = len(X1) * len(X2)
        step = 0
        throttle_rate = max(1, total_steps // 100)  # Update UI roughly every 1%

        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                # Probability of measuring the zero state |0...0>
                matrix[i, j] = q_kernel(x1, x2)[0]

                step += 1
                if progress_callback and step % throttle_rate == 0:
                    progress_callback(step / total_steps)

        # Ensure it hits 100% at the very end
        if progress_callback:
            progress_callback(1.0)

        return matrix

    def train_qsvm(self, X_train, y_train, progress_callback=None):
        """Trains a classical SVM using the quantum kernel matrix."""
        # Pass the callback down to the matrix generator
        self.kernel_matrix = self.compute_kernel_matrix(X_train, X_train, progress_callback)

        self.svm = SVC(kernel='precomputed', probability=True)
        self.svm.fit(self.kernel_matrix, y_train)
        return self.svm