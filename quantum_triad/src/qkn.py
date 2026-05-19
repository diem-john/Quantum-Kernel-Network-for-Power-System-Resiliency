import pennylane as qml
import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm  # Added for terminal progress tracking


class QuantumKernelNetwork:
    """Quantum Kernel Network for microgrid failure classification."""

    def __init__(self, n_qubits: int, layers: int = 2):
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # 🚨 CRITICAL FIX: Initialize static weights ONCE.
        # This ensures the adjoint circuit matches the forward circuit exactly,
        # preserving the symmetry and mathematical validity of the Gram matrix.
        np.random.seed(42)  # Ensures reproducibility across runs
        self.weights = np.random.uniform(0, 2 * np.pi, (self.layers, self.n_qubits, 3))

    def feature_map(self, x):
        """Hardware-efficient angle embedding with entangling layers."""
        qml.AngleEmbedding(x[:self.n_qubits], wires=range(self.n_qubits), rotation='Y')
        # Use the static weights initialized in __init__
        qml.StronglyEntanglingLayers(weights=self.weights, wires=range(self.n_qubits))

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

        print(f"\n⚛️ Starting Quantum Circuit Simulation: {len(X1)}x{len(X2)} Matrix")

        # 🚨 NEW: tqdm wrapper for terminal visibility
        # This will show a live progress bar in your command prompt/terminal
        for i in tqdm(range(len(X1)), desc="Hilbert Space Mapping", unit="row"):
            x1 = X1[i]
            for j in range(len(X2)):
                x2 = X2[j]

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

        # 🚨 NEW: Moved the Class Balancing logic natively into the backend.
        # This keeps the Streamlit app clean and guarantees the model protects against rare failures.
        self.svm = SVC(kernel='precomputed', probability=True, class_weight='balanced')
        self.svm.fit(self.kernel_matrix, y_train)
        return self.svm