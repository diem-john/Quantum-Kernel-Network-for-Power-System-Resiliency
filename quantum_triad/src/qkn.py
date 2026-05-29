import pennylane as qml
from sklearn.svm import SVC
import torch
import torch.nn as nn
import numpy as np
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

    import torch
    import torch.nn as nn
    import numpy as np

    # Add this method inside your existing QuantumKernelNetwork class in src/qkn.py
    def extract_temporal_quantum_features(self, X_sequential):
        """
        Transforms raw sequential time-series input into sequential quantum features.
        Args:
            X_sequential (np.ndarray): Shape (Samples, Time_Steps, Features)
        Returns:
            torch.Tensor: Shape (Samples, Features, Time_Steps) formatted for PyTorch Conv1D
        """
        n_samples, n_steps, n_features = X_sequential.shape

        # We map the raw features at each time step using your QKN's underlying circuit mapping
        # This acts as an explicit feature-extraction engine step
        quantum_temporal_features = np.zeros((n_samples, n_steps, self.n_qubits))

        for idx in range(n_samples):
            for step in range(n_steps):
                # Extract feature state per time step
                feature_vector = X_sequential[idx, step, :]

                # Use your existing QKN mapping logic (e.g., circuit parameter assignments)
                # Here we extract the expectation values/quantum states from your ansatz
                # Assuming your class has a way to evaluate state projections or feature mappings:
                q_state_features = self._evaluate_ansatz_features(feature_vector)
                quantum_temporal_features[idx, step, :] = q_state_features[:self.n_qubits]

        # Convert to PyTorch Tensor and Permute to match standard Conv1D shape: (Samples, Channels, Sequence)
        feature_tensor = torch.tensor(quantum_temporal_features, dtype=torch.float32)
        return feature_tensor.permute(0, 2, 1)

    def _evaluate_ansatz_features(self, x):
        """
        Fallback dummy/mock logic tracking how your QKN projects state vectors.
        Replace this internal line with your actual backend circuit call
        (e.g., qml.probs, qiskit state vector extraction, or kernel-row slices).
        """
        # Simple projection step tracking simulated hardware outputs
        return np.sin(x) * np.cos(x)


# class QuantumTemporalConvNet(nn.Module):
#     def __init__(self, in_channels, sequence_length):
#         super(QuantumTemporalConvNet, self).__init__()
#
#         # Conv1D scans across the Time Steps axis
#         self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool1d(kernel_size=2) if sequence_length >= 2 else nn.Identity()
#         self.dropout = nn.Dropout(p=0.2)
#
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.relu2 = nn.ReLU()
#
#         # Adaptive pooling collapses timeline sequence into a fixed vector size
#         self.global_pool = nn.AdaptiveAvgPool1d(1)
#
#         self.fc1 = nn.Linear(64, 32)
#         self.fc_relu = nn.ReLU()
#         self.fc2 = nn.Linear(32, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x shape: (Batch, In_Channels, Time_Steps)
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#         x = self.dropout(x)
#
#         x = self.conv2(x)
#         x = self.relu2(x)
#
#         x = self.global_pool(x)  # Shape: (Batch, 64, 1)
#         x = x.view(x.size(0), -1)  # Flatten to (Batch, 64)
#
#         x = self.fc1(x)
#         x = self.fc_relu(x)
#         x = self.fc2(x)
#         return x # self.sigmoid(x)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        # Math: Hidden projection scaled down by a factor of 2 to compress attention
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output):
        attention_scores = self.attention_layer(lstm_output)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights


class QuantumTemporalConvNet(nn.Module):
    def __init__(self, in_channels=3, sequence_length=4):
        super(QuantumTemporalConvNet, self).__init__()

        # --- STAGE 1: Feature Extraction ---
        # Math: Expanding 3 channels to 8 channels (approx factor of 2.5)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu1 = nn.ReLU()

        # --- STAGE 2: Sequential BiLSTM ---
        # Math: Input=8, Hidden=8. Output will be 16 due to bidirectionality.
        # This keeps the temporal parameter count under ~1,500.
        self.lstm_hidden = 8
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # --- STAGE 3: Temporal Attention ---
        # Hidden size is 16 (8 * 2)
        self.attention = TemporalAttention(hidden_size=self.lstm_hidden * 2)

        # --- STAGE 4: Decision Head ---
        # Math: Geometric step down from 16 -> 8 -> 1
        self.fc1 = nn.Linear(self.lstm_hidden * 2, 8)
        self.fc_relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        # x shape: (240, 3, 4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Permute for LSTM: (Batch, Time_Steps, Channels) -> (240, 4, 8)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        # Attention compresses the time axis: (240, 16)
        context_vector, _ = self.attention(lstm_out)

        # Final classification logits
        x = self.fc1(context_vector)
        x = self.fc_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x  # Reminder: Returning RAW LOGITS for BCEWithLogitsLoss