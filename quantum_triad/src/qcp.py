import numpy as np


class QuantumConformalPredictor:
    """Split Conformal Prediction wrapper for the Quantum Kernel Network."""

    def __init__(self, qkn_model, alpha=0.1):
        self.qkn = qkn_model
        self.alpha = alpha  # 1 - alpha = target coverage (e.g., 90%)
        self.q_hat = None

    def calibrate(self, X_cal, y_cal, X_train):
        """Calculates the non-conformity scores and the quantile (q_hat)."""
        K_cal = self.qkn.compute_kernel_matrix(X_cal, X_train)
        cal_probs = self.qkn.svm.predict_proba(K_cal)

        n = len(y_cal)
        scores = np.array([1 - cal_probs[i, y_cal[i]] for i in range(n)])

        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        if q_level > 1.0: q_level = 1.0
        self.q_hat = np.quantile(scores, q_level, method='higher')
        return self.q_hat

    def predict_sets(self, X_test, X_train, classes=[0, 1]):
        """Outputs a prediction set for each test instance based on q_hat."""
        K_test = self.qkn.compute_kernel_matrix(X_test, X_train)
        test_probs = self.qkn.svm.predict_proba(K_test)

        prediction_sets = []
        for probs in test_probs:
            valid_classes = [c for c, p in zip(classes, probs) if p >= (1 - self.q_hat)]
            prediction_sets.append(valid_classes)

        return prediction_sets