import numpy as np
class kalman_filter():
    """Here is the kalman filter
    The kalman filter will recieve the A B matrix from the dynamic system
    Args:
        A: state transition matrix
        B: input transition matrix (is None if model has no inputs)
        C: Observation matrix
        x0: Initial state vector
        P0: Initial state covariance matrix
        Q0: (Initial) state noise covariance
        R0: (Initial) observation noise covariance
    """
    def __init__(self,A,B,C,x0,P0,Q0,R0) -> None:
        self.F = A.copy()
        self.B = B.copy() if B is not None else None
        self.H = C.copy()
        self.x = x0.astype(float)  # Convert to float
        self.P = P0.copy()
        self.Q = Q0.copy()
        self.R = R0.copy()
        pass
    #! Here is the predict step of the kalman filter
    def predict(self,u=None):
        """Predict the next state of the system
        Args:
            u: Input vector
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if u is not None:
            self.x += self.B @ u
        pass
    #! Here is the update step of the kalman filter
    def update(self,measurement):
        """Update the state of the system
        Args:
            measurement: Observation vector
        """
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(self.F.shape[0]) - K @ self.H) @ self.P
        pass
    #! Here is the getter of the estimate
    @property
    def get_estimate(self):
        return self.x
    #! Here is the getter of the covariance
    @property
    def get_covariance(self):
        return self.P