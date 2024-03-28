import numpy as np
class kalman_filter():
    """Here is the kalman filter
    The kalman filter will recieve the A B matrix from the dynamic system
    Args:
        F: state transition matrix
        B: input transition matrix (is None if model has no inputs)
        C: Observation matrix
        x0: Initial state vector
        P0: Initial state covariance matrix
        Q0: (Initial) state noise covariance
        R0: (Initial) observation noise covariance
    """
    def __init__(self,F,B,C,x0,P0,Q0,R0) -> None:
        self.F = F
        self.B = B if B is not None else None
        self.H = C
        self.x = x0  # Convert to float
        self.P = P0
        self.Q = Q0
        self.R = R0
        pass
    #! Here is the predict step of the kalman filter
    def predict(self,u=None,F=None,B=None):
        """Predict the next state of the system
        if F and B are given, use them to update the system, ekf predict
        if F and B are not given, use the original system, kf predict
        Args:
            u: Input vector
        """
        if F is not None and B is not None:
            self.F = F
            self.B = B
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if u is not None:
            self.x += self.B @ u      
        pass
    #! Here is the update step of the kalman filter,
    #now we did not consider nonlinear measurement
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
    @property
    def get_F_matrix(self):
        return self.F
    @property
    def get_B_matrix(self):
        return self.B
    @property
    def get_predict(self):
        return self.x
    @property
    def get_initial_state(self):
        return self.x
    