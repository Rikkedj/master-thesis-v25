import numpy as np


class FlutterRejectionFilter:
    def __init__(self, k=1.0, deadband_radius=0.01, gain=1.0, dt=0.01, integrator_enabled=False):
        """
        Nonlinear flutter rejection filter with gain, deadband and optional integrator

        Parameters:
        -------------
        k (float): 
            The scaling factor for the tanh function.
        deadband_radius (float): 
            The radius around zero where the signal will be set to zero.
        gain (float): 
            Gain applied to the output to scale the response.
        dt (float): 
            The time constant for the integrator (affects smoothness).
        integrator_enabled (bool):
            If True, the filter will include an integrator to smooth the output.
        """
        self.k = k
        self.deadband_radius = deadband_radius
        self.gain = gain
        self.dt = dt
        self.integrator_enabled = integrator_enabled
        self.integrated_output = None # Store last integrated output

    def apply_deadband(self, x):
        """
        Apply deadband to the signal (set it to zero if within the deadband range).

        Parameters:
        x (float): The input signal.

        Returns:
        float: The signal after deadband is applied.
        """
        deadband_mask = np.abs(x) < self.deadband_radius
        x[deadband_mask] = 0.

        # Tried different approaches. 
        #if np.linalg.norm(x) < self.deadband_radius:
        #    x[:] = 0.
        # if abs(x) < self.deadband_radius:
        #     return 0.0
        return x
    def update_deadband_radius(self, new_radius):
        self.deadband_radius = new_radius
    def update_gain(self, new_gain):
        self.gain = new_gain
    def update_k(self, new_k):
        self.k = new_k
    def update_dt(self, new_dt):
        self.dt = new_dt
    def enable_integrator(self):
        self.integrator_enabled = True


    def apply_filter(self, x, k):
        """
        Nonlinear filter function using tanh scaling.
        """
        return np.abs(x) * np.tanh(k * x)
    
    def reset_integrator(self):
        """
        Reset the integrator state to zero.
        """
        self.integrator_output = None


    def apply(self, x):
        """
        Apply the flutter rejection filter, which includes a deadband, 
        nonlinear tanh scaling, and optional integration.

        Parameters:
        -------------
        x: array of float
            The input signal.

        Returns:
            float: The filtered output signal.
        """
        # Apply the deadband first
        x = np.array(x)
        #x_deadbanded = self.apply_deadband(x)

        # Nonlinear tanh function for flutter rejection
        y = np.abs(x) * np.tanh(self.k * x)
        y_gained = y * self.gain

        # Apply the integrator
        if self.integrator_enabled:
            if self.integrated_output is None:
                self.integrated_output = y_gained
            else: 
                self.integrated_output += y_gained * self.dt 
                 
            return self.integrated_output  # Return the integrated output
        
        else:
            return y_gained       # If integrator is not enabled, just use the current output
        