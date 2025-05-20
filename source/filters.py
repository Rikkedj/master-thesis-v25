import numpy as np

class PostPredictionController:
    def __init__(self, gain_mf1=1.0, gain_mf2=1.0, thr_angle_mf1=45, thr_angle_mf2=45, deadband_radius=0.0):
        """
        Post-prediction controller to scale the output of the prediction model.

        Parameters:
        -------------
        gain (float): Gain applied to the output to scale the response.
        """
        self.gain_mf1 = gain_mf1
        self.gain_mf2 = gain_mf2
        self.thr_angle_mf1 = thr_angle_mf1
        self.thr_angle_mf2 = thr_angle_mf2
        self.deadband_radius = deadband_radius

    def update_config(self, gain_mf1=None, gain_mf2=None, thr_angle_mf1=None, thr_angle_mf2=None, deadband_radius=None):
        """
        Set the configuration for the prosthetic device. NOTE! Could add checks here as well to ensure the values are within a certain range, or if they are valid values.
        Parameters:
        ----------
            gains: List of gains for each motor function (list of floats).
            thr_angle_mf1: Threshold angle for motor function 1 (float).
            thr_angle_mf2: Threshold angle for motor function 2 (float).
            deadband: Deadband value (float).
        """
        if gain_mf1:
            self.gain_mf1 = gain_mf1
        if gain_mf2:
            self.gain_mf2 = gain_mf2
        #if thr_angle_mf1:
        #    self.thr_angle_mf1 = thr_angle_mf1
        #if thr_angle_mf2:
        #    self.thr_angle_mf2 = thr_angle_mf2
        if deadband_radius:
            self.deadband_radius = deadband_radius


    def apply_gains(self, prediction):
        """Apply gain to each prediction value."""
        return np.array([
            prediction[0] * self.gain_mf1,
            prediction[1] * self.gain_mf2
        ])

    def apply_deadband(self, prediction):
        """Zero out values within the deadband radius."""
        return np.array([
            0 if abs(prediction[0]) < self.deadband_radius else prediction[0],
            0 if abs(prediction[1]) < self.deadband_radius else prediction[1]
        ])

    def update_prediction(self, prediction):
        """
        Update the prediction by applying gains and deadband.

        Parameters:
        -------------
        prediction (list): The input prediction signal.

        Returns:
        list: The updated prediction signal.
        """
        # Apply gain
        prediction = self.apply_gains(prediction)
        
        # Apply deadband
        prediction = self.apply_deadband(prediction)

        return prediction

class FlutterRejectionFilter:
    def __init__(self, tanh_gain=0.5, dt=0.01, integrator_enabled=False, gain=1.0):
        """
        Nonlinear flutter rejection filter with gain, deadband and optional integrator

        Parameters:
        -------------
        tanh_gain (float): 
            The scaling factor for the tanh function. A lower tanh_gain makes the nonlinear filter more linear/gradual.
        dt (float): 
            The time constant for the integrator (affects smoothness).
        integrator_enabled (bool):
            If True, the filter will include an integrator to smooth the output.
        gain (float): 
            Gain applied to the output to scale the response.
        """
        self.tanh_gain = tanh_gain
        self.dt = dt
        self.integrator_enabled = integrator_enabled
        self.state = None # Initialize the state for the integrator
        self.gain = gain


    ## Updated version of filter
    def reset_integrator(self):
        """ Reset the filter state to zero. """
        if self.state is not None:
            self.state[:] = 0.0

    def update(self, x):
        x = np.asarray(x)
        if self.state is None:
            self.state = np.zeros_like(x)

        nonlinear_output = np.abs(x) * np.tanh(self.tanh_gain * x)
        self.state += nonlinear_output * self.dt
        
        if self.integrator_enabled:
            return self.gain*self.state
        else:
            return self.gain*nonlinear_output
    
    ## Old version of filter, did not work very well
    
    # def apply_deadband(self, x):
    #     """
    #     Apply deadband to the signal (set it to zero if within the deadband range).

    #     Parameters:
    #     x (float): The input signal.

    #     Returns:
    #     float: The signal after deadband is applied.
    #     """
    #     deadband_mask = np.abs(x) < self.deadband_radius
    #     x[deadband_mask] = 0.

    #     # Tried different approaches. 
    #     #if np.linalg.norm(x) < self.deadband_radius:
    #     #    x[:] = 0.
    #     # if abs(x) < self.deadband_radius:
    #     #     return 0.0
    #     return x
    # def update_deadband_radius(self, new_radius):
    #     self.deadband_radius = new_radius
    # def update_gain(self, new_gain):
    #     self.gain = new_gain
    # def update_k(self, new_k):
    #     self.k = new_k
    # def update_dt(self, new_dt):
    #     self.dt = new_dt
    # def enable_integrator(self):
    #     self.integrator_enabled = True


    # def apply_filter(self, x, k):
    #     """
    #     Nonlinear filter function using tanh scaling.
    #     """
    #     return np.abs(x) * np.tanh(k * x)
    
    # def reset_integrator(self):
    #     """
    #     Reset the integrator state to zero.
    #     """
    #     self.integrator_output = None


    # def apply(self, x):
    #     """
    #     Apply the flutter rejection filter, which includes a deadband, 
    #     nonlinear tanh scaling, and optional integration.

    #     Parameters:
    #     -------------
    #     x: array of float
    #         The input signal.

    #     Returns:
    #         float: The filtered output signal.
    #     """
    #     # Apply the deadband first
    #     x = np.array(x)
    #     #x_deadbanded = self.apply_deadband(x)

    #     # Nonlinear tanh function for flutter rejection
    #     y = np.abs(x) * np.tanh(self.k * x)
    #     y_gained = y * self.gain

    #     # Apply the integrator
    #     if self.integrator_enabled:
    #         if self.integrated_output is None:
    #             self.integrated_output = y_gained
    #         else: 
    #             self.integrated_output += y_gained * self.dt 
                 
    #         return self.integrated_output  # Return the integrated output
        
    #     else:
    #         return y_gained       # If integrator is not enabled, just use the current output
        