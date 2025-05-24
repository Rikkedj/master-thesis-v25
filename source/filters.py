import numpy as np

class PostPredictionController:
    def __init__(self, gain_hand_open=1.0, gain_hand_close=1.0, gain_pronation=1.0, gain_supination=1.0, thr_angle_mf1=45, thr_angle_mf2=45, deadband_radius=0.0): # TODO: Consider having gain for each motion class, not only motor function
        """
        Post-prediction controller to scale the output of the prediction model.

        Parameters: TODO: Should change these to be more modular / dynamic. Works hardcoded for now. 
        -------------
        gain_hand_open: float 
            Gain for motion class hand open.
        gain_hand_close: float
            Gain for motion class hand close.
        gain_pronation: float
            Gain for motion class pronation.
        gain_supination: float
            Gain for motion class supination.
        thr_angle_mf1: float
            Threshold angle for motor function 1, i.e. hand open/close.
        thr_angle_mf2: float
            Threshold angle for motor function 2, i.e. pronation/supination.
        deadband_radius: float
            Deadband radius around origin for the prediction model. Every prediction within this range will be set to 0.
        """
        self.gain_hand_open = gain_hand_open
        self.gain_hand_close = gain_hand_close
        self.gain_pronation = gain_pronation
        self.gain_supination = gain_supination
        self.thr_angle_mf1 = thr_angle_mf1
        self.thr_angle_mf2 = thr_angle_mf2
        self.deadband_radius = deadband_radius

    # def update_config(self, gain_hand_open=None, gain_hand_close=None, gain_pronation=None, gain_supination=None, deadband_radius=None):
    #     """
    #     Set the configuration for the prosthetic device. NOTE! Could add checks here as well to ensure the values are within a certain range, or if they are valid values.
    #     Parameters:
    #     ----------
    #     gain_hand_open: float TODO: Should change these to be more modular / dynamic. Works hardcoded for now. 
    #         Gain for motion class hand open.
    #     gain_hand_close: float
    #         Gain for motion class hand close.
    #     gain_pronation: float
    #         Gain for motion class pronation.
    #     gain_supination: float
    #         Gain for motion class supination.
    #     deadband_radius: float
    #         Deadband radius around origin for the prediction model. Every prediction within this range will be set to 0.
    #     """
        # if gain_hand_open:
        #     self.gain_hand_open = gain_hand_open
        # if gain_hand_close:
        #     self.gain_hand_close = gain_hand_close
        # if gain_pronation:
        #     self.gain_pronation = gain_pronation
        # if gain_supination:
        #     self.gain_supination = gain_supination
        # if deadband_radius:
        #     self.deadband_radius = deadband_radius
        
    def update_config(self, **kwargs):
        """
        Update the configuration parameters for the post-prediction controller.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{key}' is not a valid configuration parameter.")

    def apply_gains(self, prediction):
        """Apply gain to each prediction value."""
        gain_mf1 = self.gain_hand_open if prediction[0] > 0 else self.gain_hand_close # Gain for motor function 1
        gain_mf2 = self.gain_pronation if prediction[1] > 0 else self.gain_supination
        return np.array([
            prediction[0] * gain_mf1,
            prediction[1] * gain_mf2
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
        prediction: list 
            The input prediction signal.

        Returns:
        -------------
        list: The updated prediction signal.
        """
        # Apply gain
        prediction = self.apply_gains(prediction)
        # Apply deadband
        prediction = self.apply_deadband(prediction)

        return prediction

class FlutterRejectionFilter:
    def __init__(self, tanh_gain=0.5, dt=0.01, integrator_enabled=False, gain=1.0, k=0.0):
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
        k (float):

        """
        self.tanh_gain = tanh_gain
        self.dt = dt
        self.integrator_enabled = integrator_enabled
        self.state = None # Initialize the state for the integrator
        self.gain = gain
        self.k = k

    def update_settings(self, kwargs):
        """
        Update the filter settings dynamically.

        Parameters:
        -------------
        kwargs: dict
            Dictionary of parameters to update. Valid keys are 'tanh_gain', 'dt', 'integrator_enabled', 'gain', and 'k'.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{key}' is not a valid filter parameter.")
            
    ## Updated version of filter
    def reset_integrator(self):
        """ Reset the filter state to zero. """
        if self.state is not None:
            self.state[:] = 0.0

    def filter(self, x):
        x = np.asarray(x)
        if self.state is None:
            self.state = np.zeros_like(x)

        nonlinear_output = (np.abs(x)-self.k) * np.tanh(self.tanh_gain * x)
        self.state += nonlinear_output * self.dt
        
        if self.integrator_enabled:
            return self.gain*self.state
        else:
            return self.gain*nonlinear_output
    
