import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from threading import Thread
from libemg.environments.controllers import Controller, RegressorController, ClassifierController
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue
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

def get_predictions(ip, port, pred_queue):
    """
    Get predictions from the queue and apply the filter to them.

    Parameters:
    -------------
    pred_queue: Queue
        The queue containing raw predictions.
    filtered_queue: Queue
        The queue to store filtered predictions.
    """
    controller = RegressorController(ip=ip, port=port)
    while True:
        pred = controller.get_data(['predictions'])
        print(pred)
        if pred is not None:
            pred_queue.put(pred)
        else:
            print("No prediction data available.")
            time.sleep(0.1)

def get_filtered_predictions(pred_queue, filtered_queue, filter):
    """
    Get filtered predictions from the raw predictions queue.

    Parameters:
    -------------
    pred_queue: Queue
        The queue containing raw predictions.
    filtered_queue: Queue
        The queue to store filtered predictions.
    filter: FlutterRejectionFilter
        The filter to apply to the predictions.
    """
    while True:
        if not pred_queue.empty():
            pred = pred_queue.get()
            filtered_pred = filter.apply(pred)
            filtered_queue.put(filtered_pred)
        else:
            time.sleep(0.1)

if __name__ == "__main__":
    UDP_IP = "127.0.0.1"  # Localhost (change if sending from another machine)
    UDP_PORT = 5005

    filter = FlutterRejectionFilter(k=1, integrator_enabled=True)
    #controller = RegressorController(ip=UDP_IP, port=UDP_PORT)
    
    pred_queue = Queue()
    filtered_queue = Queue()

    pred_thread = Thread(target=get_predictions, args=(UDP_IP,UDP_PORT, pred_queue, ), daemon=True)
    filtered_thread = Thread(target=get_filtered_predictions, args=(pred_queue, filtered_queue, filter), daemon=True)
    pred_thread.start()
    filtered_thread.start()
    pred_thread.join()
    filtered_thread.join()

    window_size = 50
    pred_buf = np.zeros(window_size,2)
    filtered_buf = np.zeros(window_size,2)

    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    raw_line1, = ax.plot([], [], label='Raw Predictions - Channel 1', color='blue', alpha=0.7)
    raw_line2, = ax.plot([], [], label='Raw Predictions - Channel 2', color='cyan', alpha=0.7)
    filtered_line1, = ax.plot([], [], label='Filtered Predictions - Channel 1', color='green', alpha=0.7)
    filtered_line2, = ax.plot([], [], label='Filtered Predictions - Channel 2', color='lime', alpha=0.7)

    ax.set_title("Real-Time Raw vs Filtered Predictions (Two Channels)")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Prediction Values")
    ax.legend()
    ax.grid(True)

    while True:
        # Check if new data is available in the queues
        if not pred_queue.empty() and not filtered_queue.empty():
            raw_pred = pred_queue.get()
            filtered_pred = filtered_queue.get()

            # Update the buffers with new data
            raw_buffer = np.roll(raw_buffer, -1, axis=0)
            filtered_buffer = np.roll(filtered_buffer, -1, axis=0)
            raw_buffer[-1, :] = raw_pred
            filtered_buffer[-1, :] = filtered_pred

            # Update the plot data
            raw_line1.set_data(range(window_size), raw_buffer[:, 0])
            raw_line2.set_data(range(window_size), raw_buffer[:, 1])
            filtered_line1.set_data(range(window_size), filtered_buffer[:, 0])
            filtered_line2.set_data(range(window_size), filtered_buffer[:, 1])

            # Adjust plot limits
            ax.set_xlim(0, window_size)
            ax.set_ylim(
                min(raw_buffer.min(), filtered_buffer.min()) - 0.1,
                max(raw_buffer.max(), filtered_buffer.max()) + 0.1
            )

            plt.pause(0.01)  # Pause to update the plot
