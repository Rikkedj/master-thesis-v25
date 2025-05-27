from libemg.environments.controllers import RegressorController, ClassifierController
from libemg.prosthesis import Prosthesis

import numpy as np
from queue import Queue
import threading
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

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

def run_controller(controller, prediction_queue, stop_event):
    while not stop_event.is_set():
        pred = controller.get_data(["predictions"])
        if pred is not None:
            print("Raw prediction: ", pred)
            filtered_pred = flutter_filter.apply(pred)
            print("Filtered prediction: ", filtered_pred)
            prediction_queue.put(filtered_pred)

def send_command_to_prosthesis(prosthesis, mf_selector, prediction_queue, stop_event):
    """
    Send command to the prosthesis.
    """
    while not stop_event.is_set():
        filtered_pred = prediction_queue.get()
        motor_setpoint = mf_selector.get_motor_setpoints(filtered_pred)
        prosthesis.send_command(motor_setpoint)
        time.sleep(0.1)

def update_plot(frame, plot_data, line):
    line.set_ydata(plot_data)
    line.set_xdata(range(len(plot_data)))
    return line,

if __name__ == "__main__":
    UDP_IP = '127.0.0.1'
    UDP_PORT = 5005
    PLOT_WINDOW_SIZE = 100

    prost = Prosthesis()
    
    prost.connect(port="COM9", baudrate=115200, timeout=1)  # Change this to the correct port and baudrate for your prosthesis
    prost.send_command([0,255,0,0])
    #prost.send_command([0,150,0,0])
    time.sleep(1)  # Wait for the prosthesis to initialize
    prost.send_command([0,0,0,0])


    # regression_selected = True
    # if regression_selected:
    #     controller = RegressorController(ip=UDP_IP, port=UDP_PORT)
    # else:
    #     controller = ClassifierController(output_format='predictions', ip=UDP_IP, port=UDP_PORT)
        
    # pred_queue = Queue()

    # flutter_filter = FlutterRejectionFilter(k=5.0, integrator_enabled=True, dt=0.1) # This is the filter that is used to filter the predictions from the model. It is a nonlinear filter that is used to remove noise from the predictions. It is not used in the current version of the code, but it is here for future use.            
    # mf_selector = MotorFunctionSelector() # This is the motor function selector that is used to select the motor functions that are used for the prosthesis. It is not used in the current version of the code, but it is here for future use.
    # prosthesis = Prosthesis()
    # prosthesis.connect(port="COM9", baudrate=115200, timeout=1) # Change this to the correct port and baudrate for your prosthesis
    
    # stop_event = threading.Event()
    # pred_thread = threading.Thread(target=run_controller, args=(controller, pred_queue, stop_event))
    # pred_thread.start()

    # command_thread = threading.Thread(target=send_command_to_prosthesis, args=(prosthesis, mf_selector, pred_queue, stop_event))
    # command_thread.start()
    

    # ######## PLOT PREDICITONS LIVE ########
    # plot_data = deque([0.0] * PLOT_WINDOW_SIZE, maxlen=PLOT_WINDOW_SIZE)
    # # Setup live plot
    # fig, ax = plt.subplots()
    # line, = ax.plot(range(PLOT_WINDOW_SIZE), list(plot_data))
    # ax.set_ylim(-1, 1)  # Adjust based on expected prediction range
    # ax.set_title("Live Filtered Predictions")
    # ax.set_xlabel("Time step")
    # ax.set_ylabel("Prediction")

    # def polling_pred_queue():
    #     """Fetch from pred_queue to update plot_data continuously."""
    #     while not stop_event.is_set():
    #         if not pred_queue.empty():
    #             val = pred_queue.get()
    #             # Support both scalar and list predictions
    #             if isinstance(val, (list, tuple)):
    #                 val = val[0]
    #             plot_data.append(val)
    #         else:
    #             time.sleep(0.01)

    # polling_thread = threading.Thread(target=polling_pred_queue)
    # polling_thread.start()

    # ani = animation.FuncAnimation(fig, update_plot, fargs=(plot_data, line), interval=100)

    # try:
    #     plt.show()  # This blocks until the plot window is closed
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     print("Stopping the controller...")
    #     stop_event.set()
    #     pred_thread.join()
    #     command_thread.join()
    #     prosthesis.disconnect()
    #     polling_thread.join()
    #     print("Controller stopped.")
