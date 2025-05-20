import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from threading import Thread, Event
from libemg.environments.controllers import Controller, RegressorController, ClassifierController
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue


def get_predictions(controller, pred_queue, stop_event):
    """
    Get predictions from the queue and apply the filter to them.

    Parameters:
    -------------
    pred_queue: Queue
        The queue containing raw predictions.
    filtered_queue: Queue
        The queue to store filtered predictions.
    """
    while not stop_event.is_set():
        pred = controller.get_data(['predictions'])
        #print(pred)
        if pred is not None:
            pred_queue.put(pred)
        #else:
         #   print("No prediction data available.")
        
        time.sleep(0.01)

def get_filtered_predictions(pred_queue, filtered_queue, filter, stop_event):
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
    while not stop_event.is_set():
        if not pred_queue.empty():
            pred = pred_queue.get()
            print("Raw prediction: ", pred)
            filtered_pred = filter.update(pred)
            print("Filtered prediction: ", filtered_pred)
            filtered_queue.put(filtered_pred)
        
        time.sleep(0.01)

def plot_filter_comparison(tanh_gains=[0.5, 2.0, 15.0], deadband_radius=0.2):
    x = np.linspace(-2, 2, 500)

    # Linear function
    y_linear = x

    # Nonlinear flutter rejection filter
    y_nonlinear = [np.abs(x) * np.tanh(gain * x) for gain in tanh_gains]

    # Deadband filter
    y_deadband = np.where(np.abs(x) < deadband_radius, 0, x)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x, y_linear, label='Linear (y = x)', linestyle='--', color='red')
    for i, gain in enumerate(tanh_gains):
        if i == 0: plt.plot(x, y_nonlinear[i], label=f'Nonlinear Flutter Filter (gain={gain})', color='black')
        else: plt.plot(x, y_nonlinear[i], label=f'Nonlinear Flutter Filter (gain={gain})', linestyle='--', color='black')
    
    plt.plot(x, y_deadband, label=f'Deadband Filter (radius={deadband_radius})', color='palegreen')

    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Comparison of Linear, Nonlinear, and Deadband Filters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


class FlutterRejectionFilter:
    def __init__(self, tanh_gain=0.5, dt=0.01, integrator_enabled=False, gain=1.0):
        """
        Nonlinear flutter rejection filter with gain, deadband and optional integrator

        Parameters:
        -------------
        tanh_gain (float): 
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
        

if __name__ == "__main__":
    UDP_IP = "127.0.0.1"  # Localhost (change if sending from another machine)
    UDP_PORT = 5005

    stop_event = Event()

    controller = RegressorController(ip=UDP_IP, port=UDP_PORT)
    
    dt = 0.01
    gain = 1.0
    tanh_gain = 0.5
    filter = FlutterRejectionFilter(gain=gain, dt=dt, tanh_gain=tanh_gain, integrator_enabled=True)
    #controller = RegressorController(ip=UDP_IP, port=UDP_PORT)
    
    pred_queue = Queue()
    filtered_queue = Queue()

    pred_thread = Thread(target=get_predictions, args=(controller, pred_queue, stop_event))
    filtered_thread = Thread(target=get_filtered_predictions, args=(pred_queue, filtered_queue, filter, stop_event))
    pred_thread.start()
    filtered_thread.start()
    

    tale_length = 20 # Number of history points to show
    window_size = tale_length 
    window_size = 50 # Total number of points to show in the plot along time

    pred_buf = np.zeros((window_size,2))
    filtered_buf = np.zeros((window_size,2))
    new_buf = np.zeros((window_size,2))

    plot_filter_comparison(tanh_gains=[0.5, 1.0, 5.0], deadband_radius=0.2)  # Example values for the plot

 ###########   PLOT WITH TIME ###########
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.show(block=False)
    plt.suptitle("Hand Open / Close Predictions - Filter Comparison", fontsize=16)
    plt.title("Tanh_gain = {}, Gain = {}, Integrator step = {}".format(filter.tanh_gain, filter.gain, filter.dt))    

    raw_line1, = ax.plot([], [], label='Predictions - Hand Open/Close ', color='blue', alpha=0.7)
    #raw_line2, = ax.plot([], [], label='Raw Predictions - Motor Function 2', color='cyan', alpha=0.7)
    filtered_line1, = ax.plot([], [], label='Filtered Predictions - Hand Open/Close ', color='green', alpha=0.7)
    #filtered_line2, = ax.plot([], [], label='Filtered Predictions - MF 2', color='lime', alpha=0.7)
    new_line1, = ax.plot([], [], label='Updated Predictions - Hand Open/Close ', color='orange', alpha=0.7)
    #new_line2, = ax.plot([], [], label='Updated Predictions - MF 2', color='yellow', alpha=0.7)

    #ax.set_title()
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Prediction Values")
    ax.legend()
    ax.grid(True)

    run_time = 30  # Run for 60 seconds
    post_plot_size = 500  # Size of the post-session plot

    try:
        #while True:  # Run for 60 seconds
        start_time = time.time()
        while time.time() - start_time < run_time:  # Run for 60 seconds
            # Check if new data is available in the queues
            if not pred_queue.empty() and not filtered_queue.empty():
                raw_pred = pred_queue.get()
                filtered_pred = filtered_queue.get()

                # Update the buffers with new data
                pred_buf = np.roll(pred_buf, -1, axis=0)
                filtered_buf = np.roll(filtered_buf, -1, axis=0)
                new_buf = np.roll(new_buf, -1, axis=0)
                # Add new data to the end
                pred_buf[-1, :] = raw_pred
                filtered_buf[-1, :] = filtered_pred
                new_buf[-1, :] = raw_pred - filtered_pred

                # Update the plot data
                raw_line1.set_data(range(window_size), pred_buf[:, 0])
                #raw_line2.set_data(range(window_size), pred_buf[:, 1])
                filtered_line1.set_data(range(window_size), filtered_buf[:, 0])
                #filtered_line2.set_data(range(window_size), filtered_buf[:, 1])
                new_line1.set_data(range(window_size), new_buf[:, 0])
                #new_line2.set_data(range(window_size), new_buf[:, 1])

                # Adjust plot limits
                ax.set_xlim(0, window_size)
                ax.set_ylim(
                    min(pred_buf.min(), new_buf.min()) - 0.1,
                    max(pred_buf.max(), new_buf.max()) + 0.1
                )

                plt.pause(0.01)  # Pause to update the plot

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected. Stopping threads and exiting...")
        stop_event.set()
        pred_thread.join()
        filtered_thread.join()
        plt.close()

    finally:
        # Close interactive plot
        stop_event.set()
        plt.ioff()
        plt.close(fig)

        # Post-session plot of last `post_plot_size` samples
        post_fig, post_ax = plt.subplots()
        idx = -post_plot_size if post_plot_size <= window_size else -window_size
        post_ax.plot(pred_buf[idx:, 0], label="Raw Prediction", color="blue")
        post_ax.plot(filtered_buf[idx:, 0], label="Filtered", color="green")
        post_ax.plot(new_buf[idx:, 0], label="Feedback = Raw - Filtered", color="red")
        post_ax.set_title(f"Last {post_plot_size} samples")
        post_ax.set_xlabel("Samples")
        post_ax.set_ylabel("Value")
        post_ax.legend()
        post_ax.grid(True)

        # Save to image
        image_path = "../plots/16-05/filtered_feedback_plot.png"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        post_fig.savefig(image_path, bbox_inches='tight', dpi=300)
        print(f"Saved final plot to {image_path}")
        plt.show()


    # ############ PLOT WITH MOTOR FUNCTIONS AS AXES ###########
    # plt.ion()  # Enable interactive mode
    # fig, ax = plt.subplots(figsize=(8, 8))
    # plt.show(block=False)

    # # Set up scatter plots: history and current prediction for raw and filtered
    # history_scatter_raw = ax.scatter([], [], s=30, color='salmon', alpha=0.5, label='Raw History')
    # current_scatter_raw = ax.scatter([], [], s=100, color='red', linewidths=2, label='Raw Current')

    # history_scatter_filtered = ax.scatter([], [], s=30, color='lightgreen', alpha=0.5, label='Filtered History')
    # current_scatter_filtered = ax.scatter([], [], s=100, color='green', linewidths=2, label='Filtered Current')

    # ax.set_xlim(-1, 1)  # Adjust depending on expected range of preds
    # ax.set_ylim(-1, 1)
    # ax.set_xlabel('Prediction Channel 1')
    # ax.set_ylabel('Prediction Channel 2')
    # ax.set_title("Scatter plot of Raw and Filtered Predictions")
    # ax.legend()
    # ax.grid(True)



    # try:
    #     while True:
    #         if not pred_queue.empty() and not filtered_queue.empty():
    #             raw_pred = pred_queue.get()
    #             filtered_pred = filtered_queue.get()

    #             # Shift history buffers
    #             pred_buf = np.roll(pred_buf, -1, axis=0)
    #             filtered_buf = np.roll(filtered_buf, -1, axis=0)

    #             # Store new prediction in the last position
    #             pred_buf[-1, :] = raw_pred
    #             filtered_buf[-1, :] = filtered_pred

    #             # Update scatter data:
    #             # History (all but last)
    #             history_scatter_raw.set_offsets(pred_buf[:-1])
    #             history_scatter_filtered.set_offsets(filtered_buf[:-1])

    #             # Current point (last)
    #             current_scatter_raw.set_offsets([pred_buf[-1]])
    #             current_scatter_filtered.set_offsets([filtered_buf[-1]])

    #             # Combine all points to calculate limits
    #             all_points = np.vstack((pred_buf, filtered_buf))
    #             min_x, min_y = np.min(all_points, axis=0)
    #             max_x, max_y = np.max(all_points, axis=0)

    #             margin_x = (max_x - min_x) * 0.1 if max_x != min_x else 0.1
    #             margin_y = (max_y - min_y) * 0.1 if max_y != min_y else 0.1

    #             ax.set_xlim(min_x - margin_x, max_x + margin_x)
    #             ax.set_ylim(min_y - margin_y, max_y + margin_y)

    #             plt.pause(0.01)

    # except KeyboardInterrupt:
    #     print("\n[INFO] Ctrl+C detected. Stopping threads and exiting...")
    #     stop_event.set()
    #     pred_thread.join()
    #     filtered_thread.join()
    #     plt.close()

   