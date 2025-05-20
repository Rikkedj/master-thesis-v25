import dearpygui.dearpygui as dpg
import json

from pathlib import Path
from PIL import Image as PILImage
import cv2

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

import numpy as np
import threading
from multiprocessing import Process, Queue, Manager, Value
from ctypes import c_bool
from multiprocessing.queues import Empty
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from functools import partial
import time
from collections import deque
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import re
import warnings
import inspect

from libemg.data_handler import OnlineDataHandler, OfflineDataHandler, RegexFilter, FilePackager
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import OnlineEMGClassifier, EMGClassifier, EMGRegressor, OnlineEMGRegressor
from libemg.environments.controllers import ClassifierController, RegressorController
from libemg.prosthesis import Prosthesis, MotorFunctionSelector

from motor_controller import get_motor_setpoints, hybrid_activation_profile
from filters import FlutterRejectionFilter, PostPredictionController # This is the filter that is used to filter the predictions from the model. It is a nonlinear filter that is used to remove noise from the predictions. It is not used in the current version of the code, but it is here for future use.
# Class made by me
class ParameterAdjustmentPanel:
    '''
    The Model Configuration Panel for configuring the machine learning model. 
    
    Parameters
    ----------
    window_size: int, default=150
        The window size used training the machine learning model (in ms?). 
    window_increment: int, default=50
        The window increment for when training the machine learning model (in ms?). 
    deadband_radius: float, default=0.1
        The deadband_radius value for the regression model. If value is less than deadband_radius, the model will output 0. 
    thr_angle_mf1: int, default=20
        The threshold angle for motor function 1. Must be between 0 and 45 degrees. 
        If threshold angles for both motor functions are set to 45, the system will be sequential control. With angles set to 0, both motor functions will always be activated at the same time.
    thr_angle_mf2: int, default=20
        The threshold angle for motor function 2. Must be between 0 and 45 degrees.
        If threshold angles for both motor functions are set to 45, the system will be sequential control. With angles set to 0, both motor functions will always be activated at the same time.
    gain_mf1: float, default=0.5
        The gain for motor function 1. Must be between 0.5 and 3.
    gain_mf2: float, default=0.5
        The gain for motor function 2. Must be between 0.5 and 3.
    gui: GUI
        The GUI object this panel is associated with.
    training_data_folder: str, default='./data/'
        The datafolder where the training data is stored. This is used for training the machine learning model.
    '''
    def __init__(self,
                 window_size=150,                 
                 window_increment=50,
                 deadband_radius=0.1,
                 thr_angle_mf1=20,
                 thr_angle_mf2=20,
                 gain_mf1=0.5,
                 gain_mf2=0.5,
                 training_data_folder='./data/',
                 gui=None):
        
        #self.motor_selector = MotorFunctionSelector() # Create the prosthesis controller object
        
        self.window_increment = window_increment
        self.window_size = window_size
        self.deadband_radius = deadband_radius
        self.thr_angle_mf1 = thr_angle_mf1
        self.thr_angle_mf2 = thr_angle_mf2
        self.gain_mf1 = gain_mf1
        self.gain_mf2 = gain_mf2 
        self.gui = gui
        self.training_data_folder = training_data_folder
        # Manager for multiprocessing - so that we can handle interactive inputs from GUI
        manager = Manager() # This is what takes much time when initiating
        # This needs to be a manager dict, so that it can be shared between processes. Since plotting happens in another process it needs to be this. 
        # However, updating this frequently crashes the gui, so need to be careful with this. Hence, made two separate dicts, one for the plot and one for the controller.
        self.plot_config = manager.dict({ 
            "deadband_radius": self.deadband_radius,
            "thr_angle_mf1": self.thr_angle_mf1,
            "thr_angle_mf2": self.thr_angle_mf2,
            "controller_running": False
        })
        self.post_pred_config = {
            "gain_mf1": self.gain_mf1,
            "gain_mf2": self.gain_mf2,
            "thr_angle_mf1": self.thr_angle_mf1,
            "thr_angle_mf2": self.thr_angle_mf2,
            "deadband_radius": self.deadband_radius
        }

        # self.configuration = manager.dict({
        #     "deadband": self.deadband,
        #     "thr_angle_mf1": self.thr_angle_mf1,
        #     "thr_angle_mf2": self.thr_angle_mf2,
        #     "gain_mf1": self.gain_mf1,
        #     "gain_mf2": self.gain_mf2,
        #     "window_size": self.window_size,
        #     "window_increment": self.window_increment,
        #     "controller_running": False
        # })

        self.flutter_filter = FlutterRejectionFilter() # This is the filter that is used to filter the predictions from the model. It is a nonlinear filter that is used to remove noise from the predictions. It is not used in the current version of the code, but it is here for future use.            
        self.pred_controller = PostPredictionController() # This is the filter that is used to filter the predictions from the model. It is a nonlinear filter that is used to remove noise from the predictions. It is not used in the current version of the code, but it is here for future use.

        # Predictor (classifier or regressor) and queue for predictions for passing via multiprocesses and threads
        self.predictor = None
        self.prediction_queue = Queue()
        self.predictor_running = False # This is used to check if the predictor is running or not. It is used to stop the predictor when the GUI is closed.

        # Controller (classifier or regressor) to read predictions over UDP
        self.controller = None
        self.controller_thread = None
        self.controller_running = False # This is used to check if the controller is running or not. It is used to stop the controller when the GUI is closed.
        # Prosthesis object
        self.prosthesis = None
        self.prosthesis_thread = None
        self.prosthesis_running = False # This is used to check if the prosthesis is running or not. It is used to stop the prosthesis when the GUI is closed.
        # Subprocess for prediction plot, since plotting need to happen in main thread
        self.plot_process = None
        self.plot_running = False # This is used to check if the plot is running or not. It is used to stop the plot when the GUI is closed.

        # Communication with the controller
        self.UDP_IP = "127.0.0.1"
        self.UDP_PORT = 5005

        self.widget_tags = {"configuration": ['__mc_configuration_window',
                                                'deadband_radius', 
                                                'thr_angle_mf1',
                                                'thr_angle_mf2',
                                                'gain_mf1',
                                                'gain_mf2',
                                                'window_size',
                                                'window_increment',
                                                'save_config_tag',
                                                'training_data_folder']}                           

    def cleanup_window(self, window_name): # Don't really know what this does
        widget_list = self.widget_tags[window_name]
        for w in widget_list:
            if dpg.does_alias_exist(w):
                dpg.delete_item(w)

    def spawn_configuration_window(self):
        self.cleanup_window("configuration")
        with dpg.window(tag="__mc_configuration_window",
                        label="Parameter Adjustments",
                        width=900,
                        height=480):
            dpg.add_text(label="Parameter Adjustements", color=(255, 255, 255), bullet=True)

            with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):

                dpg.add_table_column(label="")
                dpg.add_table_column(label="")
                dpg.add_table_column(label="")

                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Training data folder: ")
                        dpg.add_input_text(default_value=self.training_data_folder,
                                            tag="training_data_folder",
                                            width=200,
                                        )
                # Add window size and increment for predictor
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Window Size: ")
                        dpg.add_input_int(default_value=self.window_size, 
                                            tag="window_size",
                                            width=100, 
                                            callback=self.update_value_callback
                                        )                   

                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Window Increment: ")
                        dpg.add_input_int(default_value=self.window_increment,
                                            tag="window_increment",
                                            width=100,
                                            callback=self.update_value_callback # Give another callback, that gets settings, updates plot and updates model
                                        )
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Re-fit Model", 
                                       width = len("Re-fit Model")*10,
                                       callback=self.reset_model_callback
                                    )
                        
                # Add deadband_radius and threshold for regressor   
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Deadband radius(%): ")
                        dpg.add_input_float(default_value=self.deadband_radius, 
                                            tag="deadband_radius",
                                            width=100,
                                            min_value=0,
                                            max_value=0.4,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_value_callback
                                        )  
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Threshold angle mf1 (degrees): ")
                        dpg.add_input_int(default_value=self.thr_angle_mf1,
                                            tag="thr_angle_mf1",
                                            width=100,
                                            min_value=0,
                                            max_value=45,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_value_callback
                                        )
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Threshold angle mf2 (degrees): ")
                        dpg.add_input_int(default_value=self.thr_angle_mf2,
                                            tag="thr_angle_mf2",
                                            width=100,
                                            min_value=0,
                                            max_value=45,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_value_callback
                                        )
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Gain mf1: ")
                        dpg.add_input_float(default_value=self.gain_mf1,
                                            tag="gain_mf1",
                                            width=100,
                                            min_value=0.5,
                                            max_value=3,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_value_callback
                                        )
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Gain mf2: ")
                        dpg.add_input_float(default_value=self.gain_mf2,
                                            tag="gain_mf2",
                                            width=100,
                                            min_value=0.5,
                                            max_value=3,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_value_callback
                                        )
                    with dpg.group(horizontal=True): # Button to store the configuration
                        dpg.add_button(label="Save Configuration", 
                                       width = len("Save Configuration")*10,
                                       callback=self.save_config_callback
                                    )
                    # --- Define the file dialog (initially hidden) ---
                    with dpg.file_dialog(
                        directory_selector=False,   # We want to select a file
                        show=False,                 # Start hidden
                        callback=self._handle_save_dialog_callback, # Callback for OK/Cancel
                        tag='save_config_tag',     # Tag for the file dialog
                        width=700, height=400,
                        default_filename="parameters-adjustments",
                        default_path=str(Path('./post_training_parameters').absolute()), # Use forward slash
                        modal=True                  # Block other interaction
                        ):
                            dpg.add_file_extension(".json", color=(0, 255, 0, 255)) # Filter for json
                            dpg.add_file_extension(".*") # Allow all files too
                # Visualization buttons
                with dpg.table_row():
                    dpg.add_button(label="Visualize Prediction", callback=self.prediction_btn_callback)
                    dpg.add_button(label="Stop Prediction Plot", callback=self.stop_prediction_plot)
                    dpg.add_button(label="Visualize Raw EMG", callback=self.plot_raw_data_callback)
                with dpg.table_row():
                    dpg.add_button(label="Run Prosthesis", callback=self.run_prosthesis_callback)
                    dpg.add_button(label="Stop Prosthesis", callback=self.stop_prosthesis_callback)
    
    #--------------- Button callbacks ---------------------#
    def update_value_callback(self, sender, app_data):
        '''Updates the configuration dictionary with the new values from the GUI.'''
        self.post_pred_config[sender] = app_data      
        pred_controller_keys = {"gain_mf1", "gain_mf2", "deadband_radius"}
        if sender in pred_controller_keys:
            config_subset = {k: self.post_pred_config[k] for k in pred_controller_keys}
            self.pred_controller.update_config(**config_subset)
        plot_config_keys = {"deadband_radius", "thr_angle_mf1", "thr_angle_mf2"}
        if sender in plot_config_keys and self.plot_running:
            config_subset = {k: self.post_pred_config[k] for k in plot_config_keys}
            self.plot_config[sender] = app_data

        print(self.post_pred_config)

    def reset_model_callback(self):
        ''' Resets the model for when the window size or increment is changed, and reset model button is pressed.'''
        self.stop_prediction_plot()
        self.stop_prosthesis_callback()
        self.stop_controller()
        if self.predictor is not None:
            print("Stopping predictor")
            self.predictor.stop_running()
            self.predictor_running = False
            self.predictor = None
        self.run_predictor()
        ("Model reset. Press Visualize Prediction to start again.")
        #self.spawn_configuration_window()

    def prediction_btn_callback(self):
        if not (self.gui.online_data_handler and sum(list(self.gui.online_data_handler.get_data()[1].values()))):
            raise ConnectionError('Attempted to start data collection, but data are not being received. Please ensure the OnlineDataHandler is receiving data.')
        self.run_predictor()
        self.run_controller()
        self.start_prediction_plot()
        #self.spawn_configuration_window() # NOTE! Not sure if this is needed here
        #self.cleanup_window("configuration") # TODO: check out this, can make some problems so that the program crashes        self.spawn_configuration_window()


    def run_prosthesis_callback(self):
        # Set up model and start the predictor
        self.run_predictor()
        self.run_controller()
        # Create prothesis object and connect to it 
        if self.prosthesis is None:
            self.prosthesis = Prosthesis()
            self.prosthesis.connect(port="COM9", baudrate=115200, timeout=1) # Change this to the correct port and baudrate for your prosthesis
        # Set thresholds to the motor function selector
        # self.motor_selector.set_parameters(
        #     thr_angle_mf1=self.post_pred_config["thr_angle_mf1"], 
        #     thr_angle_mf2=self.post_pred_config["thr_angle_mf2"]
        #     )  
        # Start background thread for sending commands to the prosthesis
        self.prosthesis_running = True
        self.prosthesis_thread = threading.Thread(target=self._send_command_to_prosthesis, daemon=True)
        self.prosthesis_thread.start()
        #self.spawn_configuration_window() # Not sure if this is needed here

    def _send_command_to_prosthesis(self):
        """
        This function runs in a separate thread. It gets the predictions from the prediction queue and sends them to the prosthesis.
        """
        while self.controller_running and self.prosthesis_running: # Check if the controller is running
            latest_pred = None  
            # Flush the queue to get only the latest prediction
            while not self.prediction_queue.empty():
                try:
                    latest_pred = self.prediction_queue.get_nowait()
                except Empty:
                    break
            #if not pred_queue.empty():
            if latest_pred is not None:
                #motor_setpoint =  self.motor_selector.get_motor_setpoints(latest_pred) # Get the motor setpoints from the motor function selector
                motor_setpoint = get_motor_setpoints(pred=latest_pred,
                                                    thr_angle_mf1=self.post_pred_config["thr_angle_mf1"],
                                                    thr_angle_mf2=self.post_pred_config["thr_angle_mf2"],
                                                    max_value=255
                                                    )
                print("Motor setpoint to prosthesis: ", motor_setpoint)
                self.prosthesis.send_command(motor_setpoint)
            time.sleep(0.5) # Adjust sending rate
 

    def stop_prosthesis_callback(self):
        self.prosthesis_running = False
        if self.prosthesis_thread and self.prosthesis_thread.is_alive():
            self.prosthesis_thread.join()
        if self.prosthesis and self.prosthesis.is_connected():
            self.prosthesis.disconnect() # Disconnect the prosthesis
            self.prosthesis = None
        print("STop controller thread in stop prosthesis")
        # Only stop controller if plotting thread is not running
        if not self.plot_running:
            self.stop_controller()    
        

    ## Callbacks for plotting raw EMG - from LibEMG
    def plot_raw_data_callback(self):
        self.visualization_thread = threading.Thread(target=self._run_visualization_helper)
        self.visualization_thread.start()

    #--------------- File save dialog callback ---------------------#
    def save_config_callback(self, sender, app_data):
        """Opens the pre-defined file save dialog."""
        dpg.show_item('save_config_tag')

    def _handle_save_dialog_callback(self, sender, app_data):
        """Handles the result of the file save dialog."""
        if app_data is None or 'file_path_name' not in app_data or not app_data['file_path_name']:
            print("Save cancelled by user.")
            return # User cancelled
        
        file_path = Path(app_data['file_path_name'])

        # Ensure the extension is .json if not provided
        if file_path.suffix.lower() != ".json": 
            file_path = file_path.with_suffix(".json")

        # Check if file exists
        # if file_path.exists():
        #     # Store path and ask for confirmation
        #     self._pending_save_path = file_path
        #     self._show_overwrite_confirmation_dialog(file_path)
        # else:
        #     # File doesn't exist, save directly
        self._save_configuration_to_file(file_path) # Always overwrite at this point


    def _save_configuration_to_file(self, file_path: Path):
        """Saves the current configuration to a file."""
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.get_settings() # Get the settings from the GUI
            # TODO: Don't need this as function in motor selection, makes more sense just to have it in this class?
            #self.motor_selector.set_parameters(gain_mf1=self.gain_mf1, gain_mf2=self.gain_mf2, thr_angle_mf1=self.thr_angle_mf1, thr_angle_mf2=self.thr_angle_mf2,deadband_radius=self.deadband_radius) # Set the configuration in the prosthesis controller
            #self.motor_selector.write_to_json(file_path) # Save the configuration to a file
            # Convert Manager.dict proxy to a standard dict 
            # config_to_save = dict(self.configuration) 
            with open(file_path, 'w') as f:
                json.dump(self.post_pred_config, f, indent=4) # Save the configuration to a file
            print(f"Configuration saved to {file_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    # -------------- Helper functions for the callbacks ---------------------#
    def run_predictor(self):
        if self.predictor is None and not self.predictor_running: # Could also just check if running-flag is False?
            self.get_settings()
            self.set_up_model()
            self.predictor.run(block=False)
            self.predictor_running = True

    def get_settings(self):
        self.deadband_radius = float(dpg.get_value(item="deadband_radius"))
        self.gain_mf1 = float(dpg.get_value(item="gain_mf1"))
        self.gain_mf2 = float(dpg.get_value(item="gain_mf2"))
        self.thr_angle_mf1 = int(dpg.get_value(item="thr_angle_mf1"))
        self.thr_angle_mf2 = int(dpg.get_value(item="thr_angle_mf2"))
        self.window_size = int(dpg.get_value(item="window_size"))
        self.window_increment = int(dpg.get_value(item="window_increment"))
        self.training_data_folder = dpg.get_value(item="training_data_folder")
        print("Settings updated")

    def stop_controller(self):
        if self.controller_thread and self.controller_thread.is_alive():
            self.controller_thread.join(timeout=1)
            print("Controller thread terminated")
            self.controller_thread = None
            self.controller_running = False 
            print("Controller not running")

        # if self.controller and hasattr(self.controller, "sock"):
        #     self.controller.sock.close()
        #     print("Socket closed")
        # print("Controller stopped")
        # self.controller = None

    def start_prediction_plot(self):
        if self.plot_process and self.plot_process.is_alive(): # Does this in the stop-function as well, so might be redundant
            self.stop_prediction_plot()

        plotter = PredictionPlotter(self.gui.axis_media) #self.gui.axis_images
        self.plot_process = Process(target=plotter.run, args=(self.plot_config,self.prediction_queue)) #self.plot_process = Process(target=self.plotter, args=(self.configuration, self.prediction_queue))
        self.plot_process.start()
        self.plot_running = True

    def stop_prediction_plot(self):
        if self.plot_process and self.plot_process.is_alive():
            if not self.prosthesis_running: # Only stop the controller if the prosthesis is not running
                self.stop_controller()

            self.plot_process.join(timeout=1)
            if self.plot_process.is_alive(): # If the process is still alive after join, terminate it
                self.plot_process.terminate()
                self.plot_process.join()
            print("Plot process terminated")
            self.plot_process = None
            self.plot_running = False

    def run_controller(self):
        if self.controller_thread and self.controller_thread.is_alive(): # Can also just check the controller_running value?
            print("Controller thread already running")
            return
        self.controller_running = True
        self.plot_config["controller_running"] = True # This is for the plotter to stop if the controller is not running
        print("Starting controller thread")
        self.controller_thread = threading.Thread(target=self._run_controller_helper,)
        self.controller_thread.start()
    
    def _run_controller_helper(self):
        """
        This function runs in a separate thread. It gets the predictions from the predictor model and puts them in a queue for the plotter to use.
        """
        if self.gui.regression_selected:
            self.controller = RegressorController(ip=self.UDP_IP, port=self.UDP_PORT)
        else:
            self.controller = ClassifierController(output_format='predictions', num_classes=self.num_motions, ip=self.UDP_IP, port=self.UDP_PORT)
        
        while self.predictor_running and self.controller_running:  
            pred = self.controller.get_data(["predictions"])
            if pred is not None: 
                #filtered_pred = self.flutter_filter.filter(pred) # This is the filter that is used to filter the predictions from the model. It is a nonlinear filter that is used to remove noise from the predictions. It is not used in the current version of the code, but it is here for future use.
                filtered_pred = self.flutter_filter.update(pred)
                #print("Filtered prediction: ", filtered_pred)
                pred_feedback = pred - filtered_pred
                #print("Pred after flutter filter: ", pred_new)
                pred_controlled = self.pred_controller.update_prediction(pred_feedback)
                #print("Pred after post prediction controller: ", pred_new)
                self.prediction_queue.put(pred_controlled)
                #time.sleep(0.1) # Maybe not a good idea, but not to overload the system
        # Close socket, stop and reset controller when done running
        if self.controller and hasattr(self.controller, "sock"):
            self.controller.sock.close()
            self.controller_running = False
            self.controller = None
            print("Socket closed")
                
    
    def set_up_model(self):
        # Step 1: Parse offline training data
        with open(self.training_data_folder + '/collection_details.json', 'r') as f:
            collection_details = json.load(f)
        
        def _match_metadata_to_data(metadata_file: str, data_file: str, class_map: dict) -> bool:
            """
            Ensures the correct animation metadata file is matched with the correct EMG data file.

            Args:
                metadata_file (str): Metadata file path (e.g., "animation/collection_hand_open_close.txt").
                data_file (str): EMG data file path (e.g., "data/regression/C_0_R_01_emg.csv").
                class_map (dict): Dictionary mapping class index (str) to motion filenames.

            Returns:
                bool: True if the metadata file corresponds to the class of the data file.
            """
            # Extract class index from data filename (C_{k}_R pattern)
            match = re.search(r"C_(\d+)_R", data_file)
            if not match:
                return False  # No valid class index found

            class_index = match.group(1)  # Extract class index as a string

            # Find the expected metadata file from class_map
            expected_metadata = class_map.get(class_index)
            if not expected_metadata:
                return False  # No matching motion found

            # Construct the expected metadata filename
            expected_metadata_file = f"animation/{expected_metadata}.txt"

            return metadata_file == expected_metadata_file
        
        self.num_motions = collection_details['num_motions']
        self.num_reps = collection_details['num_reps']
        self.motion_names = collection_details['classes']
        self.class_map = collection_details['class_map']
        
        if self.gui.regression_selected:
            regex_filters = [
                RegexFilter(left_bound = f"{self.training_data_folder}/C_", right_bound="_R", values = [str(i) for i in range(self.num_motions)], description='classes'),
                RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = [str(i) for i in range(self.num_reps)], description='reps') # TODO! Add a way to remove the discard the first rep
            ]
            metadata_fetchers = [
                FilePackager(RegexFilter(left_bound='animation/', right_bound='.txt', values=self.motion_names, description='labels'), package_function=lambda meta, data: _match_metadata_to_data(meta, data, self.class_map) ) #package_function=lambda x, y: True)
            ]
            labels_key = 'labels'
            metadata_operations = {'labels': 'last_sample'}
        else:
            regex_filters = [
                RegexFilter(left_bound = "classification/C_", right_bound="_R", values = [str(i) for i in range(self.num_motions)], description='classes'),
                RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = [str(i) for i in range(self.num_reps)], description='reps')
            ]
            metadata_fetchers = None
            labels_key = 'classes'
            metadata_operations = None
        
        offline_dh = OfflineDataHandler()
        offline_dh.get_data('./', regex_filters, metadata_fetchers=metadata_fetchers, delimiter=",")
        train_windows, train_metadata = offline_dh.parse_windows(self.window_size, self.window_increment, metadata_operations=metadata_operations)

        # Step 2: Extract features from offline data
        fe = FeatureExtractor()
        print("Extracting features")
        if self.gui.feature_list is not None:
            feature_list = self.gui.feature_list
        else: 
            feature_list = fe.get_feature_groups()['HTD']
        
        training_features = fe.extract_features(feature_list, train_windows, array=True)

        # Step 3: Dataset creation
        data_set = {}
        print("Creating dataset")
        data_set['training_features'] = training_features
        data_set['training_labels'] = train_metadata[labels_key]

        # Step 4: Create and fit the prediction model
        self.gui.online_data_handler.prepare_smm()
        model = self.gui.model_str
        print('Fitting model...')
        if self.gui.regression_selected:
            # Regression
            emg_model = EMGRegressor(model=model)
            emg_model.fit(feature_dictionary=data_set)
            # consider adding a threshold angle here, or just do this when setting up the controller
            #emg_model.add_deadband_radius(self.deadband_radius) # Add a deadband_radius to the regression model. Value below this threshold will be considered 0.
            self.predictor = OnlineEMGRegressor(emg_model, self.window_size, self.window_increment, self.gui.online_data_handler, feature_list, ip=self.UDP_IP, port=self.UDP_PORT, std_out=False)
        else:
            # Classification
            emg_model = EMGClassifier(model=model)
            emg_model.fit(feature_dictionary=data_set)
            emg_model.add_velocity(train_windows, train_metadata[labels_key])
            self.predictor = OnlineEMGClassifier(emg_model, self.window_size, self.window_increment, self.gui.online_data_handler, feature_list, output_format='probabilities', ip=self.UDP_IP, port=self.UDP_PORT)

        # Step 5: Create online EMG model and start predicting.
        print('Model fitted!')
    
    
    def _run_visualization_helper(self):
        self.gui.online_data_handler.visualize(block=False)

    

# Change this to be defined somewhere else, but for testing its here
SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
class PredictionPlotter:
    '''
    The Prediction Plotter class.
    Parameters
    ----------
    axis_media: dict, default=None
        The axis media for the plot. Should be a dictionary with keys 'N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW' (or less) and values as the images or videos to be displayed.
        If None, the default images will be used.
    '''
    def __init__(self, 
                 axis_media_paths={ 
                    'N': Path('images/gestures', 'hand_open.png'),
                    'S': Path('images/gestures', 'hand_close.png'),
                    'E': Path('images/gestures', 'pronation.png'),
                    'W': Path('images/gestures', 'supination.png')
                }):
        self.config = None
        self.pred_queue = None
        #self.axis_media_paths = axis_media_paths
        self.axis_media = axis_media_paths # This is the dictionary with the images/videos to be displayed in the plot
        self.loaded_media = {}  # Stores {'location': ('type', media_object)}
        self.media_artists = {} # Stores {'location': AxesImage artist}
        self.video_caps = {}    # Stores {'location': cv2.VideoCapture} for cleanup

        self.history = deque(maxlen=1000)
        
        self._load_all_media() # Load media during initialization

    def _load_media(self, location, media):
        '''Loads media (image or video) from a file path or uses the media directly if provided.'''
        if isinstance(media, (PILImage.Image, np.ndarray)): 
            # Media is already loaded as an image or video frame
            if isinstance(media, PILImage.Image):
                print(f"Using preloaded image for {location}.")
                return 'image', media
            elif isinstance(media, np.ndarray):
                print(f"Using preloaded video frame for {location}.")
                return 'video', media
        elif isinstance(media, Path):
            # Media is provided as a file path, load it
            if not media.exists():
                warnings.warn(f"Media file not found at {media}. Skipping {location}.")
                return None, None

            ext = media.suffix.lower()

            try:
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    img = PILImage.open(media)
                    # Ensure image is in RGB or RGBA for matplotlib
                    if img.mode not in ['RGB', 'RGBA']:
                        img = img.convert('RGB')
                    return 'image', img

                elif ext in SUPPORTED_VIDEO_EXTENSIONS:
                    cap = cv2.VideoCapture(str(media))
                    if not cap.isOpened():
                        warnings.warn(f"Could not open video file: {media}. Skipping {location}.")
                        return None, None
                    ret, frame = cap.read()
                    if not ret:
                        warnings.warn(f"Could not read first frame from video: {media}. Skipping {location}.")
                        cap.release()
                        return None, None
                    # Convert BGR (OpenCV default) to RGB (Matplotlib default)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.video_caps[location] = cap  # Store for later access and release
                    return 'video', frame_rgb  # Return the first frame for initial display
                else:
                    warnings.warn(f"Unsupported file extension '{ext}' for {media}. Skipping {location}.")
                    return None, None
            except Exception as e:
                warnings.warn(f"Error loading media {media} for {location}: {e}")
                # Ensure capture is released if error occurred during video loading
                if 'cap' in locals() and isinstance(cap, cv2.VideoCapture) and cap.isOpened():
                    cap.release()
                return None, None
        else:
            warnings.warn(f"Unsupported media type for {location}. Expected Path, PIL.Image, or np.ndarray.")
            return None, None   

    def _load_all_media(self):
        """Loads all media specified in axis_media_paths."""
        print("Loading axis media...")
        for location, media in self.axis_media.items():
            media_type, media_data = self._load_media(location, media)
            if media_type:
                self.loaded_media[location] = (media_type, media_data)
                print(f"  Loaded {media_type} for {location}")#: {os.path.basename(path)}")
            else:
                 print(f"  Failed to load media for {location}")#: {os.path.basename(path)}")
        print("Finished loading axis media.")
                    
    def _initialize_plot(self, config):
        self.fig = plt.figure(figsize=(8, 8))
        # Define grid locations mapped to more descriptive names
        gs_map = {
            'NW': (0, 0), 'N': (0, 1), 'NE': (0, 2),
            'W':  (1, 0), 'C': (1, 1), 'E':  (1, 2), # C for Center/Main
            'SW': (2, 0), 'S': (2, 1), 'SE': (2, 2)
        }
        gs = gridspec.GridSpec(3, 3, figure=self.fig, width_ratios=[1, 2, 1], height_ratios=[1, 2, 1])

        # Create axes dictionary
        self.axes = {}
        for loc, (r, c) in gs_map.items():
            self.axes[loc] = self.fig.add_subplot(gs[r, c])

        # --- Configure Main Plot (Center) ---
        self.ax_main = self.axes['C']
        self.ax_main.set_title("Estimated Motor Function")
        self.ax_main.set_xlabel("MF 1")
        self.ax_main.set_ylabel("MF 2")
        self.ax_main.grid(True)
        
        # # Create subplots in the grid
        # self.ax_main = self.fig.add_subplot(gs[1, 1])   # Main (center)
        # self.ax_north = self.fig.add_subplot(gs[0, 1])  # North (top center)
        # self.ax_west = self.fig.add_subplot(gs[1, 0])   # West (left center)
        # self.ax_east = self.fig.add_subplot(gs[1, 2])   # East (right center)
        # self.ax_south = self.fig.add_subplot(gs[2, 1])  # South (bottom center)
        # self.ax_ne = self.fig.add_subplot(gs[0, 2])     # North East (top right)
        # self.ax_nw = self.fig.add_subplot(gs[0, 0])     # North West (top left)

        #self.ax_main.plot(np.sin(np.linspace(0, 10, 100)))  # Example data
        # Plot of predictions
        self.tale_plot, = self.ax_main.plot([], [], 'o', color='gray', markersize=4, alpha=0.5, label='Tale')
        self.current_plot, = self.ax_main.plot([], [], 'o', color='red', markersize=8, markeredgecolor='black', label='Current Prediction')
        
        # Create a circle for the deadband_radius
        self.circle = plt.Circle((0, 0), config["deadband_radius"], color='r', fill=False, linestyle='dashed')
        self.ax_main.add_patch(self.circle)
        # Threshold lines
        self.threshold_lines = [
            self.ax_main.plot([], [], 'b')[0],
            self.ax_main.plot([], [], 'b')[0],
            self.ax_main.plot([], [], 'b')[0],
            self.ax_main.plot([], [], 'b')[0]
        ]

        # --- Configure Surrounding Media Axes ---
        for location, ax in self.axes.items():
            if location == 'C': # Skip center plot
                continue

            # Hide axis labels for surrounding figures
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off') # Turn off axis decorations

            if location in self.loaded_media:
                media_type, media_data = self.loaded_media[location]
                # Display the image or the first frame of the video
                img_artist = ax.imshow(media_data)
                self.media_artists[location] = img_artist # Store artist for updates
            # else:
                # Optionally display a placeholder if media failed to load
                # ax.text(0.5, 0.5, f'{location}\n(No Media)', ha='center', va='center', fontsize=9, color='grey')

        # # Figures for images of motor functions
        # #self.ax_north.set_title()
        # self.ax_north.imshow(self.axis_images['N'])

        # #self.ax_south.set_title("South")
        # self.ax_south.imshow(self.axis_images['S'])

        # #self.ax_west.set_title("West")
        # self.ax_west.imshow(self.axis_images['W'])

        # #self.ax_east.set_title("East")
        # self.ax_east.imshow(self.axis_images['E'])

        # # Hide axis labels for surrounding figures
        # for ax in [self.ax_north, self.ax_south, self.ax_west, self.ax_east]:
        #     ax.set_xticks([])
        #     ax.set_yticks([])


    def _calculate_range(self):
        if len(self.history) > 1:
            history_array = np.array(self.history)
            x_min, x_max = np.min(history_array[:, 0]), np.max(history_array[:, 0])
            y_min, y_max = np.min(history_array[:, 1]), np.max(history_array[:, 1])
            return max(x_max - x_min, y_max - y_min, 1)
        return 1
    
    def _update_plot_limits(self):
        """Updates the main plot limits based on data range."""
        plot_range = self._calculate_range()
        # Center the plot around the origin or the data mean if preferred
        center_x, center_y = 0, 0

        self.ax_main.set_xlim(center_x - plot_range, center_x + plot_range)
        self.ax_main.set_ylim(center_y - plot_range, center_y + plot_range)
        #self.ax_main.set_aspect('equal', adjustable='box') # Keep the aspect ratio equal
    
    def _draw_threshold(self, config):
        thresh_rad_x = np.deg2rad(config["thr_angle_mf1"])
        range = self._calculate_range()
        x_vals = np.array([-range, range]) # TODO: Make range dynamic or something
        y_vals1, y_vals2 = np.tan(thresh_rad_x) * x_vals, -np.tan(thresh_rad_x) * x_vals
        
        thresh_rad_y = np.deg2rad(config["thr_angle_mf2"])
        y_vals = np.array([-range, range]) # TODO: Make range dynamic or something
        x_vals1, x_vals2 = np.tan(thresh_rad_y) * y_vals, -np.tan(thresh_rad_y) * y_vals
        
        self.threshold_lines[0].set_data(x_vals, y_vals1)
        self.threshold_lines[1].set_data(x_vals, y_vals2)
        self.threshold_lines[2].set_data(x_vals1, y_vals)
        self.threshold_lines[3].set_data(x_vals2, y_vals)
    
    def _draw_deadband(self, config):
        ''' Updates the deadband circle, given as percent of the range of the data. '''
        self.circle.set_radius(config["deadband_radius"] * self._calculate_range())
    
    def _draw_prediction(self, pred_queue):
        latest_pred = None
        # Flush the queue to get only the latest prediction
        while not pred_queue.empty():
            try:
                latest_pred = pred_queue.get_nowait()
            except Empty:
                break
            
        if latest_pred is not None:
            print("In plotter, pred: ", latest_pred)
            self.history.append(latest_pred.copy())
            tale_array =  np.array(list(self.history)[-5:])
            
            if tale_array.shape[0] > 1:
                self.tale_plot.set_xdata(tale_array[:, 0])
                self.tale_plot.set_ydata(tale_array[:, 1])
            
            self.current_plot.set_xdata(tale_array[-1:, 0])
            self.current_plot.set_ydata(tale_array[-1:, 1])
    
    def _update_media_frames(self):
        """Updates frames for any video media."""
        for location, cap in self.video_caps.items():
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    artist = self.media_artists.get(location)
                    if artist:
                        artist.set_data(frame_rgb)
                else:
                    # End of video: loop back to the beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read() # Read the first frame again
                    if ret:
                         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                         artist = self.media_artists.get(location)
                         if artist:
                            artist.set_data(frame_rgb)

    def update(self, frame, config, pred_queue):
        if not config["controller_running"]:
            print("Stopping animation...")
            if hasattr(self, 'anim') and self.anim.event_source:
                 self.anim.event_source.stop()
            self.close() # Release resources
            # plt.close(self.fig) # Closing handled in self.close()
            return [] # Return empty list of artists

        self._draw_threshold(config=config)
        self._draw_deadband(config=config)
        self._draw_prediction(pred_queue=pred_queue)
        self._update_plot_limits()
        #self._update_media_frames() # NOTE! Only do this if you have videos playing - not fixed for now

        # Efficiently redraw necessary parts
        self.fig.canvas.draw_idle()
        # self.ax_main.relim()
        # self.ax_main.autoscale_view() # self.ax updatet initilize function27.03 11:52

        # Return list of artists that have been updated
        # If blit=True, this is crucial. If blit=False, less so, but good practice.
        updated_artists = [self.tale_plot, self.current_plot, self.circle]
        updated_artists.extend(self.threshold_lines)
        updated_artists.extend(self.media_artists.values())
        # Include axes if limits changed, but return artists for blitting
        # If not using blit=True, returning [] or the list is often fine.
        return updated_artists
    
    def run(self, config, pred_queue):
        self._initialize_plot(config)
        #self._add_images()
        self.anim = FuncAnimation(self.fig, partial(self.update, config=config, pred_queue=pred_queue), interval=50, blit=False, cache_frame_data=False, repeat=False)
        plt.tight_layout()
        plt.show()
        # Cleanup after window is closed (optional, depends if run() is the end)
        self.close()

    def close(self):
        """Releases resources, especially video captures."""
        print("Closing PredictionPlotter and releasing resources...")
        # Release OpenCV VideoCapture objects
        for loc, cap in self.video_caps.items():
            if cap.isOpened():
                print(f"  Releasing video capture for {loc}...")
                cap.release()
        self.video_caps.clear() # Clear the dictionary

        # Close the matplotlib figure if it exists and is managed by this instance
        if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number):
             print("  Closing Matplotlib figure...")
             plt.close(self.fig)

        print("Resources released.")

    # ########## Got from LibEMG (CartesianPlotAnimator in animator.py)  #################  
    # def _format_figure(self):
    #     max_range = self._calculate_range()
    #     axis_limits = (-max_range*2, max_range*2)
    #     if self.axis_images is not None:
    #         #self.ax.axis('off')  # hide default axis
    #         # Make 3 x 3 grid
    #         grid_shape = (3, 3)
    #         gs = self.fig.add_gridspec(grid_shape[0], grid_shape[1], width_ratios=[1, 2, 1], height_ratios=[1, 2, 1])

    #         # Create subplots using the gridspec
    #         axs = np.empty(shape=grid_shape, dtype=object)
    #         for row_idx in range(grid_shape[0]):
    #             for col_idx in range(grid_shape[1]):
    #                 ax = plt.subplot(gs[row_idx, col_idx])
    #                 if (row_idx, col_idx) != (1, 1):
    #                     # Disable axis for figures/, not for main plot
    #                     ax.axis('off')
    #                 axs[row_idx, col_idx] = ax

    #         loc_axis_map = {
    #             'NW': axs[0, 0],
    #             'N': axs[0, 1],
    #             'NE': axs[0, 2],
    #             'W': axs[1, 0],
    #             'E': axs[1, 2],
    #             'SW': axs[2, 0],
    #             'S': axs[2, 1],
    #             'SE': axs[2, 2]
    #         }
    #         for loc, image in self.axis_images.items():
    #             self.ax = loc_axis_map[loc]
    #             self.ax.imshow(image)
    #         # Set main axis so icon is drawn correctly
    #         plt.sca(axs[1, 1])    
    #         self.ax = axs[1, 1]
        
    #     ticks = [-1., -0.5, 0, 0.5, 1.]
    #     plt.xticks(ticks)
    #     plt.yticks(ticks)
    #     plt.axis('equal')
    #     self.ax.set(xlim=axis_limits, ylim=axis_limits)
    #     #return fig, ax
    
    @staticmethod
    def _add_image_label_axes(fig: Figure):
        """Add axes to a matplotlib Figure for displaying figures/ in the top, right, bottom, and left of the Figure. 
        
        Parameters
        ----------
        fig: matplotlib.pyplot.Figure
            Figure to add image axes to.
        
        Returns
        ----------
        np.ndarray
            Array of matplotlib axes objects. The location in the array corresponds to the location of the axis in the figure.
        """
        # Make 3 x 3 grid
        grid_shape = (3, 3)
        gs = fig.add_gridspec(grid_shape[0], grid_shape[1], width_ratios=[1, 2, 1], height_ratios=[1, 2, 1])

        # Create subplots using the gridspec
        axs = np.empty(shape=grid_shape, dtype=object)
        for row_idx in range(grid_shape[0]):
            for col_idx in range(grid_shape[1]):
                ax = plt.subplot(gs[row_idx, col_idx])
                if (row_idx, col_idx) != (1, 1):
                    # Disable axis for figures/, not for main plot
                    ax.axis('off')
                axs[row_idx, col_idx] = ax
        
        return axs