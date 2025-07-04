import dearpygui.dearpygui as dpg
import json
import os

from pathlib import Path
from PIL import Image as PILImage
import cv2

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

import numpy as np
import threading
from multiprocessing import Process, Queue, Manager, Value
from ctypes import c_bool
from queue import Empty
import time
from collections import deque
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import re
import warnings
import inspect
from tkinter import messagebox

from libemg.data_handler import OnlineDataHandler, OfflineDataHandler, RegexFilter, FilePackager
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import OnlineEMGClassifier, EMGClassifier, EMGRegressor, OnlineEMGRegressor
from libemg.environments.controllers import ClassifierController, RegressorController
from libemg.post_processing import FlutterRejectionFilter, PostPredictionAdjuster # This is the filter that is used to filter the predictions from the model. It is a nonlinear filter that is used to remove noise from the predictions. It is not used in the current version of the code, but it is here for future use.

from prosthesis import Prosthesis
from motor_controller import get_motor_setpoints, hybrid_activation_profile
from plotter import plot_emg_data_and_features, PredictionPlotter
from ml_help_functions import auto_split_reps

# Class made by me
class PostProcessingPanel:
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
                 window_size=400,
                 window_increment=100,
                 deadband_radius=0.1,
                 thr_angle_mf1=45,
                 thr_angle_mf2=45,
                 gain_hand_open=1,
                 gain_hand_close=1,
                 gain_pronation=1,
                 gain_supination=1,
                 training_data_folder='data/',
                 training_media_folder='animation/', # default for regression
                 gui=None):

        self.window_increment = window_increment
        self.window_size = window_size
        self.deadband_radius = deadband_radius
        self.thr_angle_mf1 = thr_angle_mf1
        self.thr_angle_mf2 = thr_angle_mf2
        self.gain_hand_open = gain_hand_open
        self.gain_hand_close = gain_hand_close
        self.gain_pronation = gain_pronation
        self.gain_supination = gain_supination
        self.gui = gui

        self.offline_data_handler = None # This is the offline data handler that is used to get the data from the training data folder. It is set in set_up_model.
        self.training_data_folder = training_data_folder
        self.training_media_folder = training_media_folder
        self.model_str = gui.model_str # Initial model string, but can be changed in gui
        self.active_channels = {} # This is a dictionary that will hold the active channels, i.e. the channels that are selected in the GUI. The keys are the channel numbers and the values are booleans.

        self.optional_features = FeatureExtractor().get_feature_list() #["MAV", "ZC", "WL", "MYOP",]
        self.optional_feature_groups = [fs for fs in FeatureExtractor().get_feature_groups().keys()] #["Time domain", "Frequency domain", "Time-frequency domain"]
        self.selected_features = {}
        self.model_options = { True: ["LR", "SVM", "MLP", "RF", "GB"],                  # If regression is selected
                               False: ["LDA", "KNN", "SVM", "MLP", "RF", "QDA", "NB"]}  # classification

        self.flutter_filter = FlutterRejectionFilter() # This is the filter that is used to filter the predictions from the model. It is a nonlinear filter that is used to remove noise from the predictions. It is not used in the current version of the code, but it is here for future use.
        self.pred_controller = PostPredictionAdjuster() # This is the filter that is used to filter the predictions from the model. It is a nonlinear filter that is used to remove noise from the predictions. It is not used in the current version of the code, but it is here for future use.

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
            "gain_hand_open": self.gain_hand_open,
            "gain_hand_close": self.gain_hand_close,
            "gain_pronation": self.gain_pronation,
            "gain_supination": self.gain_supination,
            "thr_angle_mf1": self.thr_angle_mf1,
            "thr_angle_mf2": self.thr_angle_mf2,
            "deadband_radius": self.deadband_radius
        }
        self.ml_model_config = {
            "window_size": self.window_size,
            "window_increment": self.window_increment,
            "training_data_folder": self.training_data_folder,
            "training_media_folder": self.training_media_folder,
            "model_str": self.model_str,
            "selected_features": self.selected_features
        }


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

        # TODO: Should seperate the tags for each window
        self.widget_tags = {"model_settings": ['model_adjustment_window',
                                                'window_size',
                                                'window_increment',
                                                'model_str',
                                                'training_data_folder',
                                                'training_media_folder'
                                                ],
                            "parameters": [ 'parameter_adjustment_window',
                                            'deadband_radius',
                                            'thr_angle_mf1',
                                            'thr_angle_mf2',
                                            'gain_hand_open',
                                            'gain_hand_close',
                                            'gain_pronation',
                                            'gain_supination',
                                            'save_config_tag'],
                            "flutter_filter": [ 'flutter_filter_window',
                                                'flutter_tanh_gain',
                                                'flutter_gain',
                                                'flutter_dt',
                                                'flutter_filter_enabled']
                                           }

    def cleanup_window(self, window_name):
        widget_list = self.widget_tags[window_name]
        for w in widget_list:
            if dpg.does_alias_exist(w):
                dpg.delete_item(w)

    def spawn_configuration_window(self):
        self.cleanup_window("model_settings")
        self.cleanup_window("parameters")
        self.cleanup_window("flutter_filter")

        #### WINDOW FOR MACHINE LEARNING MODEL SETTINGS ############
        with dpg.window(tag="model_adjustment_window",
                        label="ML Model Settings",
                        width=1100,
                        height=300,
                        pos=(0, 0)):
            #dpg.add_text(label="Model Configuration", color=(255, 255, 255))
            with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp,
                           borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):

                #dpg.add_table_row(label="ML Model Settings", tag="__ml_model_settings")
                dpg.add_table_column(label="")
                dpg.add_table_column(label="")

                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Training data folder \n(must be set correctly before starting predictions): ")
                        dpg.add_input_text(default_value=self.training_data_folder,
                                            tag="training_data_folder",
                                            width=400,
                                        )
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Training media folder \n(where labels for regression is stored): ")
                        dpg.add_input_text(default_value=self.training_media_folder,
                                            tag="training_media_folder",
                                            width=400,
                                        )
                # Add window size and increment for predictor
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Window Size: ")
                        dpg.add_input_int(default_value=self.window_size,
                                            tag="window_size",
                                            width=100,
                                            callback=self.update_ml_model_callback,
                                            step=50
                                        )

                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Window Increment: ")
                        dpg.add_input_int(default_value=self.window_increment,
                                            tag="window_increment",
                                            width=200,
                                            callback=self.update_ml_model_callback, # Give another callback, that gets settings, updates plot and updates model
                                            step=50
                                        )
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Features: ")
                        with dpg.child_window(height=50, autosize_x=True, horizontal_scrollbar=True): # Make this a child window to make it scrollable
                            with dpg.group(horizontal=True):
                                for feature in self.optional_features:
                                    dpg.add_checkbox(label=feature,
                                                    tag=feature,
                                                    default_value=False,
                                                    callback=self.update_ml_model_callback)
                                    #self.selected_features[feature] = False # Initialize selected features

                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Feature sets: ")
                        with dpg.child_window(height=50, autosize_x=True, horizontal_scrollbar=True): # Make this a child window to make it scrollable
                            with dpg.group(horizontal=True):
                                for fg in self.optional_feature_groups:
                                    dpg.add_checkbox(label=fg,
                                                    tag=f"fg_{fg}",
                                                    default_value=False,
                                                    callback=self.update_ml_model_callback
                                                    )

                with dpg.table_row():
                    with dpg.group(horizontal=True, height=60):
                        dpg.add_text(default_value="Model: ")
                        dpg.add_combo(
                                      items=self.model_options[self.gui.regression_selected],
                                      default_value=self.model_str,
                                      tag="model_str",
                                      width=200,
                                      callback=self.update_ml_model_callback
                                    )

                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Active channels: ")
                        with dpg.child_window(height=50, autosize_x=True, horizontal_scrollbar=True): # Make this a child window to make it scrollable
                            with dpg.group(horizontal=True):
                                for i in range(1, 17): # 16 channels
                                    dpg.add_checkbox(label=f"Channel {i}", tag=f"channel_{i}", default_value=False, callback=self.update_channel_callback)

                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Visualize Offline Predictions", callback=self.visualize_offline_predictions_callback)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Visualize Offline EMG and Features", callback=self.visualize_offline_emg_and_features_callback)
                        #dpg.add_button(label="Visualize Offline Features", callback=self.visualize_offline_features_callback)

                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Re-fit Model",
                                       width=len("Re-fit Model") * 15,
                                       callback=self.reset_model_callback
                                    )

        # WINDOW FOR NONLINEAR DEAD-BAND FILTER (FLUTTER FILTER) SETTINGS ############
        with dpg.window(tag="flutter_filter_window",
                        label="Nonlinear Deadband Filter",
                        width=400,
                        height=300,
                        pos=(1200, 0)):
            with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp,
                     borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                dpg.add_table_column(label="")
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_input_float(label="Tanh Gain",
                            default_value=self.flutter_filter.tanh_gain,
                                            tag="flutter_tanh_gain",
                                            width=100,
                                            min_value=0.0,
                                            max_value=40.0,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.flutter_filter_callback
                                        )
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_input_float(label="Gain",
                                            default_value=self.flutter_filter.gain,
                                            tag="flutter_gain",
                                            width=100,
                                            min_value=0.0,
                                            max_value=10.0,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.flutter_filter_callback
                                        )
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_input_float(label="dt (s) (for integrator)",
                                            default_value=self.flutter_filter.dt,
                                            tag="flutter_dt",
                                            width=100,
                                            min_value=0.0,
                                            max_value=1.0,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.flutter_filter_callback
                                        )
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_checkbox(label="Enable Filter",
                                         tag="flutter_filter_enabled",
                                         default_value=True
                                        )

        #### WINDOW FOR POST-PREDICTION CONTROLLER SETTINGS ############
        with dpg.window(tag="parameter_adjustment_window",
                        label="Parameter Adjustments",
                        width=1000,
                        height=300,
                        pos=(0, 350)):
            #dpg.add_text(label="Parameter Adjustements", color=(255, 255, 255))

            with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):

                dpg.add_table_column(label="")
                dpg.add_table_column(label="")
                dpg.add_table_column(label="")

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
                                            callback=self.update_post_pred_callback
                                        )
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="")

                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Visualize Real-Time EMG", callback=self.plot_raw_data_callback)

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
                                            callback=self.update_post_pred_callback
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
                                            callback=self.update_post_pred_callback
                                        )
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Gain Hand Open: ")
                        dpg.add_input_float(default_value=self.gain_hand_open,
                                            tag="gain_hand_open",
                                            width=100,
                                            min_value=0.5,
                                            max_value=3,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_post_pred_callback
                                        )
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Gain Hand Close: ")
                        dpg.add_input_float(default_value=self.gain_hand_close,
                                            tag="gain_hand_close",
                                            width=100,
                                            min_value=0.5,
                                            max_value=3,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_post_pred_callback
                                        )
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Gain Pronation: ")
                        dpg.add_input_float(default_value=self.gain_pronation,
                                            tag="gain_pronation",
                                            width=100,
                                            min_value=0.5,
                                            max_value=3,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_post_pred_callback
                                        )
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Gain Supination: ")
                        dpg.add_input_float(default_value=self.gain_supination,
                                            tag="gain_supination",
                                            width=100,
                                            min_value=0.5,
                                            max_value=3,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_post_pred_callback
                                        )

                # Visualization buttons
                with dpg.table_row():
                    dpg.add_button(label="Visualize Real-Time Predictions", callback=self.real_time_pred_btn_callback)
                    dpg.add_button(label="Stop Prediction Plot", callback=self.stop_prediction_plot)

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

                with dpg.table_row():
                    dpg.add_button(label="Run Prosthesis", callback=self.run_prosthesis_callback)
                    dpg.add_button(label="Stop Prosthesis", callback=self.stop_prosthesis_callback)

    #--------------- Button callbacks ---------------------#
    def update_post_pred_callback(self, sender, app_data):
        '''Updates the post-prediction configuration dictionary with the new values from the GUI.'''
        self.post_pred_config[sender] = app_data
        print(self.post_pred_config)

        # Update post-prediction controller with the new values
        if sender in self.pred_controller.__dict__.keys():
            self.pred_controller.update_config(**{sender: app_data})
            self.pred_controller.__dict__[sender] = app_data

        # Check if plot is running and update plot manager dict. Do this check to avoid
        plot_config_keys = {"deadband_radius", "thr_angle_mf1", "thr_angle_mf2"}
        if sender in plot_config_keys and self.plot_running:
            #config_subset = {k: self.post_pred_config[k] for k in plot_config_keys}
            self.plot_config[sender] = app_data

    def update_ml_model_callback(self, sender, app_data):
        '''Updates the ML model configuration dictionary with the new values from the GUI.'''
        # Check if it's an optional feature
        if sender in self.optional_features:
            self.selected_features[sender] = app_data
            print(self.ml_model_config)
            return

        # Check if it's a feature group (tagged with 'fg_')
        if sender.startswith("fg_"):
            group_name = sender[3:] # Remove 'fg_' prefix to get the group name
            # Check if the group name is valid
            if group_name in self.optional_feature_groups:
                features = FeatureExtractor().get_feature_groups().get(group_name, [])
                # add all features in the group to the selected features
                for feature in features:
                    self.selected_features[feature] = app_data
                    if dpg.does_item_exist(feature):
                        dpg.set_value(feature, app_data)

                print(self.ml_model_config)
                return

        # Otherwise, default to ML model config
        self.ml_model_config[sender] = app_data
        print(self.ml_model_config)

    def update_channel_callback(self, sender, app_data):
        '''Updates the active channels list with the new values from the GUI.'''
        # Check if the sender is a channel checkbox
        channel_number = int(sender.split("_")[1])
        if app_data: # If the checkbox is checked
            self.active_channels[channel_number] = True
        else:
            self.active_channels[channel_number] = False
        print(f"Active channels updated: {self.active_channels}")

    # def update_value_callback(self, sender, app_data):
    #     '''Updates the configuration dictionary with the new values from the GUI.'''
    #     if sender in self.post_pred_config.keys():
    #         self.post_pred_config[sender] = app_data
    #         print(self.post_pred_config)
    #         return
    #     else:  # Not the best solution, if the sender doesn't exist in any it just ends up in ml_model_config. Could do it better later.
    #         if sender in self.optional_features: #self.selected_features.keys():
    #             self.selected_features[sender] = app_data
    #         elif sender in self.optional_feature_groups:
    #             selected_features = FeatureExtractor().get_feature_groups()[sender]
    #             for feature in selected_features:
    #                 self.selected_features[feature] = app_data
    #         else:
    #             self.ml_model_config[sender] = app_data
    #         print(self.ml_model_config)

    #     # pred_controller_keys = {"gain_hand_open", "gain_hand_close", "gain_pronation" ,"gain_supination", "deadband_radius"}
    #     # if sender in pred_controller_keys:
    #     #     config_subset = {k: self.post_pred_config[k] for k in pred_controller_keys}
    #     #     self.pred_controller.update_config(**config_subset)


    def reset_model_callback(self):
        ''' Resets the model for when the window size or increment is changed, and reset model button is pressed.'''
        self.stop_prediction_plot()
        self.stop_prosthesis_callback()
        self.stop_controller()
        self.stop_predictor()
        #self.run_predictor()
        print("Model reset. Press Visualize Prediction to start again.")
        #self.spawn_configuration_window()

    def visualize_offline_predictions_callback(self):
        self.get_settings() # Get the settings from the GUI
        success = self.set_up_model(visualize_preds=True)
        if not success:
            print("Failed to set up model for offline predictions. Check the settings and try again.")
            return


    def visualize_offline_emg_and_features_callback(self):
        self.get_settings() # Get the settings from the GUI
        success = self.set_up_model()
        if not success:
            print("Failed to set up model for offline EMG visualization. Check the settings and try again.")
            return
        # Prevent duplicate windows
        if dpg.does_item_exist("offline_emg_popup"):
            dpg.delete_item("offline_emg_popup")
        # Create a popup window for visualizing offline EMG - to choose channels and repetitions
        if not self.offline_data_handler:
            print("No offline data found. Ensure you have recorded training data.")
            return

        with dpg.window(tag="offline_emg_popup", label="Visualize Offline EMG", width=600, height=400):
                dpg.add_text("Select repetitions:")
                with dpg.child_window(height=150):
                    for rep in range(self.num_reps):
                        dpg.add_checkbox(label=f"Repetition {rep+1}", tag=f"rep_{rep}")

                dpg.add_spacer(height=10)
                dpg.add_text("Select classes:")

                with dpg.child_window(height=150):
                    for class_idx in range(self.num_motions): # Num channels in training dataset. NOTE: Could be called something else
                        dpg.add_checkbox(label=self.class_map[str(class_idx)], tag=f"class_idx_{class_idx}")

                dpg.add_spacer(height=10)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="OK", callback=self._visualize_emg_ok)
                    dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("offline_emg_popup"))

    def _visualize_emg_ok(self):
        selected_reps = [i for i in range(self.num_reps) if dpg.get_value(f"rep_{i}")]
        selected_class_nums = [i for i in range(self.num_motions) if dpg.get_value(f"class_idx_{i}")]
        # Close popup window
        dpg.delete_item("offline_emg_popup")
        if not selected_reps or not selected_class_nums:
            print("No repetitions or channels selected. Please select at least one repetition and one channel.")
            return
        plot_odh = self.offline_data_handler.isolate_data("classes", selected_class_nums)
        plot_odh = plot_odh.isolate_data("reps", selected_reps)
        train_windows, _ = plot_odh.parse_windows(self.window_size, self.window_increment, metadata_operations = {'labels': 'last_sample'}) # Need to extract this again so we only get features from included classes and reps. Is also done in set_up_model
        fe = FeatureExtractor()
        selected_feature_list = [feature for feature, is_selected in self.selected_features.items() if is_selected]
        if not selected_feature_list:
            selected_feature_list = fe.get_feature_groups()['HTD'] # If no features are selected, use the default feature group
        training_features_dict = fe.extract_features(selected_feature_list, train_windows, array=False) # This is used for visualization purposes, so we can plot the features later.

        class_names = {int(c): self.class_map[str(c)].replace("collection_", "").replace("_", " ") for c in selected_class_nums} # Only for when the classes are named collection_"classname". TODO: Remove when this is not the case.

        # Make dictionary for the classnames and which repetitions that belongs to it
        segment_classnames_dict = {name: [] for name in class_names.values()} # Dictionary for the emg and feature plot, classnames with the corresponding segements in the data from odh
        for seg_idx, segment in enumerate(plot_odh.classes): # TODO: Handle this if attribute is not present
                # Flatten all class values from this segment (in case of multiple channels)
                unique_classes_in_segment = np.unique(segment)
                for class_idx, class_name in class_names.items():
                    if class_idx in unique_classes_in_segment:
                        segment_classnames_dict[class_name].append(seg_idx)

        #plot_odh.visualize(block=False, title=f"Offline EMG Visualization for class(es): {[self.class_map[str(cls)] for cls in selected_class_nums]}")
        plot_emg_data_and_features(plot_odh.data,
                                   training_features_dict,
                                   class_names_dict=segment_classnames_dict,
                                   window_size=self.window_size,
                                   window_increment=self.window_increment
                                   )

    def real_time_pred_btn_callback(self):
        if not (self.gui.online_data_handler and sum(list(self.gui.online_data_handler.get_data()[1].values()))):
            raise ConnectionError('Attempted to start data collection, but data are not being received. Please ensure the OnlineDataHandler is receiving data.')
        success = self.run_predictor()
        if not success:
            print("Failed to start online predictions. Check the settings and try again.")
            return
        self.run_controller()
        self.start_prediction_plot()

    def run_prosthesis_callback(self):
        # Set up model and start the predictor
        success = self.run_predictor()
        if not success:
            print("Failed to set up model for online predictions. Check the settings and try again.")
            return
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
                motor_setpoint = get_motor_setpoints(
                                                    pred=latest_pred,
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
        print("Stop controller thread in stop prosthesis")
        # Only stop controller if plotting thread is not running
        if not self.plot_running:
            self.stop_controller()
            #self.stop_predictor()

    def flutter_filter_callback(self, sender, app_data):
        """Updates the flutter filter configuration dictionary with the new values from the GUI."""
        try:
            setting = sender.removeprefix("flutter_")
            self.flutter_filter.update_settings(**{setting: app_data})
            print(self.flutter_filter.__dict__)
        except Exception as e:
            print(f"Error updating flutter filter settings: {e}")


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

        self._save_configuration_to_file(file_path) # Always overwrite at this point


    def _save_configuration_to_file(self, file_path: Path):
        """Saves the current configuration to a file, both post-prediction adjustments and model settings."""
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.get_settings() # Get the settings from the GUI
            flutter_filter_settings = self.flutter_filter.get_settings() # Get the flutter filter settings
            combined_config = {**self.post_pred_config, **self.ml_model_config, **flutter_filter_settings} # Combine the two dictionaries
            with open(file_path, 'w') as f:
                json.dump(combined_config, f, indent=4) # Save the combined settings to a file
            print(f"Parameters saved to {file_path}")
        except Exception as e:
            print(f"Error saving parameters: {e}")

    # -------------- Helper functions for the callbacks ---------------------#
    def run_predictor(self):
        """
        Runs the online emg predictor model in a separate thread.
        """
        if not self.predictor_running: # Could also just check if running-flag is False?
            self.get_settings()
            success = self.set_up_model()
            if not success:
                print("Failed to set up model for online predictions. Check the settings and try again.")
                return False
            self.predictor.run(block=False)
            self.predictor_running = True
            print("Predictor started")
            return True
        else:
            print("Predictor already running")
            return True

    def stop_predictor(self):
        if self.predictor is not None:
            print("Stopping predictor")
            self.predictor.stop_running()
            self.predictor_running = False
            self.predictor = None
        else:
            print("Predictor not running, nothing to stop")

    def get_settings(self):
        # Do I need this? This gets updated every time a change in gui happens. Training data and media folder does not get updated when written, so that needs to be done here for now.
        self.deadband_radius = float(dpg.get_value(item="deadband_radius"))
        self.gain_hand_open = float(dpg.get_value(item="gain_hand_open"))
        self.gain_hand_close = float(dpg.get_value(item="gain_hand_close"))
        self.gain_pronation = float(dpg.get_value(item="gain_pronation"))
        self.gain_supination = float(dpg.get_value(item="gain_supination"))
        self.thr_angle_mf1 = int(dpg.get_value(item="thr_angle_mf1"))
        self.thr_angle_mf2 = int(dpg.get_value(item="thr_angle_mf2"))
        self.window_size = int(dpg.get_value(item="window_size"))
        self.window_increment = int(dpg.get_value(item="window_increment"))
        self.training_data_folder = dpg.get_value(item="training_data_folder")
        self.training_media_folder = dpg.get_value(item="training_media_folder")
        self.model_str = dpg.get_value(item="model_str")
        #self.selected_features = {feature: dpg.get_value(item=feature) for feature in self.optional_features}
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
        print("Starting prediction plotter")
        self.plot_process = Process(target=plotter.run, args=(self.plot_config,self.prediction_queue)) #self.plot_process = Process(target=self.plotter, args=(self.configuration, self.prediction_queue))
        self.plot_process.start()
        self.plot_running = True

    def stop_prediction_plot(self):
        if not self.prosthesis_running: # Only stop the controller if the prosthesis is not running
                self.stop_controller()
                #self.stop_predictor() Tried adding this so you can restart plotting, but that did not work
        if self.plot_process and self.plot_process.is_alive():
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
        
        filtered_pred = [0.0, 0.0]
        while self.predictor_running and self.controller_running:
            pred = self.controller.get_data(["predictions"])
            if pred is not None:
                #filtered_pred = self.flutter_filter.filter(pred) # This is the filter that is used to filter the predictions from the model. It is a nonlinear filter that is used to remove noise from the predictions. It is not used in the current version of the code, but it is here for future use.
                if self.flutter_filter and dpg.get_value("flutter_filter_enabled"):
                    feedback = np.array(pred) - np.array(filtered_pred) # Feedback after removing the filtered prediction
                    filtered_pred = self.flutter_filter.filter(feedback) # Filter prediction to reject fluttering
                else:
                    filtered_pred = pred # If the filter is not enabled, just use the prediction as feedback

                pred_controlled = self.pred_controller.update_prediction(filtered_pred) # Apply gain and deadband to the prediction
                pred_clipped = [self._signed_clip(p) for p in pred_controlled] # Clip the prediction to the range [-1, 1]
                #print("(In run_controller) Prediction after clipping: ", pred_clipped)
                self.prediction_queue.put(pred_clipped)

        # Close socket, stop and reset controller when done running
        if self.controller and hasattr(self.controller, "sock"):
            self.controller.sock.close()
            self.controller_running = False
            self.controller = None
            print("Controller socket closed")


    def set_up_model(self, visualize_preds: bool = False):
        """
        Sets up the model for the predictor. This includes loading the training data, extracting features, and fitting the machine learning model.
        NOTE: This is a bit redundant if if experementing much with offline data, since it will load the training data every time this is called -> could be optimized later.

        Parameters
        ----------
        visualize_preds: bool, default=False
            If True, a plot of the offline prediction stream from the training data will be shown.
        visualize_features: bool, default=False
            If True, a plot of the offline features extracted from the training data will be shown.
        """
        # Read collection details from JSON file, storing pre-training metadata
        collection_file_path = os.path.join(self.training_data_folder, 'collection_details.json')
        if not os.path.exists(collection_file_path):
            print(f"[INFO] File not found: {collection_file_path}. Skipping loading collection details. Try giving the correct training data path.")
            return False

        try:
            with open(collection_file_path, 'r') as f:
                collection_details = json.load(f)
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load collection_details.json: {e}")
            return False

        try:
            self.num_motions = collection_details['num_motions']
            self.num_reps = collection_details['num_reps']
            self.motion_names = collection_details['classes']
            self.class_map = collection_details['class_map']


            def _match_metadata_to_data(metadata_file: str, data_file: str, class_map: dict) -> bool:
                """
                Ensures the correct animation metadata file is matched with the correct EMG data file.

                Parameters:
                    metadata_file (str): Metadata file path (e.g., "animation/collection_hand_open_close.txt").
                    data_file (str): EMG data file path (e.g., "data/regression/C_0_R_1_emg.csv").
                    class_map (dict): Dictionary mapping class index (str) to motion filenames.

                Returns:
                    bool: True if the metadata file corresponds to the class of the data file.
                """
                # Extract class index from data filename (C_{k}_R pattern)
                match = re.search(r"C_(\d+)_R", data_file)
                if not match:
                    print(f"No valid class index found in data file: {data_file}")
                    return False  # No valid class index found

                class_index = match.group(1)  # Extract class index as a string

                # Find the expected metadata file from class_map
                expected_metadata = class_map.get(class_index)
                if not expected_metadata:
                    print(f"No matching motion found for class index {class_index}.")
                    return False  # No matching motion found

                # Construct the expected metadata filename
                #expected_metadata_file = f"animation/test/saw_tooth/{expected_metadata}.txt"
                expected_metadata_file = str(self.training_media_folder)+f"/{expected_metadata}.txt"

                return metadata_file == expected_metadata_file

            # Step 1: Parse offline training data

            if self.gui.regression_selected:
                regex_filters = [
                    RegexFilter(left_bound = f"{self.training_data_folder}/C_", right_bound="_R", values = [str(i) for i in range(self.num_motions)], description='classes'),
                    RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = [str(i) for i in range(self.num_reps)], description='reps') # TODO! Add a way to remove the discard the first rep
                ]
                metadata_fetchers = [
                    FilePackager(RegexFilter(left_bound=self.training_media_folder+"/", right_bound='.txt', values=self.motion_names, description='labels'), package_function=lambda meta, data: _match_metadata_to_data(meta, data, self.class_map) ) #package_function=lambda x, y: True)
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
        except Exception as e:
            messagebox.showerror("Metadata Error", f"Could not process metadata: {e}")
            return False

        try:
            # Create an OfflineDataHandler instance to handle the offline data
            self.offline_data_handler = OfflineDataHandler()
            self.offline_data_handler.get_data('./', regex_filters, metadata_fetchers=metadata_fetchers, delimiter=",")

            self.num_training_channels = self.offline_data_handler.data[0].shape[1] # Get the number of channels from the data
            print(f"Number of channels in the training data: {self.num_training_channels}")
            active_channels = [channel-1 for channel, is_active in self.active_channels.items() if is_active] # Get the active channels from the GUI
            if active_channels: # Only isolate channels if there are active channels. Else it will use all channels.
                if len(active_channels) > self.num_training_channels:
                    raise ValueError(f"Number of active channels ({len(active_channels)}) exceeds the number of channels in the data ({self.num_training_channels}). Please check your settings.")
                self.offline_data_handler = self.offline_data_handler.isolate_channels(active_channels) # Isolate the active channels

            if visualize_preds:
                rep_indices = [i for i in range(self.num_reps)] # Get the selected repetitions from the GUI
                train_reps, val_reps = auto_split_reps(rep_indices=rep_indices, split_ratio=0.6) # Split the repetitions into training and validation sets
                train_odh = self.offline_data_handler.isolate_data("reps", train_reps) # Use the offline data handler to get the training data
                val_odh = self.offline_data_handler.isolate_data("reps", val_reps) # Use the offline data handler to get the validation data
                train_windows, train_metadata = train_odh.parse_windows(self.window_size, self.window_increment, metadata_operations=metadata_operations) # Parse the windows from the training data
                val_windows, val_metadata = val_odh.parse_windows(self.window_size, self.window_increment, metadata_operations=metadata_operations) # Parse the windows from the validation data
            else:
            # Parse the windows from the offline data
                train_windows, train_metadata = self.offline_data_handler.parse_windows(self.window_size, self.window_increment, metadata_operations=metadata_operations)
                val_windows, val_metadata = train_windows, train_metadata # No validation data if not visualizing predictions
        except Exception as e:
            messagebox.showerror("Data Error", f"Could not load or parse training data: {e}")
            return False

        try:
            # Step 2: Extract features from offline data
            fe = FeatureExtractor()
            print("Extracting features")
            selected_feature_list = [feature for feature, is_selected in self.selected_features.items() if is_selected]
            if not selected_feature_list:
                print("No features selected for extraction. HTD is selected by default.")
                selected_feature_list = fe.get_feature_groups()['HTD'] # Default to HTD features if no features are selected

            training_features = fe.extract_features(selected_feature_list, train_windows, array=True)
            val_features = fe.extract_features(selected_feature_list, val_windows, array=True) # Extract features from the validation data
            # Step 3: Dataset creation for EMG model training
            data_set = {}
            print("Creating dataset")
            data_set['training_features'] = training_features
            data_set['training_labels'] = train_metadata[labels_key]
        except Exception as e:
            messagebox.showerror("Feature Extraction Error", f"Could not extract features or create dataset: {e}")
            return False

        try:
            # Step 4: Create and fit the prediction model
            self.gui.online_data_handler.prepare_smm()
            #model = self.gui.model_str

            print('Fitting model...')
            if self.gui.regression_selected:
                # Regression
                emg_model = EMGRegressor(model=self.model_str)
                emg_model.fit(feature_dictionary=data_set)
                self.predictor = OnlineEMGRegressor(emg_model, self.window_size, self.window_increment, self.gui.online_data_handler, selected_feature_list, ip=self.UDP_IP, port=self.UDP_PORT, std_out=False)
                if visualize_preds:
                    mf_labels = [{1:"Hand Open", -1:"Hand Close"}, {1:"Pronation", -1:"Supination"}] # NOTE: This is a hardcoded mapping for the motor functions, so my plot get nice for report.
                    mf_titles = ["Hand Open/Close", "Pronation/Supination"] # This is a hardcoded mapping for the motor functions, so my plot get nice for report.
                    val_labels = val_metadata[labels_key]
                    preds = emg_model.run(val_features)
                    preds_tresh = [self.pred_controller.update_prediction(pred) for pred in preds] # Add gain and deadband to the predictions
                    preds_tresh = np.squeeze(np.array(preds_tresh))  # Removes dimensions of size 1
                    emg_model.visualize(val_labels, preds_tresh, mf_labels_dict=mf_labels, dof_titles=mf_titles, time_axis=True, sample_rate=2000)
            else:
                # Classification
                emg_model = EMGClassifier(model=self.model_str)
                emg_model.fit(feature_dictionary=data_set)
                #emg_model.add_velocity(train_windows, train_metadata[labels_key])
                self.predictor = OnlineEMGClassifier(emg_model, self.window_size, self.window_increment, self.gui.online_data_handler, selected_feature_list, output_format='probabilities', ip=self.UDP_IP, port=self.UDP_PORT)
                if visualize_preds:
                    train_labels = train_metadata[labels_key]
                    preds, probs = emg_model.run(training_features)
                    emg_model.visualize(train_labels, preds, probs) # Visualize the predictions and probabilities
            # Step 5: Create online EMG model and start predicting.
            print('Model fitted!')
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not create or fit the model: {e}")
            return False

        return True  # Indicate that the model was set up successfully

    def _run_visualization_helper(self):
        self.gui.online_data_handler.visualize(block=False)

    @staticmethod
    def _signed_clip(value, min_value=-1, max_value=1):
        """
        Clips a value to the range [min_value, max_value] and preserves the sign.
        """
        if value > 0:
            return min(value, max_value)
        elif value < 0:
            return max(value, min_value)
        else:
            return 0

