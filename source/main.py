from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import cv2

from tkinter import *
from tkinter import messagebox
import re, json
import argparse 

from libemg.streamers import delsys_streamer 
from libemg.gui import GUI
from libemg.data_handler import OnlineDataHandler, OfflineDataHandler, RegexFilter, FilePackager
from libemg.animator import ScatterPlotAnimator
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import OnlineEMGClassifier, EMGClassifier, EMGRegressor, OnlineEMGRegressor
from libemg.environments.controllers import ClassifierController, RegressorController
from libemg.environments.fitts import ISOFitts, FittsConfig
from libemg.post_processing import PostPredictionAdjuster, FlutterRejectionFilter

from parameterGUI import ParameterAdjustmentGUI

class ProsthesisControlGUI:
    """
    Class for a Simultaneous Proportional Prosthesis control system. Inspired by the Menu class in Menu.py from https://github.com/LibEMG/LibEMG_Isofitts_Showcase.git.
    """
    def __init__(self, sensor_ids=[]):
        """
        Initializes the ProsthesisControlGUI class.
        Parameters:
        ----------
        sensor_ids : list of int, default=[]
            List of sensor IDs to be used for the Delsys streamer. Note! The IDs are zero-indexed, so if you want to use the first two sensors, you should pass [0, 1].
        """
        streamer, sm = delsys_streamer(channel_list=sensor_ids) # returns streamer and the shared memory object, need to give in the active channels number -1, so 2 is sensor 3
        print(f"Streamer initialized with channels: {sensor_ids}")
        # Create online data handler to listen for the data
        self.odh = OnlineDataHandler(sm)
        # Learning model
        self.predictor = None # The classifier or regressor object
        self.model_str = None # The model string, used to identify the model type.
        self.training_data_folder = None  # default value
        self.training_media_folder = None  # default value
        self.results_file = None  # default value

        # The UDP IP and port of the controller. This is used to send the data from predictor to the controller.
        self.controller_IP = '127.0.0.1'
        self.controller_PORT = 5005 

        self.post_prediction_adjuster = PostPredictionAdjuster()  # For post prediction adjustments
        self.flutter_rejection_filter = FlutterRejectionFilter()

        # Initialize motion classes for the training protocol
        self.motion_classes = {
            'hand_open': (1, 0),            # Movement along x-axis
            'hand_close': (-1, 0),          # Movement along x-axis
            'pronation': (0, 1),            # Movement along y-axis
            'supination': (0, -1),          # Movement along y-axis
            'rest': (0, 0),                 # Rest position
            #'open_pronate': tuple(np.array([1, 1]) / np.sqrt(2)),          # Combined movement
            #'close_pronate': tuple(np.array([-1, 1]) / np.sqrt(2)),        # Combined movement
            #'open_supinate': tuple(np.array([1, -1]) / np.sqrt(2)),        # Combined movement
            #'close_supinate': tuple(np.array([-1, -1]) / np.sqrt(2))       # Combined movement
        }
        # Medi afor training prompt
        self.axis_media = {
            'N': PILImage.open(Path('media/images/gestures', 'pronation.png')),
            'S': PILImage.open(Path('media/images/gestures', 'supination.png')),
            'E': PILImage.open(Path('media/images/gestures', 'hand_open.png')),
            'W': PILImage.open(Path('media/images/gestures', 'hand_close.png')),
            'NE': PILImage.open(Path('media/images/gestures', 'open_pronate.png')), 
            'NW': PILImage.open(Path('media/images/gestures', 'close_pronate.png')),
            'SE': PILImage.open(Path('media/images/gestures', 'open_supinate.png')),
            'SW': PILImage.open(Path('media/images/gestures', 'close_supinate.png'))
            #'NE': cv2.VideoCapture(Path('images_master/videos', 'IMG_4930.MOV')),  # store path instead
            #'NW': cv2.VideoCapture(Path('images_master/videos', 'IMG_4931.MOV'))
        }
        ########### NOTE: This is for when training with simultaneous gestures can handle videos as media #################
        self.axis_media_paths = {
            'N': Path('media/images/gestures', 'pronation.png'),
            'S': Path('media/images/gestures', 'supination.png'),
            'E': Path('media/images/gestures', 'hand_open.png'),
            'W': Path('media/images/gestures', 'hand_close.png'),
            'NE': Path('media/videos', 'IMG_4930.MOV'), 
            'NW': Path('media/videos', 'IMG_4931.MOV')
        }
        self.window = None
        self.initialize_ui()
        self.window.mainloop()

    def initialize_ui(self):
        '''
        Initializes the UI for the prosthesis control system. This is the main menu for the system.
        The order of execution is as follows:
        1. Choose which model you want. The suppored models are listed. Default is Regression with LR. Choose if you want regression (simultaneous control) or classification (pattern recognition).
        2. Train the model. This will open a new window where you can choose training parameters, such as window size, window increment, repetition of training, etc. 
        3. After training you can adjust control parameters, such as adding deadband, gain and threshold angles.
        4. When model is trained and tuned, you can try the system with Isofitts or run the prosthesis. 
        '''
        # Create the simple menu UI:
        self.window = Tk()
        if not self.training_data_folder:
            self.training_data_folder = StringVar(value='data/regression')  # default value
        else:
            self.training_data_folder = StringVar(value=self.training_data_folder.get())
        
        if not self.training_media_folder:
            self.training_media_folder = StringVar(value='animation/saw_tooth')  # default value
        else:
            self.training_media_folder = StringVar(value=self.training_media_folder.get())
        
        if not self.model_str:
            self.model_str = StringVar(value='LR')
        else:
            self.model_str = StringVar(value=self.model_str.get())
        
        if not self.results_file:
            self.results_file = StringVar(value=f'results/{self.model_str.get()}_reg/[filename].pkl')
        else:
            self.results_file = StringVar(value=self.results_file.get())

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.title("Main Menu")
        self.window.geometry("800x800") 

        # Label 
        Label(self.window, font=("Arial bold", 20), text = 'Simultaneous Proportional Prosthesis Control').pack(pady=(10,20))
        # Train Model Button
        Button(self.window, font=("Arial", 18), text = 'Data Collection', command=self.launch_training).pack(pady=(0,20))
        # Post Training
        Button(self.window, font=("Arial", 18), text = 'Post-Process Parameter Tuning and Run Prosthesis', command=self.adjust_param_callback).pack(pady=(0,20))
        # Start Isofitts
        Button(self.window, font=("Arial", 18), text = 'Start Isofitts Test', command=self.start_test).pack(pady=(0,20))
        
        # Put in training data folder and media folder for running isofitts.
        # Training Data Folder Frame
        training_data_frame = Frame(self.window)
        Label(training_data_frame, text="Training Data Folder:", font=("Arial bold", 14)).pack(side=LEFT, padx=(0,10))
        Entry(training_data_frame, font=("Arial", 14), textvariable=self.training_data_folder, width=40).pack(side=LEFT, fill=X, expand=True)
        training_data_frame.pack(pady=(20,10), padx=20, fill=X)

        # Media Folder Frame  
        media_folder_frame = Frame(self.window)
        Label(media_folder_frame, text="Media Folder:", font=("Arial bold", 14)).pack(side=LEFT, padx=(0,10))
        Entry(media_folder_frame, font=("Arial", 14), textvariable=self.training_media_folder, width=40).pack(side=LEFT, fill=X, expand=True)
        media_folder_frame.pack(pady=(0,10), padx=20, fill=X)
        
        # Result from Fitts test Folder Frame  
        results_file_frame = Frame(self.window)
        Label(results_file_frame, text="Results File (.pkl or .json, for Fitts Test. ):", font=("Arial bold", 14)).pack(side=LEFT, padx=(0,10))
        Entry(results_file_frame, font=("Arial", 14), textvariable=self.results_file, width=40).pack(side=LEFT, fill=X, expand=True)
        results_file_frame.pack(pady=(0,10), padx=20, fill=X)

        # Model Input
        self.model_type = IntVar()
        r1 = Radiobutton(self.window, text='Classification / Pattern Recognition', variable=self.model_type, value=1)
        r1.pack()
        r2 = Radiobutton(self.window, text='Regression / Proportional Control', variable=self.model_type, value=2)
        r2.pack()
        r2.select() # default to regression

        frame = Frame(self.window)
        Label(self.window, text="Model:", font=("Arial bold", 18)).pack(in_=frame, side=LEFT, padx=(0,10))
        Entry(self.window, font=("Arial", 18), textvariable=self.model_str).pack(in_=frame, side=LEFT)
        frame.pack(pady=(20,10))
       
        Label(self.window, text="Classifier Model Options: LDA, KNN, SVM, MLP, RF, QDA, NB", font=("Arial", 15)).pack(pady=(0,10))       
        Label(self.window, text="Regressor Model Options: LR, SVM, MLP, RF, GB", font=("Arial", 15)).pack(pady=(0,10))

    # ---------- Functions for the buttons in the menu -----------
    # Sourced from LibEMG Menu.py from https://github.com/LibEMG/LibEMG_Isofitts_Showcase.git
    def launch_training(self):
        self.window.destroy()
        media_folder = self.training_media_folder.get()
        if not media_folder or not Path(media_folder).exists():
            messagebox.showerror("Invalid Path", "Media folder does not exist. Please provide a valid path.")
            self.initialize_ui()
            return
        
        data_folder = self.training_data_folder.get()
        if not data_folder:
            messagebox.showerror("Invalid Path", "No training data folder given. Please provide a valid path.")
            self.initialize_ui()
            return
        
        args = {'media_folder': media_folder, 'data_folder': data_folder, 'rest_time': 3} # Default arguments for the GUI
        # if self.regression_selected():
        #     #self.training_media_folder = 'animation/test/saw_tooth/'           
        #     args = {'media_folder': media_folder, 'data_folder': Path('data', 'regression').absolute().as_posix(), 'rest_time': 3} 
        # else:
        #     #self.training_media_folder = 'media/images/'
        #     args = {'media_folder': media_folder, 'data_folder': Path('data', 'classification').absolute().as_posix()} # self.training_media_folder
        
        training_ui = GUI(self.odh, args=args, width=1100, height=1000, gesture_height=700, gesture_width=700)
        #training_ui.download_gestures([1,2,3,6,7], "media/images/") # Downloading gestures from github repo. Videos for simultaneous gestures are located in images_master/videos
        if self.regression_selected():
            self.create_animation(transition_duration=2, hold_duration=2, rest_duration=0) # Create animations for the intended motions used in the training prompt.
        training_ui.start_gui()
        self.initialize_ui()
    
    def adjust_param_callback(self):
        self.window.destroy()
        media_folder = self.training_media_folder.get()
        if not media_folder or not Path(media_folder).exists():
            messagebox.showerror("Invalid Path", "Media folder does not exist. Please provide a valid path.")
            self.initialize_ui()
            return
        
        data_folder = self.training_data_folder.get()
        if not data_folder or not Path(data_folder).exists():
            messagebox.showerror("Invalid Path", "Data folder does not exist. Please provide a valid path.")
            self.initialize_ui()
            return
        
        model_str = self.model_str.get()
        if not model_str:
            messagebox.showerror("Invalid Model", "Model string is empty. Please provide a valid model string.")
            self.initialize_ui()
            return
        
        default_params = {'window_size':400, 'window_increment':100, 'deadband': 0., 'thr_angle_mf1': 45, 'thr_angle_mf2': 45, 'gain_mf1': 1, 'gain_mf2': 1} #deafult values for the parameters. 
        adjust_param_ui = ParameterAdjustmentGUI(online_data_handler=self.odh, 
                                                 regression_selected=self.regression_selected(), 
                                                 model_str=model_str, 
                                                 axis_media=self.axis_media, 
                                                 params=default_params,                                                  
                                                 training_data_folder=data_folder,
                                                 training_media_folder=media_folder,
                                                 debug=True)
        adjust_param_ui.start_gui()
        self.initialize_ui() # When the user is done adjusting parameters, the GUI will be closed and the main menu will be re-opened.

    # Sourced from LibEMG Menu.py from https://github.com/LibEMG/LibEMG_Isofitts_Showcase.git
    def start_test(self):
        self.window.destroy()
        success = self.set_up_model()
        if not success:
            return

        save_file = self.results_file.get()
        if not save_file or not (save_file.endswith(".pkl") or save_file.endswith(".json")):
            messagebox.showerror("Invalid Path", "No result file for Fitts test given, or invalid file extension. Please provide a .pkl or .json file.")
            self.initialize_ui()
            return
        
        if self.regression_selected():
            controller = RegressorController(ip=self.controller_IP, port=self.controller_PORT) 
            #save_file = Path('results', self.model_str.get() +'_reg'+ '_unconstrained_NOT_flutter_NOT_tresh' + ".pkl").absolute().as_posix()
        else:
            controller = ClassifierController(ip=self.controller_IP, port=self.controller_PORT, output_format=self.predictor.output_format, num_classes=len(self.motion_classes)) # NOTE! num_classes hardcoded -> get from config
            #save_file = Path('results', self.model_str.get() + '_clf' + ".pkl").absolute().as_posix()
        
        config = FittsConfig(num_trials=16, save_file=save_file)
        ISOFitts(controller, config, post_prediction_adjuster=self.post_prediction_adjuster, flutter_rejection_filter=self.flutter_rejection_filter).run()
        # Its important to stop the model after the game has ended
        # Otherwise it will continuously run in a seperate process
        self.predictor.stop_running()
        self.initialize_ui()

    
    # --------- Helper functions for the buttons in the menu -----------
    def _create_default_parameters(self, file_path):
        """Creates a default model parameters and saves it to a JSON file."""
        default_parameters = {
            'window_size': 400,
            'window_increment': 100,
            'gain_hand_open': 1,
            'gain_hand_close': 1,
            'gain_pronation': 1,
            'gain_supination': 1,
            'thr_angle_mf1': 45,
            'thr_angle_mf2': 45,
            'deadband_radius': 0.1
        }
        
        with open(file_path, 'w') as f:
            json.dump(default_parameters, f, indent=4)
        print(f"Default post-training adjustments saved to {file_path}.")
        return default_parameters

    # Gotten from LibEMG Menu.py. Used when doing Isofitts test.
    def set_up_model(self):
        """Sets up the model by checking for a configuration JSON file or creating a default one."""
        parameters_file_path = Path('./post_training_parameters/parameters-adjustments.json')

        # 1. Load or create post-training parameters
        try:
            if parameters_file_path.exists():
                with open(parameters_file_path, 'r') as f:
                    post_training_parameters = json.load(f)
            else:
                raise FileNotFoundError("Configuration file not found.")
        except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
            print(f"Error reading parameters: {e}. Creating default.")
            post_training_parameters = self._create_default_parameters(parameters_file_path)

        try:
            WINDOW_SIZE = post_training_parameters['window_size']
            WINDOW_INCREMENT = post_training_parameters['window_increment']
            features = post_training_parameters.get('selected_features', {'MAV': True})  # Default to MAV if not specified
            feature_list = [f for f in features.keys() if features[f]]  # Filter out features that are not selected
            post_pred_controller = {
                "gain_hand_open": post_training_parameters['gain_hand_open'],
                "gain_hand_close": post_training_parameters['gain_hand_close'],
                "gain_pronation": post_training_parameters['gain_pronation'],
                "gain_supination": post_training_parameters['gain_supination'],
                "thr_angle_mf1": post_training_parameters['thr_angle_mf1'],
                "thr_angle_mf2": post_training_parameters['thr_angle_mf2'],
                "deadband_radius": post_training_parameters['deadband_radius']
            }
            flutter_filter_settings = { 
                "tanh_gain": post_training_parameters["tanh_gain"],
                "dt": post_training_parameters["dt"],
                "integrator_enabled": post_training_parameters["integrator_enabled"],
                "gain": post_training_parameters["gain"]
            }

            self.post_prediction_adjuster.update_config(**post_pred_controller)  # Update the post prediction adjuster with the parameters
            self.flutter_rejection_filter.update_settings(**flutter_filter_settings)
        except KeyError as e:
            messagebox.showerror("Parameter Error", f"Missing parameter in configuration: {e}. Please check the configuration file.")
            self.initialize_ui()
            return False
                
        
        # 2. Validate media folder
        media_folder = self.training_media_folder.get()
        if not media_folder or not Path(media_folder).exists():
            messagebox.showerror("Invalid Path", "Media folder does not exist. Please provide a valid path.")
            self.initialize_ui()
            return False
        # 3. Validate data folder
        data_folder = self.training_data_folder.get()
        if not data_folder or not Path(data_folder).exists():
            messagebox.showerror("Invalid Path", "Data folder does not exist. Please provide a valid path.")
            self.initialize_ui()
            return False
        # # Set the data folder based on the model type
        # if self.regression_selected():
        #     data_folder = 'data/regression'
        # else:
        #     data_folder = 'data/classification'

        # Load and validate collection details
        try:
            with open(data_folder + '/collection_details.json', 'r') as f:
                self.collection_details = json.load(f)
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load collection_details.json: {e}")
            self.initialize_ui()
            return False
        
        # 4. Process filters and metadata
        try:
            num_motions = self.collection_details['num_motions']
            num_reps = self.collection_details['num_reps']
            motion_names = self.collection_details['classes']
            class_map = self.collection_details['class_map']

            def _match_metadata_to_data(metadata_file: str, data_file: str, class_map: dict, media_folder: str) -> bool:
                """
                Ensures the correct class labels file is matched with the correct EMG data file.

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
                expected_metadata_file = media_folder+f"/{expected_metadata}.txt"

                return metadata_file == expected_metadata_file
        
        
            # The same regexfilter for both classification and regression since data-folder is changed further up. 
            regex_filters = [
                    RegexFilter(left_bound = data_folder + "/C_", right_bound="_R", values = [str(i) for i in range(num_motions)], description='classes'),
                    RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = [str(i) for i in range(num_reps)], description='reps')
                ]
            
            if self.regression_selected():
                metadata_fetchers = [
                    FilePackager(RegexFilter(left_bound=media_folder+"/", right_bound='.txt', values=motion_names, description='labels'), package_function=lambda meta, data: _match_metadata_to_data(meta, data, class_map, media_folder) ) #package_function=lambda x, y: True)
                ]
                labels_key = 'labels'
                metadata_operations = {'labels': 'last_sample'}
            else:
                metadata_fetchers = None
                labels_key = 'classes'
                metadata_operations = None

        except Exception as e:
            messagebox.showerror("Metadata Error", f"Could not process metadata: {e}")
            self.initialize_ui()
            return False

        # 5. Parse training data and fit model
        try:
            offline_dh = OfflineDataHandler()
            offline_dh.get_data('./', regex_filters, metadata_fetchers=metadata_fetchers, delimiter=",")
            train_windows, train_metadata = offline_dh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT, metadata_operations=metadata_operations)

            # Extract features from offline data
            fe = FeatureExtractor()
            feature_list = feature_list # Default feature list is HTD features.
            training_features = fe.extract_features(feature_list, train_windows, array=True)

            # Step 3: Dataset creation
            data_set = {
                'training_features': training_features,
                'training_labels': train_metadata[labels_key]
            }

            # Step 4: Create the EMG model
            model = self.model_str.get()
            if not model:
                messagebox.showerror("Model Error", "Please select a valid model type.")
                self.initialize_ui()
                return False
            print('Fitting model...')

            if self.regression_selected():
                emg_model = EMGRegressor(model=model)
                emg_model.fit(feature_dictionary=data_set)
                #emg_model.add_deadband(DEADBAND) # Add a deadband to the regression model. Value below this threshold will be considered 0.
                self.predictor = OnlineEMGRegressor(emg_model, WINDOW_SIZE, WINDOW_INCREMENT, self.odh, feature_list, std_out=True, ip=self.controller_IP, port=self.controller_PORT) # ip and port are used to send the data to the controller.
            else:
                emg_model = EMGClassifier(model=model)
                emg_model.fit(feature_dictionary=data_set)
                emg_model.add_velocity(train_windows, train_metadata[labels_key])
                self.predictor = OnlineEMGClassifier(emg_model, WINDOW_SIZE, WINDOW_INCREMENT, self.odh, feature_list, output_format='probabilities', std_out=True, ip=self.controller_IP, port=self.controller_PORT)

            # Step 5: Create online EMG model and start predicting.
            print('Model fitted and running!')
            self.predictor.run(block=False) # block set to false so it will run in a seperate process.
        except Exception as e:
            messagebox.showerror("Training Error", f"Model training or prediction failed: {e}")
            self.initialize_ui()
            return False
        
        return True
    
    def create_animation(self, transition_duration=1, hold_duration=2, rest_duration=0):
        """
        Creates animations for different motor functions used in the training protocol.
        Parameters:
        ----------
        transition_duration : float
            Duration of the transition phase from origin to endpoint of motion (in seconds).
        hold_duration : float
            Duration of the hold phase at the endpoint of motion (in seconds).
        rest_duration : float
            Duration of the rest phase after each motion (in seconds).
        """
        fps = 24  # Frames per second for the animation
        media_folder = self._get_media_folder()
        for motion_class, (x_factor, y_factor) in self.motion_classes.items(): # Called motion class but can include simultaneous motions as well.
            output_filepath = Path(media_folder, f'{motion_class}.mp4').absolute() 
            if output_filepath.exists():
                print(f'Animation file for {motion_class} already exists. Skipping creation.')
                continue

            # Generate base movement
            #base_motion = self._generate_base_signal()
            #base_motion = self._generate_trapezoid_signal(sampling_rate=24, transition_duration=transition_duration, hold_duration=hold_duration, rest_duration=rest_duration)
            base_motion = self._generate_sawtooth_signal(rise_time=transition_duration, 
                                                         hold_time= hold_duration, 
                                                         rest_time=rest_duration, 
                                                         n_repeats=1, 
                                                         sampling_rate=fps, 
                                                         amplitude=1
                                                         ) # 1 second rise time, 3 seconds rest time, 1 repeat, 24 fps
            # Apply movement transformation
            x_coords = x_factor * base_motion
            y_coords = y_factor * base_motion

            coordinates = np.stack((x_coords, y_coords), axis=1)

            #plotter = PredictionPlotter(self.axis_media_paths, real_time=False)
            #plotter.make_animation(coordinates, output_filepath=output_filepath.as_posix())
            
            #animator = MediaGridAnimator(output_filepath=output_filepath.as_posix(), show_direction=True, show_countdown=True, axis_media_paths=self.axis_media_paths, figsize=(10,10), normalize_distance=True, show_boundary=True, tpd=2)#, plot_line=True) # plot_line does not work
            #animator.plot_center_icon(coordinates, title=f'Regression Training - {mf}', save_coordinates=True, xlabel='MF 1', ylabel='MF 2')
            scatter_animator_x = ScatterPlotAnimator(output_filepath=output_filepath.as_posix(),
                                                    show_direction=True, 
                                                    show_countdown=True, 
                                                    axis_images=self.axis_media, 
                                                    figsize=(10,10), 
                                                    normalize_distance=False, 
                                                    show_boundary=True, 
                                                    fps=fps
                                                    )# ,(tpd=5 this does not make any diffrence..)#, plot_line=True) # plot_line does not work
            scatter_animator_x.save_plot_video(coordinates, title=f'Regression Training - {motion_class}', save_coordinates=True, verbose=True)
            #arrow_animator = ArrowPlotAnimator(output_filepath=output_filepath.as_posix(), show_direction=True, show_countdown=True, axis_images=self.axis_media, figsize=(10,10), normalize_distance=True, show_boundary=True, tpd=2)#, plot_line=True) # plot_line does not work
            #arrow_animator.save_plot_video(coordinates, title=f'Regression Training - {mf}', save_coordinates=True, verbose=True)

    def _generate_sawtooth_signal(self, rise_time, hold_time, rest_time, n_repeats, sampling_rate, amplitude=1):
        """
        Generate a sawtooth signal that rises linearly over 'rise_time' seconds, then rests flat for 'rest_time' seconds.

        Parameters
        ----------
        rise_time : float
            Duration of the rising edge (in seconds).
        rest_time : float
            Duration of the rest period after each rise (in seconds).
        n_repeats : int
            Number of sawtooth cycles to generate.
        sampling_rate : int
            Samples per second (Hz).
        amplitude : float
            Peak value of the signal.

        Returns
        -------
        signal : np.ndarray
            The generated sawtooth signal.
        time_vec : np.ndarray
            Corresponding time vector in seconds.
        """
        # Number of samples for rise and rest
        rise_samples = int(rise_time * sampling_rate)
        hold_samples = int(hold_time * sampling_rate)
        rest_samples = int(rest_time * sampling_rate)

        # Create one cycle: rise + rest
        rise_part = np.linspace(0, amplitude, rise_samples, endpoint=False)
        hold_part = np.full(hold_samples, amplitude)
        rest_part = np.zeros(rest_samples)

        # One full cycle
        cycle = np.concatenate([rise_part, hold_part, rest_part])

        # Repeat the cycle
        signal = np.tile(cycle, n_repeats)

        return signal
    
    def _generate_trapezoid_signal(self, sampling_rate, transition_duration, hold_duration, rest_duration):
        """ Generate a trapezoidal base signal: 0 -> 1 -> hold -> -1 -> hold -> 0 """
        num_transition = max(1, int(transition_duration * sampling_rate))
        num_hold = max(1, int(hold_duration * sampling_rate))

        ramp_up = np.linspace(0, 1, num_transition, endpoint=False)
        hold_high = np.ones(num_hold)
        ramp_down = np.linspace(1, -1, 2 * num_transition, endpoint=False)
        hold_low = -np.ones(num_hold)
        ramp_back = np.linspace(-1, 0, num_transition)
        rest = np.zeros(int(rest_duration * sampling_rate))

        signal = np.concatenate((ramp_up, hold_high, ramp_down, hold_low, ramp_back, rest))
        return signal

    def _generate_base_signal(self):
        """Generates a sinusoidal motion profile for training animation."""
        period = 3      # period of sinusoid (seconds)
        cycles = 1      # number of reps of the motorfunction
        steady_time = 2 # how long you want to stay in end position (seconds)
        rest_time = 5   # time for rest between reps (seconds)
        fps = 24        # frames per second

        coordinates = []
        # Time vectors
        t_right = np.linspace(0, period / 2, int(fps * period / 2))#, endpoint=False)  # Rising part
        t_left = np.linspace(period / 2, period, int(fps * period / 2))#, endpoint=False)  # Falling part

        # Generate sine wave motion
        sin_right = np.sin(2 * np.pi * (1 / period) * t_right)  # 0 to 1
        sin_left = np.sin(2 * np.pi * (1 / period) * t_left)  # 1 to -1

        # Create steady phases
        steady_right = np.ones(fps * steady_time)  # Hold at max (1)
        steady_left = -np.ones(fps * steady_time)  # Hold at min (-1)

        # Build full movement cycle
        one_cycle = np.concatenate([
            sin_right[0:int(len(sin_right)/2)], steady_right, 
            sin_right[int(len(sin_right)/2):], sin_left[0:int(len(sin_right)/2)], 
            steady_left, sin_left[int(len(sin_right)/2):]])
        
        # Repeat for the number of cycles
        full_motion = np.tile(one_cycle, cycles)
        coordinates.append(full_motion)  # add sinusoids
        coordinates.append(np.zeros(fps * rest_time))   # add rest time

        # Convert into 2D (N x M) array with isolated sinusoids per DOF
        coordinates = np.expand_dims(np.concatenate(coordinates, axis=0), axis=1)
        return coordinates
    
    def _get_media_folder(self):
        media_folder = self.training_media_folder.get()
        if not media_folder or not Path(media_folder).exists():
            messagebox.showerror("Invalid Path", "Media folder does not exist. Please provide a valid path.")
            self.initialize_ui()
            return
        return media_folder
    
    def _get_training_data_folder(self):
        training_data_folder = self.training_data_folder.get()
        if not training_data_folder or not Path(training_data_folder).exists():
            messagebox.showerror("Invalid Path", "Training data folder does not exist. Please provide a valid path.")
            self.initialize_ui()
            #return
        return training_data_folder
    
    def regression_selected(self):  
        return self.model_type.get() == 2
    
    def on_closing(self):
        # Clean up all the processes that have been started
        self.window.destroy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prosthesis control GUI")
    parser.add_argument('--sensors', nargs='+', type=int, default=[1,2,4,6,9],
                        help='List of sensor IDs (e.g. --sensors 0 1 2)')

    args = parser.parse_args()

    sensors_ids = args.sensors
    gui = ProsthesisControlGUI(sensor_ids=sensors_ids)