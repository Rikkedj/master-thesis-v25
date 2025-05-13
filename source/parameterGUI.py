import dearpygui.dearpygui as dpg
import time
import inspect
from _parameter_gui_panel import ParameterAdjustmentPanel # Could remove this to my own folder or something to make it clear its made by me


class ParameterAdjustmentGUI: 
    '''
    The GUI for adjusting the parameters for the predictor (either the classifier or the regressor) and the motor controller.

    Parameters
    ----------
    online_data_handler : OnlineDataHandler
        The online data handler that is used to get the data from the streamer. 
    params : dict
        The parameters that are to be adjusted for the controller. This is a dictionary with the following keys:
            'window_size' : int
                The window size for when training the prediction model.
            'window_increment': int
                The window increment for when training the prediction model.
            'thr_mf1': float
                The threshold for the first motor function.
            'thr_mf2': float
                The threshold for the second motor function.
            'gain_mf1': float
                The gain for the first motor function.
            'gain_mf2': float
                The gain for the second motor function.
            'deadband': float
                The deadband for the prediction model. If the prediction is within this range, the prediction will be set to 0.
    axis_media : dict
        Dictionary mapping compass directions to media (image or video). Media will be displayed in the corresponding compass direction (i.e., 'N' correponds to the top of the image).
        Valid keys are 'NW', 'N', 'NE', 'W', 'E', 'SW', 'S', 'SE'. If None, no media will be displayed. The media shows the motor functions that are being predicted.
    model_str : str
        The prediction model used, given as string.
    training_data_folder : str
        The folder where the training data is stored. This is used to load the training data for the prediction model.
    feature_list : list
        The list of features that are used for the prediction model.
    regression_selected : bool
        If True, the regression model is selected. This is used to determine which model to use for prediction.
    width : int
        The width of the GUI window.
    height : int
        The height of the GUI window.
    debug : bool CHECK THIS PUT. DON'T KNOW IF ITS NECESSEARY
        If True, the GUI will run in debug mode. This means that the GUI will not be closed when the user clicks the close button. Instead, the GUI will be stopped and the program will continue to run.
    
    '''
    def __init__(self, 
                 online_data_handler, 
                 params,
                 axis_media=None,
                 model_str=None,
                 training_data_folder = './data/', 
                 feature_list=None,
                 regression_selected=False,
                 width=1700,
                 height=1080,
                 debug=False, # Not shure if this is necessary
                 clean_up_on_kill=False):
        
        self.width = width
        self.height = height
        self.online_data_handler = online_data_handler
        self.axis_media = axis_media
        self.model_str = model_str
        self.params = params
        self.regression_selected = regression_selected # Bool that tells if the regression model is selected
        self.training_data_folder = training_data_folder
        self.feature_list = feature_list 
        
        self.debug = debug # Usikker på hva denne er til, kan være den ikke trengs
        self.clean_up_on_kill = clean_up_on_kill # Not shure what this is either, but is used in the GUI class
        

    def start_gui(self):
        """
        Opens the Model Configuration GUI
        """
        self._window_init(self.width, self.height, self.debug)

    def _window_init(self, width, height, debug=False):
        dpg.create_context()
        dpg.create_viewport(title="Adjust Post-Training Parameters",
                            width=width,
                            height=height)
        dpg.setup_dearpygui()
        

        self._file_menu_init()

        dpg.show_viewport()
        dpg.set_exit_callback(self._on_window_close)
        
        if debug:
            dpg.configure_app(manual_callback_management=True)
            while dpg.is_dearpygui_running():
                jobs = dpg.get_callback_queue()
                dpg.run_callbacks(jobs)
                dpg.render_dearpygui_frame()
        else:
            dpg.start_dearpygui()
        
        dpg.start_dearpygui()
        dpg.destroy_context()

    def _file_menu_init(self):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="Exit"):
                dpg.add_menu_item(label="Exit", callback=self._exit_window_callback) # Need to add a callback to close the window
                
            with dpg.menu(label="Model"):
                dpg.add_menu_item(label="Adjust Parameters", callback=self._adjust_param_callback, show=True)
        #self._adjust_param_callback()
                
    def _adjust_param_callback(self):
        panel_arguments = list(inspect.signature(ParameterAdjustmentPanel.__init__).parameters) 
        passed_arguments = {i: self.params[i] for i in self.params.keys() if i in panel_arguments} 
        self.pap = ParameterAdjustmentPanel(**passed_arguments, gui=self, training_data_folder=self.training_data_folder) 
        self.pap.spawn_configuration_window()

    def _exit_window_callback(self):
        #self.clean_up_on_kill = True
        dpg.stop_dearpygui()

    def _on_window_close(self):
        #if self.clean_up_on_kill:
        print("Window is closing. Performing clean-up...")
        if 'streamer' in self.params.keys(): # It isn't, find another place to store streamer
            self.params['streamer'].signal.set()
        time.sleep(3)
    