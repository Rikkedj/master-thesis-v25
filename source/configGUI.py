import dearpygui.dearpygui as dpg
import time
import inspect
from _gui._config_panel import ModelConfigPanel # Could remove this to my own folder or something to make it clear its made by me


class ML_GUI: # Kalle det CONTROL_SYSTEM_GUI elns?
    '''
    The GUI for configuring the prediction model, either the classifier or the regressor, and the controller.

    Parameters
    ----------
    online_data_handler : OnlineDataHandler
        The online data handler that is used to get the data from the streamer. 
    args : dict
        The arguments that are used to configure the controller. This is a dictionary with the following keys:
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
    width : int
        The width of the GUI window.
    height : int
        The height of the GUI window.
    debug : bool CHECK THIS PUT. DON'T KNOW IF ITS NECESSEARY
        If True, the GUI will run in debug mode. This means that the GUI will not be closed when the user clicks the close button. Instead, the GUI will be stopped and the program will continue to run.
    regression_selected : bool
        If True, the regression model is selected. This is used to determine which model to use for prediction.

    '''
    def __init__(self, 
                 online_data_handler, 
                 args,
                 axis_media=None,
                 model_str=None,
                 training_data_folder = './data/', # added 22.04 -> think this is chosen in main window
                 width=1700,
                 height=1080,
                 debug=False, # Not shure if this is necessary
                 regression_selected=False,
                 clean_up_on_kill=False):
        
        self.width = width
        self.height = height
        self.online_data_handler = online_data_handler
        self.axis_media = axis_media
        self.model_str = model_str
        self.args = args
        self.regression_selected = regression_selected # Bool that tells if the regression model is selected, gotten from function in TrainingProtocol
        self.training_data_folder = training_data_folder
        
        self.debug = debug # Usikker på hva denne er til, kan være den ikke trengs
        self.clean_up_on_kill = clean_up_on_kill # Not shure what this is either, but is used in the GUI class
        

    def start_gui(self):
        """
        Opens the Model Configuration GUI
        """
        self._window_init(self.width, self.height, self.debug)

    def _window_init(self, width, height, debug=False):
        dpg.create_context()
        dpg.create_viewport(title="Configure Machine Learning Model",
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
                dpg.add_menu_item(label="Configure Model", callback=self._model_config_callback, show=True)
        self._model_config_callback()
                
    def _model_config_callback(self):
        panel_arguments = list(inspect.signature(ModelConfigPanel.__init__).parameters) # Get the arguments of the ModelConfigPanel class
        passed_arguments = {i: self.args[i] for i in self.args.keys() if i in panel_arguments} 
        self.mcp = ModelConfigPanel(**passed_arguments, gui=self, training_data_folder=self.training_data_folder) # Create the ModelConfigPanel object
        self.mcp.spawn_configuration_window()

    def _exit_window_callback(self):
        #self.clean_up_on_kill = True
        dpg.stop_dearpygui()

    def _on_window_close(self):
        #if self.clean_up_on_kill:
        print("Window is closing. Performing clean-up...")
        if 'streamer' in self.args.keys():
            self.args['streamer'].signal.set()
        time.sleep(3)
    