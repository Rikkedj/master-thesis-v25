import time
import sys
from pathlib import Path
import cv2
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from functools import partial

import warnings
from collections import deque
import os
from matplotlib.animation import FuncAnimation
from functools import partial
import numpy as np
import threading
from queue import Empty

from libemg.environments.controllers import  RegressorController
from libemg.animator import MediaGridAnimator
from libemg.data_handler import OfflineDataHandler
from libemg.feature_extractor import FeatureExtractor

# Communication with the controller
UDP_IP = "127.0.0.1"
UDP_PORT = 5005


def plot_emg_data_and_features(raw_data, features_dict, class_names_dict, window_size, window_increment, sampling_rate=2000, title="EMG and Features Visualization", save_path=None, savefig=False):
    """
    Plot EMG raw data and extracted features, color-coded by class.
    
    Parameters:
    ----------
    raw_data: list of ndarray
        List of ndarray of shape (samples, channels) of one segment of EMG data. This could be a repetition done in the system training. For instance odh.data
        Example: [ np.array([[0.1, 0.2], [0.15, 0.25], ..., n_samples]), np.array([[0.3, 0.4], [0.35, 0.45], ..., n_samples]), ..., n_segments ]  
    features_dict: dict
        Dictionary where each key is a feature name and the value is a numpy array of shape (num_windows, num_channels).
        Example: { "MAV": np.array([[0.1, 0.2], [0.15, 0.25], ..., n_windows]), "RMS": np.array([[0.3, 0.4], [0.35, 0.45], ..., n_windows]) }
    class_names_dict: dict 
        Dictionary where keys are name of the class, and values are a list of the corresponding data segments in raw_data.
        Example: { "hand_open": [0, 1, 2], "hand_close": [3, 4, 5] } given raw_data = [segment_0, segment_1, ..., segment_6]
    window_size: int
        Size of the window used to extract features, in samples (200 samples -> 100 ms at 2000 Hz)
    window_increment: int
        The number of samples that advances before next window.
    sampling_rate: int
        Samples per second for raw_data (default: 2000 Hz)
    save_path: str or Path, optional
        Path to save the figure. If None, the figure will not be saved.
    savefig: bool, optional
        If True, the figure will be saved to save_path. If False, the figure will not be saved.
    """
    total_num_segments = len(raw_data) # The number of segments, in our case the number of repetitions in system training
    num_samples, num_channels = raw_data[0].shape
    num_features = len(features_dict)

    cmap = cm.get_cmap('Pastel1')  # or 'tab10', 'Set2', etc.
    fmap = cm.get_cmap('tab10')  # Feature colors, can be customized
    class_names = list(class_names_dict.keys())
    class_colors = [cmap(i) for i in range(len(class_names_dict))]
    class_colors = dict(zip(class_names, class_colors))
    feature_colors = [fmap(i + 1) for i in range(num_features)]  # Colors for features, can be customized
    # Create figure with subplots
    fig, axs = plt.subplots(num_features + 1, num_channels, figsize=(18, 4 * (num_features + 1)), sharex=True)
    fig.suptitle(title, fontsize=16)  # Title for the entire figure
    fig.text(0.5, 0.90, f"Reps per class: {total_num_segments//len(class_names_dict)}", ha='center', fontsize=10)

    # Plot raw data with background shading by class
    ax_emg = axs[0]
    ax_emg[0].set_ylabel("EMG")

    # Track time
    current_time = 0
    class_patches = []
    # --- Global y-limits ---
    emg_global_min, emg_global_max = float('inf'), float('-inf')
    feature_global_minmax = {f_name: [float('inf'), float('-inf')] for f_name in features_dict}

    for emg_segment_data in raw_data:
        emg_global_min = min(emg_global_min, emg_segment_data.min())
        emg_global_max = max(emg_global_max, emg_segment_data.max())

    for f_name, f_data in features_dict.items():
        feature_global_minmax[f_name][0] = min(feature_global_minmax[f_name][0], f_data.min())
        feature_global_minmax[f_name][1] = max(feature_global_minmax[f_name][1], f_data.max())

    for segment_idx, emg_segment_data in enumerate(raw_data):
        num_samples_segment = emg_segment_data.shape[0]
        duration_segment = num_samples_segment / sampling_rate
        t_emg = np.linspace(current_time, current_time + duration_segment, num_samples_segment)
        # Determine class label for this repetition
        class_label = next((cls for cls, reps in class_names_dict.items() if segment_idx in reps), "Unknown")
        color = class_colors.get(class_label, (0.5, 0.5, 0.5, 0.2))  # fallback gray
        # Shade background for class
        
        for ch in range(num_channels):
            ax_emg[ch].plot(t_emg, emg_segment_data[:, ch], label=f"Ch {ch+1}", linewidth=0.8, color="steelblue") # + ch * 300
            ax_emg[ch].axvspan(current_time, current_time + duration_segment, color=color, alpha=0.3)
            ax_emg[ch].set_ylim(emg_global_min*1.01, emg_global_max*1.01)  # Set global y-limits for EMG
            ax_emg[ch].set_title(f"Ch {ch+1}")
       

        for idx, (f_name, f_data) in enumerate(features_dict.items()):
            tot_windows = f_data.shape[0]
            n_windows_per_segment = tot_windows // total_num_segments
            #n_windows_current_segment = int(round((num_samples_segment - window_size) / window_increment + 1))
            samples_per_segment = n_windows_per_segment * window_increment + window_size
            t_feat = current_time + np.arange(tot_windows) * window_increment / sampling_rate + (window_size / (2 * sampling_rate)) # Center the feature windows in time, i.e. chooese the feature to be at the center of the window
            t_current_feat = current_time + np.arange(n_windows_per_segment) * window_increment / sampling_rate + (window_size / (2 * sampling_rate))

            for ch in range(num_channels):
                axs[idx + 1][ch].plot(t_current_feat, f_data[segment_idx*n_windows_per_segment:(segment_idx+1)*n_windows_per_segment, ch], label=f"Ch {ch+1}" if segment_idx == 0 else "", alpha=0.8, color=feature_colors[idx])
                axs[idx + 1][ch].axvspan(current_time, current_time + duration_segment, color=color, alpha=0.3)
                axs[idx + 1][ch].set_ylim(feature_global_minmax[f_name][0]*1.01, feature_global_minmax[f_name][1]*1.01)  # Set global y-limits for features
                axs[-1][ch].set_xlabel("Time (s)")

            axs[idx + 1][0].set_ylabel(f_name)
            
        current_time += duration_segment  # Update current time for next segment

     # Add legend for classes in upper right outside the plot
    for cls, col in class_colors.items():
        class_patches.append(mpatches.Patch(color=col, label=cls))

    # Place legend outside the figure area (top right)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)  # Adjust top margin to make space for the legend and title  
    fig.legend(handles=class_patches, 
               loc='upper right', 
               bbox_to_anchor=(0.95, 0.98), 
               ncol=len(class_patches), 
               borderaxespad=0.5, 
               title="Class labels")

    if save_path and savefig is True:
        save_path = Path(save_path)
        # Ensure save_path exists, creating it if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
    plt.show()


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
        self.axis_media = axis_media_paths # This is the dictionary with the images/videos to be displayed in the plot
        self.loaded_media = {}  # Stores {'location': ('type', media_object)}
        self.media_artists = {} # Stores {'location': AxesImage artist}
        self.video_caps = {}    # Stores {'location': cv2.VideoCapture} for cleanup

        # Target plot for comparing with predictions
        self.target_sequence = []  # List of (mf1, mf2) targets
        self.target_index = 0
        self.predictions = []
        self.targets = []

        self.history = deque(maxlen=1000)

        self._load_all_media() # Load media during initialization

    def define_target_sequence(self):
        # Example motor function positions (x, y) for different gestures
        gesture_map = {
            "rest": (0, 0),
            "open": (1, 1),
            "close": (-1, -1),
            "pronate": (1, -1),
            "supinate": (-1, 1),
            "open+pronate": (1.5, 0),
            "pronate+close": (0, -1.5),
            "close+supinate": (-1.5, 0),
            "supinate+open": (0, 1.5)
        }

        sequence_labels = [
            "open", "rest", "open+pronate", "rest", "pronate", "rest",
            "pronate+close", "rest", "close", "rest", "close+supinate",
            "rest", "supinate+open", "rest"
        ]

        self.target_sequence = [gesture_map[g] for g in sequence_labels]

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
    
    # --- Set Main Plot Limits ---
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

    # --- Drawing Functions ---
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

class VideoElement:
    def __init__(self, ax, video_path, playback_speed=0.05):
        self.ax = ax
        self.video_path = str(video_path)
        self.playing = True
        self.cap = cv2.VideoCapture(video_path) if video_path else None
        self.playback_speed = playback_speed
        self.frame_counter = 0
        self.lock = threading.Lock()

        # if self.cap and self.cap.isOpened():
        #     ret, frame = self.cap.read()
        #     if ret:
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         self.image = self.ax.imshow(frame)
        # else:
        #     self.image = self.ax.imshow(np.zeros((10,10,3), dtype=np.uint8))  # placeholder

    # def update(self):
    #     if not self.playing or not self.cap or not self.cap.isOpened():
    #         return

    #     self.frame_counter += 1
    #     if self.frame_counter >= self.playback_speed:
    #         ret, frame = self.cap.read()
    #         if not ret:
    #             self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #             ret, frame = self.cap.read()
    #         if ret:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             self.image.set_data(frame)
    #         self.frame_counter = 0
    def update_thread(self, skip_frames=1):
        if not self.playing or not self.cap or not self.cap.isOpened():
            return

        # ret, frame = self.cap.read()
        # if not ret:
        #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #     ret, frame = self.cap.read()
        # if ret:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     self.image.set_data(frame)

        while self.cap.isOpened():
            for _ in range(skip_frames - 1):
                self.cap.read()  # Discard

            ret, f = self.cap.read()
            if not ret:
                break
            with self.lock:
                frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

        self.cap.release()

    def pause(self):
        self.playing = False

    def play(self):
        self.playing = True

    def reset(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        if self.cap:
            self.cap.release()