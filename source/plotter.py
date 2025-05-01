import time
import sys
from pathlib import Path
import cv2
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings
from collections import deque
import os
from matplotlib.animation import FuncAnimation
from functools import partial
import numpy as np
from multiprocessing import Process, Queue, Manager
import threading
from libemg.environments.controllers import  RegressorController
from libemg.animator import MediaGridAnimator

# Communication with the controller
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

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
        # self.config = config
        # self.pred_queue = pred_queue
        self.SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
        self.SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

        self.axis_media_paths = axis_media_paths
        self.video_elements = {} # Stores VideoElement objects for each video location
        self.image_artists = {} # Stores image artists for each location

        # self.loaded_media = {}  # Stores {'location': ('type', media_object)}. 'Location' is the location on the grid (e.g., 'N', 'S', etc.)
        # self.media_artists = {} # Stores {'location': AxesImage artist}
        # self.video_caps = {}    # Stores {'location': cv2.VideoCapture}
        # self.video_info = {}    # Stores {'location': {'total_frames': int, 'last_frame_idx': int}}
        # self.video_playing = False
        # self.anim_interval_ms = 50
        # self.video_elements = {} # Stores VideoElement objects for each video location

        self.history = deque(maxlen=1000) # Keeps track of history for scaling the main plot
        self.tale = [] # Keeps track of the last few points for the 'tale' effect. Could be replaced with an arrow.
        self.current_pred = np.array([0.0, 0.0]) # Store the latest prediction coords. NOTE! May not be necessary
        self.current_direction = 'NE' # Need this to know which video to play
        
        #self._load_all_media() # Load media during initialization
       

    def _load_media(self, location, file_path):
        """Loads media (image or video) from a file path. If video, store info for playback speed."""
        #file_path = Path(file_path)
        if not file_path.exists():
            warnings.warn(f"Media file not found at {file_path}. Skipping {location}.")
            return None, None # Return None if file doesn't exist

        ext = file_path.suffix.lower()

        try:
            if ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                img = PILImage.open(file_path)
                # Ensure image is in RGB or RGBA for matplotlib
                if img.mode not in ['RGB', 'RGBA']: img = img.convert('RGB')
                img = np.array(img)  # Convert to numpy array for matplotlib
                img_artist = self.axes[location].imshow(img) # Display the image in the corresponding axis
                self.image_artists[location] = img_artist # Store the image artist
                #self.video_dims[location] = img.size # Store image dimensions. NOTE! Could be done later
                #return 'image', img
            
            elif ext in self.SUPPORTED_VIDEO_EXTENSIONS:
                ve = VideoElement(self.axes[location], file_path) # Create a VideoElement for the video
                if ve.cap: # Only store if successfully opened
                    self.video_elements[location] = ve
                    print(f"  Created VideoElement for {location}: {os.path.basename(file_path)}")
                else:
                        print(f"  Failed to create VideoElement for {location}")
                # cap = cv2.VideoCapture(file_path)
                # if not cap.isOpened():
                #     warnings.warn(f"Could not open video file: {file_path}. Skipping {location}.")
                #     return None, None
                
                # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # if total_frames <= 0:
                #     warnings.warn(f"Could not get frame count or video empty for {file_path}. Skipping {location}.")
                #     cap.release()
                #     return None, None
                
                # ret, frame = cap.read() # Read first frame for initial display
                # if not ret:
                #     warnings.warn(f"Could not read first frame from video: {file_path}. Skipping {location}.")
                #     cap.release()
                #     return None, None
                
                # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # self.video_caps[location] = cap
                # # Store total frames and initialize last displayed frame index to -1 (or 0)
                # self.video_info[location] = {'total_frames': total_frames, 'last_frame_idx': 0} # Start at frame 0
                
                # fps = cap.get(cv2.CAP_PROP_FPS)
                # if fps == 0:  # Fallback if FPS is not detected
                #     fps = 30.0
                # video_interval = 1000 / fps  # in milliseconds

                # # Calculate how often we should update frames given the FuncAnimation interval
                # update_every_n = max(1, round(video_interval / self.anim_interval_ms))  # <-- store this

                # self.video_update_counters[location] = {
                #     "counter": 0,
                #     "update_every_n": update_every_n
                # }

                # # Convert BGR (OpenCV default) to RGB (Matplotlib default)
                # self.video_update_counters[location] = 0 # Initialize counter for video frames
                # return 'video', frame_rgb # Return the first frame for initial display
            else:
                warnings.warn(f"Unsupported file extension '{ext}' for {file_path}. Skipping {location}.")
                return None, None
        
        except Exception as e:
            warnings.warn(f"Error loading media {file_path} for {location}: {e}")
            # Ensure capture is released if error occurred during video loading
            # if 'cap' in locals() and isinstance(cap, cv2.VideoCapture) and cap.isOpened():
            #     cap.release()
            # return None, None

    def _load_all_media(self):
        """Loads all media specified in axis_media_paths."""
        print("Loading axis media...")
        for location, path in self.axis_media_paths.items():
            media_type, media_data = self._load_media(location, path)
            if media_type:
                self.loaded_media[location] = (media_type, media_data)
                print(f"  Loaded {media_type} for {location}: {os.path.basename(path)}")
            else:
                 print(f"  Failed to load media for {location}: {os.path.basename(path)}")
        print("Finished loading axis media.")
                    
    def _initialize_plot(self, config):
        '''Initialize the plot with a grid layout and configure axes for media and main plot. '''
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
        self.ax_main.axis('equal') # Ensure aspect ratio is equal
      
        # Plot of predictions
        self.tale_plot, = self.ax_main.plot([], [], 'o', color='gray', markersize=4, alpha=0.5, label='Tale')
        self.current_plot, = self.ax_main.plot([], [], 'o', color='red', markersize=8, markeredgecolor='black', label='Current Prediction')
        
        # Create a circle for the deadband
        self.circle = plt.Circle((0, 0), config["__mc_deadband"], color='r', fill=False, linestyle='dashed')
        self.ax_main.add_patch(self.circle)
        # Threshold lines
        self.threshold_lines = [
            self.ax_main.plot([], [], 'b')[0],
            self.ax_main.plot([], [], 'b')[0],
            self.ax_main.plot([], [], 'b')[0],
            self.ax_main.plot([], [], 'b')[0]
        ]

        # --- Configure Surrounding Media Axes ---
        # could add some resizing logic here to make sure the images are not too big
        for location, ax in self.axes.items():
            if location == 'C': continue # Skip center plot

            # Hide axis labels for surrounding figures
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off') # Turn off axis decorations

            path = self.axis_media_paths.get(location)
            if path and path.exists(): self._load_media(location, path)
            # if location in self.loaded_media:
            #     media_type, media_data = self.loaded_media[location]
            #     # Display the image or the first frame of the video
            #     #### Resize media if implemented
            #     if media_type == 'video':
            #         # Create a VideoElement for the video
            #         video_element = VideoElement(ax, self.axis_media_paths[location])
            #         self.video_elements[location] = video_element

            #     img_artist = ax.imshow(media_data)
            #     self.media_artists[location] = img_artist 

    def _calculate_range(self):
        """Calculates a dynamic range for plot limits based on history."""
        if len(self.history) > 1:
            history_array = np.array(self.history)
            # Avoid numpy warnings if history contains NaN or Inf
            if np.any(~np.isfinite(history_array)): return 1.0 # Default range if bad data

            x_min, x_max = np.min(history_array[:, 0]), np.max(history_array[:, 0])
            y_min, y_max = np.min(history_array[:, 1]), np.max(history_array[:, 1])
            range_val = max(x_max - x_min, y_max - y_min) #, 0.5) * 1.2 # Add some padding NOTE! May add the padding
            return max(range_val, 1)
        return 1
    
    def _update_plot_limits(self):
        """Updates the main plot limits based on data range."""
        plot_range = self._calculate_range()*1.2 # Add some padding to the range
        center_x, center_y = 0, 0 # Center the plot around the origin or the data mean if preferred

        self.ax_main.set_xlim(center_x - plot_range, center_x + plot_range)
        self.ax_main.set_ylim(center_y - plot_range, center_y + plot_range)
    
    def _draw_threshold(self, config):
        thresh_rad_x = np.deg2rad(config["__mc_alpha_mf1"])
        x_vals = np.array([-1.5, 1.5]) # Hardcoded values for the x-axis limits. TODO! Change to dynamic
        y_vals1, y_vals2 = np.tan(thresh_rad_x) * x_vals, -np.tan(thresh_rad_x) * x_vals
        
        thresh_rad_y = np.deg2rad(config["__mc_alpha_mf2"])
        y_vals = np.array([-1.5, 1.5]) # Hardcoded values for the y-axis limits. TODO! Change to dynamic
        x_vals1, x_vals2 = np.tan(thresh_rad_y) * y_vals, -np.tan(thresh_rad_y) * y_vals
        
        self.threshold_lines[0].set_data(x_vals, y_vals1)
        self.threshold_lines[1].set_data(x_vals, y_vals2)
        self.threshold_lines[2].set_data(x_vals1, y_vals)
        self.threshold_lines[3].set_data(x_vals2, y_vals)
    
    def _draw_deadband(self, config):
        ''' Updates the deadband circle, given as percent of the range of the data. '''
        self.circle.set_radius(config["__mc_deadband"] * self._calculate_range())
    
    def _draw_prediction(self, config, pred_queue):
        if not pred_queue.empty():
            pred = pred_queue.get()
            pred[0] *= config["__mc_gain_mf1"]
            pred[1] *= config["__mc_gain_mf2"]
            self.history.append(pred.copy())
            self.tale.append(pred.copy())
            self.tale = self.tale[-5:] # Only keep the last 5 points for the 'tale' effect. Could be done as input, or with arrow
            tale_array = np.array(self.tale)
            
            if tale_array.shape[0] > 1:
                self.tale_plot.set_xdata(tale_array[:, 0])
                self.tale_plot.set_ydata(tale_array[:, 1])
            
            self.current_plot.set_xdata(tale_array[-1:, 0])
            self.current_plot.set_ydata(tale_array[-1:, 1])


    def _update_media_frames(self):
        """Updates frames for any video media."""
        # Iterate through all managed video elements
        for location, video_element in self.video_elements.items():
            if location == self.current_direction:
                # This is the active video
                video_element.play()  # Ensure it's set to play
                video_element.update() # Update its frame based on its internal counter
            else:
                # This video should be paused
                video_element.pause()
        # for location, cap in self.video_caps.items():
        #     # Check if the video is playing and the location is valid 
        #     if not cap.isOpened() or location not in self.video_info: continue 
        #     if location != self.current_direction:
        #         self.video_elements[location].pause()
        #     else:
        #         self.video_elements[location].play()
        #         self.video_elements[location].update() # Update the video frame


    def _update_main_plot(self, config, pred_queue):
        self._draw_threshold(config=config)
        self._draw_deadband(config=config)
        self._draw_prediction(config=config, pred_queue=pred_queue)

    def update(self, frame, config, pred_queue):
        if not config["running"]:
            print("Stopping animation...")
            if hasattr(self, 'anim') and self.anim.event_source:
                 self.anim.event_source.stop()
            self.close() # Release resources
            # plt.close(self.fig) # Closing handled in self.close()
            return [] # Return empty list of artists
        
        self._update_main_plot(config=config, pred_queue=pred_queue)
        #self._update_plot_limits()
        self._update_media_frames()

        # Efficiently redraw necessary parts
        self.fig.canvas.draw_idle()
        self.ax_main.relim()
        self.ax_main.autoscale_view() # self.ax updatet initilize function27.03 11:52

        # Return list of artists that have been updated
        # If blit=True, this is crucial. If blit=False, less so, but good practice.
        updated_artists = [self.tale_plot, self.current_plot, self.circle]
        updated_artists.extend(self.threshold_lines)
        #updated_artists.extend(self.media_artists.values())
        # Include axes if limits changed, but return artists for blitting
        # If not using blit=True, returning [] or the list is often fine.
        return updated_artists
    
    def run(self, config, pred_queue):
        #self._initialize_plot(config)
        self._initialize_plot(config)
        #self._add_images()
        self.anim = FuncAnimation(self.fig, partial(self.update, config=config, pred_queue=pred_queue), interval=10, blit=False, cache_frame_data=False, repeat=False)
        plt.tight_layout()
        plt.show()
        # Cleanup after window is closed (optional, depends if run() is the end)
        #self.close()

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

    # @staticmethod  ## NOTE! Made by Google AI studio. Not tested yet. May need modifications
    # def preprocess_video_speed(input_path: Path, output_path: Path, speed_factor: float = 0.20, ffmpeg_path: str = "ffmpeg", overwrite: bool = False):
    #     """
    #     Uses ffmpeg to create a sped-up version of a video file.

    #     Args:
    #         input_path: Path to the original video file.
    #         output_path: Path where the sped-up video file will be saved.
    #         speed_factor: Factor to change the PTS. < 1.0 speeds up, > 1.0 slows down.
    #         ffmpeg_path: Path to the ffmpeg executable (or just 'ffmpeg' if in PATH).
    #         overwrite: If True, overwrite the output file if it exists.

    #     Returns:
    #         Path object to the output file if successful or if it already existed (and not overwriting),
    #         None otherwise.
    #     """
    #     input_path = Path(input_path)
    #     output_path = Path(output_path)

    #     if not input_path.is_file():
    #         print(f"Error: Input video file not found: {input_path}")
    #         return None

    #     if output_path.exists() and not overwrite:
    #         print(f"Skipping pre-processing: Output file already exists: {output_path}")
    #         return output_path # Return existing path

    #     # Ensure output directory exists
    #     output_path.parent.mkdir(parents=True, exist_ok=True)

    #     command = [
    #         ffmpeg_path,
    #         "-y" if overwrite else "-n",
    #         "-i", str(input_path),
    #         "-vf", f"setpts={speed_factor:.4f}*PTS",
    #         "-an", # Remove audio
    #         str(output_path)
    #     ]

    #     print(f"Running ffmpeg: {' '.join(shlex.quote(str(c)) for c in command)}")
    #     start_time = time.time()
    #     try:
    #         result = subprocess.run(command, check=False, capture_output=True, text=True, encoding='utf-8', errors='replace')
    #         duration = time.time() - start_time

    #         if result.returncode == 0:
    #             if output_path.exists(): # Double check file was created
    #                 print(f"Successfully created sped-up video: {output_path} (took {duration:.2f}s)")
    #                 return output_path
    #             else:
    #                 # This case might happen if ffmpeg exits 0 but didn't create the file for some reason
    #                 print(f"Error: ffmpeg reported success but output file not found: {output_path}")
    #                 print("--- ffmpeg stdout ---")
    #                 print(result.stdout)
    #                 print("--- ffmpeg stderr ---")
    #                 print(result.stderr)
    #                 return None

    #         else:
    #             print(f"Error: ffmpeg failed (return code {result.returncode}) for {input_path}.")
    #             print("--- ffmpeg stdout ---")
    #             print(result.stdout)
    #             print("--- ffmpeg stderr ---")
    #             print(result.stderr)
    #             if output_path.exists():
    #                 try: output_path.unlink()
    #                 except OSError: pass
    #             return None

    #     except FileNotFoundError:
    #         print(f"Error: '{ffmpeg_path}' command not found. Make sure ffmpeg is installed and in your PATH.")
    #         return None
    #     except Exception as e:
    #         print(f"An unexpected error occurred during ffmpeg execution: {e}")
    #         return None

def predict(pred_queue):
    """
    Simulates a prediction process by generating random predictions and putting them in the queue.
    """
    controller = RegressorController(ip=UDP_IP, port=UDP_PORT)

    while True:
        pred = controller.get_data(["predictions"])
        if pred is not None:
            pred_queue.put(pred)
        
        time.sleep(0.1)  # Simulate time delay between predictions

if __name__ == "__main__":
    config = {
        "__mc_deadband": 0.1,
        "__mc_gain_mf1": 1.0,
        "__mc_gain_mf2": 1.0,
        "__mc_alpha_mf1": 45, # degrees
        "__mc_alpha_mf2": 45, # degrees
        "running": True
    }
    axis_media_paths = {
            'N': Path('media/images', 'pronation.png'),
            'S': Path('media/images', 'supination.png'),
            'E': Path('media/images', 'hand_open.png'),
            'W': Path('media/images', 'hand_close.png'),
            'NE': Path('media/videos', 'IMG_4930.MOV'),  # store path instead
            'NW': Path('media/videos', 'IMG_4931.MOV')
        }
    
    # plotter = PredictionPlotter(axis_media_paths=axis_media_paths)
    # pred_queue = Queue()
   
    # pred_thread = threading.Thread(target=predict, args=(pred_queue,), daemon=True) # Daemon thread will exit when main program exits
    # pred_thread.start()
    # plotter.run(config=config, pred_queue=pred_queue) # Example queue for testing
    # pred_thread.join() # Wait for the prediction thread to finish (if needed)
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

    
    animator = MediaGridAnimator(axis_media_paths=axis_media_paths)
    animator.plot_center_icon(coordinates=coordinates)