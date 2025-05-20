import numpy as np
from pathlib import Path
from PIL import Image as PILImage
import time
import csv
import threading
import os

from libemg.training import TrainingPrompt
from libemg.streamers import delsys_streamer
from libemg.data_handler import OnlineDataHandler

def sample_emg_data(sampling_rate, duration, online_data_handler, filename):
    """
    Sample EMG data from the online data handler.

    Parameters:
    - sampling_rate: Sampling rate in Hz.
    - duration: Duration in seconds.
    - online_data_handler: The online data handler to sample data from.

    Returns:
    - numpy.ndarray: Sampled EMG data.
    """
    odh.reset()
    data = []
    while time.perf_counter_ns() < duration * 1e9:
        # Sample data from the online data handler
        vals, count = online_data_handler.get_data()
        data.append(vals)
        time.sleep(1 / sampling_rate)
    
    with open(filename, "w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for row in data:
                writer.writerow(row)

if __name__ == "__main__":
    # mf = {
    #     'hand_open': (1, 0),            # Movement along x-axis
    #     'hand_close': (-1, 0),          # Movement along x-axis
    #     'pronation': (0, 1),            # Movement along y-axis
    #     'supination': (0, -1),          # Movement along y-axis
    #     }
    
    # axis_media =  {
    #         'N': PILImage.open(Path('media/images/gestures', 'pronation.png')),
    #         'S': PILImage.open(Path('media/images/gestures', 'supination.png')),
    #         'E': PILImage.open(Path('media/images/gestures', 'hand_open.png')),
    #         'W': PILImage.open(Path('media/images/gestures', 'hand_close.png')),
    #         'NE': PILImage.open(Path('media/images/gestures', 'open_pronate.png')), 
    #         'NW': PILImage.open(Path('media/images/gestures', 'close_pronate.png')),
    #         'SE': PILImage.open(Path('media/images/gestures', 'open_supinate.png')),
    #         'SW': PILImage.open(Path('media/images/gestures', 'close_supinate.png'))
    #     }
    
    # rise_duration = 2  # seconds
    # steady_state_duration = 1  # seconds
    # amplitude = 1  # amplitude of the signal
    # rest_between_reps = 1  # seconds
    # num_reps = 3  # number of repetitions
    # tp = TrainingPrompt(motor_functions=mf,
    #                     axis_media=axis_media,
    #                     rise_duration=rise_duration, 
    #                     steady_state_duration=steady_state_duration, 
    #                     amplitude=amplitude, 
    #                     rest_between_reps=rest_between_reps, 
    #                     num_reps=3, 
    #                     sampling_rate=24
    #                     )
    

    streamer, sm = delsys_streamer(channel_list=[2,5,6,10]) # returns streamer and the shared memory object, need to give in the active channels number -1, so 2 is sensor 3
    # Create online data handler to listen for the data
    odh = OnlineDataHandler(sm)
    filepath = './test-data/18-05/hand_open'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    #sample_thread = threading.Thread(target=sample_emg_data, args=(24, 10, odh, filename))
    #sample_thread.start()
    start_time = time.perf_counter_ns()
    print("Sampling EMG data...")
    #odh.analyze_hardware(analyze_time=10)
    odh.visualize()
    #odh.log_to_file(file_path=filepath, timestamps=True, block=False)
    # with open(filepath, "w", newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     # Write header
    #     writer.writerow(["Timestamp"] + [f"Channel {i}" for i in range(1, 17)])
    # while (time.perf_counter_ns()-start_time)/1e9 < 30:
    #     val, count = odh.get_data()
    #     if (int((time.perf_counter_ns()-start_time)/1e9)) == 10: 
    #         print("10 seconds passed")
    #     if (time.perf_counter_ns()-start_time)/1e9 >= 20:
    #         print("20 seconds passed")

    # #     time.sleep(0.01)
    # odh.stop_log()
    # print("Finished sampling EMG data.")