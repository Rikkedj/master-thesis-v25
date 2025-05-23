
from libemg.streamers import delsys_streamer 
from libemg.data_handler import OnlineDataHandler

if __name__ == "__main__":
    # Generate the raw EMG signal
    streamer, sm = delsys_streamer(channel_list=[0,1,4,5]) # returns streamer and the shared memory object, need to give in the active channels number -1, so 2 is sensor 3
    # Create online data handler to listen for the data
    odh = OnlineDataHandler(sm)
    odh.visualize()