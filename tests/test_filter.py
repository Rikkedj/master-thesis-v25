import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time

from libemg.environments.controllers import Controller, RegressorController, ClassifierController

from source import filters

if __name__ == "__main__":
    UDP_IP = "127.0.0.1"  # Localhost (change if sending from another machine)
    UDP_PORT = 5005

    filter = filters.FlutterRejectionFilter()

    controller = RegressorController(ip=UDP_IP, port=UDP_PORT)
    while True:
        pred = controller.get_data(['predictions'])
        if pred:
            print("Prediction: ", pred)
            # Apply the filter to the prediction data
            filtered_pred = filter.apply(pred)
            print("Filtered Prediction: ", filtered_pred)
        else:
            print("No prediction data available.")
        time.sleep(0.1)
