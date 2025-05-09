

from libemg.environments.controllers import Controller, RegressorController, ClassifierController
from libemg.streamers import delsys_streamer
import serial
import time
import json


# This code is part of the libemg library, this one is not used
class Prosthesis_NOTUSED:
    """
    Class for communicating with prosthetic device.
    """

    def __init__(self):
        """
        Initialize the Prosthesis object.
        """
        self.serial_port = "COM3"  # Default serial port (change as needed)
        self.baudrate = 9600
        self.timeout = 1
        self.connection = None

    def connect(self, serial_port, baudrate=9600, timeout=1):
        """
        Establish a connection to the prosthetic device.
        Parameters:
        ----------
            serial_port: The serial port to connect to (string).
            baudrate: The baud rate for the connection (int).
            timeout: Timeout for the connection (int).
        """
        try:
            self.connection = serial.Serial(
                port=serial_port,
                baudrate=baudrate,
                timeout=timeout
            )
            self.connection.flush()  # Clear input/output buffers
            print(f"Connected to prosthetic device on {self.serial_port}")
        except serial.SerialException as e:
            print(f"Error: Could not connect to prosthetic device: {e}")
            self.connection = None
    
    def disconnect(self):
        """
        Close the connection to the prosthetic device.
        """
        if self.connection and self.connection.is_open:
            self.connection.close()
            print("Disconnected from prosthetic device.")
    
    def send_command(self, command):
        """
        Send a command to the prosthetic device.
        Parameters:
        ----------
            command: The command to send (string).
        """
        if self.connection and self.connection.is_open:
            try:
                packet = bytearray(command)
                #self.connection.write(command.encode('utf-8'))
                self.connection.write(packet)
                print(f"Sent command: {command}")
            except serial.SerialException as e:
                print(f"Error: Failed to send command: {e}")
        else:
            print("Error: No active connection to the prosthetic device.")


    def is_connected(self):
        """
        Check if the device is connected.
        ----------
        Return: True if connected, False otherwise.
        """
        return self.connection is not None and self.connection.is_open
    

class MotorFunctionSelector_NOTUSED: # Made a specific controller-class for the prosthesis, try making it more modular
    """
    Class for controlling the prosthetic device using EMG signals. Call it something else, like ActuatorFunctionSelector?
    """

    def __init__(self, prosthesis = None, controller: Controller = None):
        """
        Initialize the ProsthesisController object.
        Parameters:
        ----------
            prosthesis: An instance of the Prosthesis class (optional).
        """
        # Tanken her er å gjøre det mer generelt, altså hente gains og thresholds utfra
        # for hor mange predictions en leser, men er kanskje lettere 
        self.prosthesis = prosthesis if prosthesis else Prosthesis()
        self.controller = controller if controller else RegressorController() # Default to regressor controller
        self.gain_mf1 = 1 
        self.gain_mf2 = 1
        self.thr_angle_mf1 = 0.5
        self.thr_angle_mf2 = 0.5
        self.deadband = 0.1
        self.num_motor_functions = 2  # Number of motor functions


    def get_num_motor_functions(self):
        """
        Get the number of motor functions.
        ----------
        Return: Number of motor functions (int).
        """
        if not self.controller:
            print("Controller not initialized.")
            return 0
        
        start_time = time.time()
        while time.time() - start_time < 5:  # Wait for 5 seconds to get data
            data = self.controller.get_data(['predictions'])
            if data:
                print(f"Number of motor functions: {len(data)}")
                return len(data)
            time.sleep(0.1)
        
        print("No prediction data available.")
        return 0
    
    def set_configuration(self, gain_mf1=None, gain_mf2=None, thr_angle_mf1=None, thr_angle_mf2=None, deadband=None):
        """
        Set the configuration for the prosthetic device. NOTE! Could add checks here as well to ensure the values are within a certain range, or if they are valid values.
        Parameters:
        ----------
            gains: List of gains for each motor function (list of floats).
            thr_angle_mf1: Threshold angle for motor function 1 (float).
            thr_angle_mf2: Threshold angle for motor function 2 (float).
            deadband: Deadband value (float).
        """
        if gain_mf1:
            self.gain_mf1 = gain_mf1
        if gain_mf2:
            self.gain_mf2 = gain_mf2
        if thr_angle_mf1:
            self.thr_angle_mf1 = thr_angle_mf1
        if thr_angle_mf2:
            self.thr_angle_mf2 = thr_angle_mf2
        if deadband:
            self.deadband = deadband
    
    def write_to_json(self, file_path):
        """
        Write the configuration to a JSON file.
        Parameters:
        ----------
            folder: The folder to save the JSON file (string).
            filename: The name of the JSON file (string).
        """
        config = {
            'gain_mf1': self.gain_mf1,
            'gain_mf2': self.gain_mf2,
            'thr_angle_mf1': self.thr_angle_mf1,
            'thr_angle_mf2': self.thr_angle_mf2,
            'deadband': self.deadband,
            'num_motor_functions': self.num_motor_functions
        }
        try: 
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=4) # Save the configuration to a file
                print(f"Configuration saved to {file_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
        


#if __name__ == "__main__":
    # Example usage
    # prosthesis = Prosthesis()
    # prosthesis.connect("COM3")
    
    # controller = RegressorController(ip='127.0.0.1', port=5005)
    # prost_controller = ProsthesisController(prosthesis=prosthesis, controller=controller)
    # prost_controller.get_num_motor_functions()
    # prost_controller.set_configuration(gain_mf1=1.5, gain_mf2=2.0, thr_angle_mf1=0.6, thr_angle_mf2=0.7, deadband=0.05)
    # prost_controller.write_to_json("config.json")
    
    # # Disconnect when done
    # prosthesis.disconnect()
    # Note: The above code is a simplified example. In a real-world scenario, you would need to handle exceptions and errors more robustly.