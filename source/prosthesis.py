import serial
from libemg.environments.controllers import Controller
import time
import numpy as np
from pathlib import Path

class Prosthesis:
    """
    Class for communicating with prosthetic device.
    """

    def __init__(self):
        """
        Initialize the Prosthesis object.
        # """
        # self.serial_port = "COM3"  # Default serial port (change as needed)
        # self.baudrate = 9600
        # self.timeout = 1
        self.connection = None

    def connect(self, port, baudrate=9600, timeout=1):
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
                port=port,
                baudrate=baudrate,
                timeout=timeout
            )
            self.connection.flush()  # Clear input/output buffers
            print(f"Connected to prosthetic device on {port}")
        except serial.SerialException as e:
            print(f"Error: Could not connect to prosthetic device: {e}")
            self.connection = None
    
    def disconnect(self):
        """
        Close the connection to the prosthetic device.
        """
        if self.connection and self.connection.is_open:
            self.send_command([0, 0, 0, 0])  # Send stop command to the device
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