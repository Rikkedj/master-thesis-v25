# master-thesis-v25
This is the main repository for my MSc thesis at The Norwegian University of Science and Technology (NTNU) in spring 2025. This repository relies on a forked version of the open-source LibEMG library: https://github.com/Rikkedj/master25-libemg.git. 

## Setup and Usage:
1. Clone this repository at your local machine.
2. Make a virtual environment with Python version below 3.11. This is due to the current (June 2025) LibEMG version depends on some libraries only supported by those versions.
3. Clone the forked LibEMG repository (https://github.com/Rikkedj/master25-libemg.git) into the same root folder as this project.
4. From the root folder with both cloned repos, navigate to **this** project directory.
5. Activate your virtual environment and install the required packages:
   ```python
   pip install -r reguirements.txt
7. Run the control system interface with the desired sensors by executing:
   ```python
   python .\source\main.py --sensors 1 2 3

Replace 1 2 3 with the sensor IDs you intend to use.

This will launch the interface where you can collect data, tune parameters, and control the prosthesis, as illustrated below.
![main-menu](https://github.com/user-attachments/assets/218c83bb-4887-4802-8585-ec8c1f9bb76c)
