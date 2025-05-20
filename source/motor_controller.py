import numpy as np
def sigmoid(x):
    """
    Sigmoid function.
    Parameters:
    ----------
    x: float
        Input value (float).
    Returns:
    ----------
    float: 
        Sigmoid of the input value (float).
    """
    return 1 / (1 + np.exp(-x))

def hybrid_activation_profile(pred, k1=1, k2=1, max_output=255):
    """
    Applies stacked sigmoid activation profile for each prediction.
    Parameters:
    ----------
    pred: float
        Prediction for a single motor function (e.g., mf1).
    k1: float
        Steepness of the first sigmoid function (float).
    k2: float
        Steepness of the second sigmoid function (float).
    max_output: float
        Maximum output value (float).
    Returns:
    ----------
    float: 
        Activation value for the motor function (e.g., mf1_activation).
    """

    return max_output*(sigmoid(k1 * pred) + sigmoid(k2 * pred) - 1) # -1 to stay within the range of 0 to 1 

def get_motor_setpoints(pred, thr_angle_mf1=45, thr_angle_mf2=45, max_value=1):
    """
    Get the motor setpoints for the actuator function.
    Parameters:
    ----------
    pred: list
        List of predictions for each motor function (e.g., [mf1, mf2]).
    thr_angle_mf1: float
        Threshold angle for motor function 1 (float).
    thr_angle_mf2: float
        Threshold angle for motor function 2 (float).
    max_value: float
        Maximum value for the motor function (float).
    Returns:   
    ----------
    list: 
        List of motor setpoints for each motor function (e.g., [mf1_setpoint, mf2_setpoint]).
    """
    motor_setpoint = [0, 0, 0, 0] # rest position

    theta = np.arctan2(pred[1], pred[0])
    theta_deg = np.rad2deg(theta)
    mf1, mf2 = pred[0], pred[1]

    if thr_angle_mf1 < abs(theta_deg) < (180 - thr_angle_mf1): # mf2 active
        activation = hybrid_activation_profile(mf2, k1=1, k2=1, max_output=255)
        motor_setpoint[2 if mf2 > 0 else 3] = int(min(abs(activation), 255))
        #int(min(abs(mf2 / max_value * 255), 255))
    if abs(theta_deg) < (90 - thr_angle_mf2) or abs(theta_deg) > (90 + thr_angle_mf2): # mf 1 active
        activation = hybrid_activation_profile(mf1, k1=1, k2=1, max_output=255)
        motor_setpoint[0 if mf1 > 0 else 1] = int(min(abs(activation), 255))
        #int(min(abs(mf1 / max_value * 255), 255))
   
    return motor_setpoint

