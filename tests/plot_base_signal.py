import numpy as np
import matplotlib.pyplot as plt

def generate_and_plot_signal(transition_duration, hold_duration, rest_duration, sampling_rate):
    # Generate the signal
    num_transition = max(1, int(transition_duration * sampling_rate))
    num_hold = max(1, int(hold_duration * sampling_rate))

    ramp_up = np.linspace(0, 1, num_transition, endpoint=False)
    hold_high = np.ones(num_hold)
    ramp_down = np.linspace(1, -1, 2 * num_transition, endpoint=False)
    hold_low = -np.ones(num_hold)
    ramp_back = np.linspace(-1, 0, num_transition)
    rest = np.zeros(int(rest_duration * sampling_rate))

    signal = np.concatenate((ramp_up, hold_high, ramp_down, hold_low, ramp_back, rest))
    time = np.arange(len(signal)) / sampling_rate

    # Key time intervals
    t_hold_high_start = len(ramp_up) / sampling_rate
    t_hold_high_end = (len(ramp_up) + len(hold_high)) / sampling_rate
    t_hold_low_start = (len(ramp_up) + len(hold_high) + len(ramp_down)) / sampling_rate
    t_hold_low_end = (len(ramp_up) + len(hold_high) + len(ramp_down) + len(hold_low)) / sampling_rate
    t_rest_start = (len(signal) - len(rest)) / sampling_rate
    t_rest_end = len(signal) / sampling_rate

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, signal, label='Generated Signal')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('s(t)')
    #ax.axvline(x=t_hold_high_start, color='green', linestyle='--', label='Hold High Start')
    ax.set_title('Base signal for system training')
    ax.grid(True)

    # Vertical positions for text just above signal levels
    y_offset = 0.05
    ax.text((t_hold_high_start + t_hold_high_end)/2, 1 + y_offset, 'Steady-state',
            ha='center', va='bottom', fontsize=9, color='green')
    ax.text((t_hold_low_start + t_hold_low_end)/2, -1 + y_offset, 'Steady-state',
            ha='center', va='bottom', fontsize=9, color='green')
    ax.text((t_rest_start + t_rest_end)/2, 0 + y_offset, 'Rest',
            ha='center', va='bottom', fontsize=9, color='green')

    ax.set_ylim(-1.2, 1.2)
    plt.tight_layout()
    plt.show()

    return signal

def plot_activation_profile(k1=10, k2=10, a=0.3, b=0.7):
    """
    Plots the double-sigmoid activation profile for input values between 0 and 1.
    
    Parameters:
    - k1, k2: Steepness of the sigmoid functions
    - a, b: Centers of the two sigmoid transitions (must be in [0,1])
    """
    x = np.linspace(0, 1, 500)
    sigmoid1 = 1 / (1 + np.exp(-k1 * (x - a)))
    sigmoid2 = 1 / (1 + np.exp(-k2 * (x - b)))
    activation = sigmoid1 + sigmoid2 -1

    plt.figure(figsize=(8, 4))
    plt.plot(x, activation, label='Activation Profile', color='blue')
    #plt.plot(x, sigmoid1, '--', label='Sigmoid 1 (low activation)', color='green')
    #plt.plot(x, sigmoid2, '--', label='Sigmoid 2 (high activation)', color='red')
    plt.xlabel("Input Prediction")
    plt.ylabel("Activation Level")
    plt.title("Double-Sigmoid Activation Profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example parameters
    transition_duration = 1  # seconds
    hold_duration = 2        # seconds
    rest_duration = 3        # seconds
    sampling_rate = 24      # Hz

    #signal = generate_and_plot_signal(transition_duration, hold_duration, rest_duration, sampling_rate)
    plot_activation_profile(k1=20, k2=20, a=0.0, b=0.0)