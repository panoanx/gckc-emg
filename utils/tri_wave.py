import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.signal import sawtooth


def pseudo_emg(num=5, array_length=6144, num_channels=1):
    result_array = np.zeros((num, array_length))
    emg_simulation = np.zeros((num_channels, array_length))
    triangle_wave = np.zeros((num, 11))

    Fs = 2048  # Sampling frequency
    t_tri = np.arange(0, 1 / 200, 1 / Fs)  # Time vector for one period at 200 Hz

    for i in range(num):
        result_array_test = np.zeros(array_length)
        min_interval = 20
        max_interval = 400

        # Generate random pulse positions
        current_position = 0
        while current_position < array_length:
            next_pulse_position = current_position + np.random.randint(
                min_interval, max_interval
            )
            if next_pulse_position < array_length:
                result_array_test[next_pulse_position] = 1
            current_position = next_pulse_position

        # Generate a random amplitude and phase triangle wave
        amplitude_tri = 6 * np.random.random() - 3  # Amplitude between -3 and 3
        phase_tri = 2 * np.pi * np.random.random()  # Phase between 0 and 2*pi

        # Generate triangle waveform
        triangle_wave[i, :] = amplitude_tri * (
            sawtooth(2 * np.pi * 200 * t_tri + phase_tri, width=0.5)
        )

        # Convolve pulse signal with triangle wave
        convolution_result = np.convolve(
            result_array_test, triangle_wave[i, :], mode="same"
        )

        result_array[i, :] = result_array_test
        for j in range(num_channels):
            emg_simulation[j, :] += convolution_result * (
                1 + 0.1 * np.random.randn()
            )  # Add some noise to the amplitude

    return (
        cp.asarray(result_array.T),
        cp.asarray(triangle_wave),
        cp.asarray(emg_simulation.T),
    )
