import cupy as np


def pseudo_emg(amp: float) -> np.ndarray:
    """
    Generate a pseudo EMG signal with random spikes.

    Args:
    amp: Amplitude of the spikes

    Returns:
    smoothed_spike_train: Smoothed pseudo EMG signal
    """
    duration = 1
    fs = 2048
    prob_spike = 0.01

    t = np.arange(0, duration, 1 / fs)
    spike_train = (np.random.rand(len(t)) < prob_spike) * amp

    width_ms = 10
    width_samples = int(width_ms * fs / 1000)
    triangle = np.concatenate(
        (np.linspace(0, 1, width_samples // 2), np.linspace(1, 0, width_samples // 2))
    )

    smoothed_spike_train = np.convolve(spike_train, triangle, mode="same")

    return smoothed_spike_train


def generate_test_data(
    time_length=10000,
    num_sources=5,
    num_observations=5,
    kernel_length=10,
    num_realizations=2,
    snr_range=(10, 0),
):
    # Generate random input pulse trains
    T = np.random.randint(-10, 11, size=(num_sources, 200))
    t = np.zeros((time_length, num_sources))
    for j in range(num_sources):
        for k in range(200):
            idx = k * 100 + T[j, k]
            if 0 <= idx < time_length:
                t[idx, j] = 1

    # Generate random mixing matrix H
    H = np.random.randn(num_observations, num_sources, kernel_length)

    # Convolve input pulse trains with mixing matrix H
    x = np.zeros((time_length, num_observations))
    for m in range(num_observations):
        for n in range(num_sources):
            x[:, m] += np.convolve(t[:, n], H[m, n], mode="same")

    # Add noise with varying SNR
    signal_power = np.mean(x**2)
    sig_mat = np.zeros((time_length, num_observations * num_realizations))
    for realization in range(num_realizations):
        snr = np.random.uniform(snr_range[0], snr_range[1])
        noise_power = signal_power / (10 ** (snr / 10))
        noise = np.sqrt(noise_power) * np.random.randn(time_length, num_observations)
        sig_mat[
            :, realization * num_observations : (realization + 1) * num_observations
        ] = (x + noise)

    return sig_mat, t
