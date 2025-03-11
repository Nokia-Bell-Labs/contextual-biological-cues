import time
import math
import random
import urandom
from array import array

# ========================================================= #
# Inference system cost for small MCU (Raspberry Pi Pico W)
# ========================================================= #
# Experiments are done by simulating the PPG signal and generating bio-cues & matching.
# Pi Pico W does not support virtual env, PyTorch or NumPy.
# Therefore, we make the following adjustments:
# 1. For statistical features, we consider: mean, std, min, max, and range.
# 2. We create custom data cleaning functions.
# 3. We implement neural network layers using basic math operations.

# === Pi Pico W Specs ===
# Form factor: 21 mm × 51 mm
# CPU: Dual-core Arm Cortex-M0+ @ 133MHz
# Memory: 264KB on-chip SRAM; 2MB on-board QSPI flash
# Connectivity: 2.4GHz IEEE 802.11b/g/n wireless LAN, on-board antenna Bluetooth 5.2
# • Support for Bluetooth LE Central and Peripheral roles • Support for Bluetooth Classic
# Input power: 1.8–5.5V DC

def simulate_ppg(duration=10, sampling_rate=100, frequency=1.2, amplitude=0.5, noise=0):
    """
    Simulate a PPG signal similar to neurokit2's signal_simulate().

    Parameters:
    - duration (float): Duration of the signal in seconds.
    - sampling_rate (int): Sampling rate in Hz (samples per second).
    - frequency (float or list): Frequency of oscillations in Hz.
    - amplitude (float or list): Amplitude of the oscillations.
    - noise (float): Amplitude of Laplace noise.

    Returns:
    - array('f') containing the simulated PPG signal.
    """
    # Ensure frequency and amplitude are lists
    if isinstance(frequency, (int, float)):
        frequency = [float(frequency)]
    if isinstance(amplitude, (int, float)):
        amplitude = [float(amplitude)]

    # Match frequency & amplitude list sizes correctly
    max_len = max(len(frequency), len(amplitude))
    if len(frequency) == 1:
        frequency = frequency * max_len  # Expand to match
    if len(amplitude) == 1:
        amplitude = amplitude * max_len  # Expand to match

    total_points = int(duration * sampling_rate)  # Total number of samples
    period = 1.0 / sampling_rate  # Time step (ensure float)
    data = array('f')  # Pre-allocate storage

    for i in range(total_points):  # Loop over each time step
        t = float(i) * period  # Compute current time (force float calculation)
        signal_value = 0.0  # Initialize signal at this time step

        for j in range(len(frequency)):  # Loop over each frequency component
            freq = float(frequency[j])
            amp = float(amplitude[j])
            nyquist = sampling_rate * 0.1  # Conservative Nyquist limit

            if freq > nyquist or (1.0 / freq) > duration:
                continue  # Skip unresolved frequencies

            # Compute sine wave correctly (avoid integer division)
            signal_value += amp * math.sin(2.0 * math.pi * freq * t)

        # Add Laplace noise if specified
        if noise > 0:
            noise_value = (urandom.getrandbits(8) / 255.0 - 0.5) * noise
            signal_value += noise_value

        data.append(signal_value)  # Store computed value

    return data


def apply_iir_filter(signal, coeffs):
    """Apply an IIR filter to the PPG signal using Direct Form II Transposed method."""
    b, a = coeffs["b"], coeffs["a"]
    filtered_signal = array('f', [0] * len(signal))  # Pre-allocate memory
    x = [0] * len(b)  # Input signal delay buffer
    y = [0] * len(a)  # Output signal delay buffer

    for i in range(len(signal)):
        x.pop()
        x.insert(0, signal[i])  # Shift buffer for input samples

        # Apply filter using Direct Form II Transposed
        y[0] = sum(b[j] * x[j] for j in range(len(b))) - sum(a[j] * y[j] for j in range(1, len(a)))
        filtered_signal[i] = y[0]  # Store the filtered sample

        # Shift buffer for output samples
        y.pop()
        y.insert(0, y[0])

    return filtered_signal


def ppg_clean_elgendi(ppg_signal, sampling_rate=50):
    """Clean PPG signal using Elgendi's method in MicroPython with updated Butterworth filter."""
    # Apply bandpass filter (0.5 - 8 Hz) using manually implemented IIR filter
    # Butterworth filter coefficients (0.5-8 Hz bandpass at 50 Hz sampling rate)
    FILTER_COEFFS = {
        "b": [0.049533, 0, -0.148599, 0, 0.148599, 0, -0.049533],  # Numerator coefficients
        "a": [1, -4.02055, 6.77970, -6.29013, 3.46489, -1.07155, 0.13776]  # Denominator coefficients
    }
    filtered = apply_iir_filter(ppg_signal, FILTER_COEFFS)
    return filtered


def compute_statistical_features(data):
    """
    Compute basic statistical features: min, max, mean, standard deviation, and range.
    Returns a list of 5 numbers.
    """
    data_list = list(data)
    n = len(data_list)
    min_val = min(data_list)
    max_val = max(data_list)
    mean_val = sum(data_list) / n
    variance = sum((x - mean_val) ** 2 for x in data_list) / n
    std_val = math.sqrt(variance)
    feature_range = max_val - min_val
    return [min_val, max_val, mean_val, std_val, feature_range]


# === Model-related code === #
# Helper function: Matrix multiplication
def matmul(A, B):
    """Performs matrix multiplication A (m x n) * B (n x p) -> C (m x p)"""
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


# Helper function: Apply ReLU activation
def relu(matrix):
    """Applies ReLU activation function element-wise."""
    return [[max(0, x) for x in row] for row in matrix]


# Optimized function to initialize layers with arrays
def init_layer(input_dim, output_dim):
    """Initializes a layer with memory-efficient arrays."""
    return {
        "weights": [array('f', [random.uniform(-0.1, 0.1) for _ in range(output_dim)]) for _ in range(input_dim)],
        "biases": array('f', [random.uniform(-0.1, 0.1) for _ in range(output_dim)])
    }


# Forward pass through a single linear layer
def linear_forward(input_matrix, layer):
    """Computes Linear transformation: y = Wx + b"""
    output = matmul(input_matrix, layer["weights"])
    return [[output[i][j] + layer["biases"][j] for j in range(len(output[0]))] for i in range(len(output))]


# Define the equivalent of the Embedding_Model
class EmbeddingModel:
    def __init__(self, input_dim, emb_dim):
        self.emb_dim = emb_dim

        # Initialize layers (matching PyTorch structure)
        self.encoder_layers = [
            init_layer(input_dim, 64),
            init_layer(64, 32),
            init_layer(32, 16)
        ]

        self.projection_layers = [
            init_layer(16, 16),
            init_layer(16, self.emb_dim)
        ]

    def forward(self, x):
        """Forward pass through the network"""
        # Ensure input is a 2D list
        if isinstance(x[0], float) or isinstance(x[0], int):
            x = [x]  # Convert single input to batch format

        # Encoder part
        for i, layer in enumerate(self.encoder_layers):
            x = linear_forward(x, layer)
            if i < len(self.encoder_layers) - 1:
                x = relu(x)  # Apply ReLU on all layers except last

        # Projection head
        for i, layer in enumerate(self.projection_layers):
            x = linear_forward(x, layer)
            if i < len(self.projection_layers) - 1:
                x = relu(x)  # Apply ReLU on all except last layer

        return x


class MatchingModel:
    def __init__(self, emb_dim):
        self.concat_embedding_dim = emb_dim * 2  # A pair of embeddings from two devices

        # Initialize layers (matching PyTorch structure)
        self.matching_layers = [
            init_layer(self.concat_embedding_dim, 64),
            init_layer(64, 32),
            init_layer(32, 16),
            init_layer(16, 1)  # Output a single value (binary classification)
        ]

    def forward(self, x):
        """Forward pass through the matching model"""
        # Ensure input is a 2D list (batch format)
        if isinstance(x[0], float) or isinstance(x[0], int):
            x = [x]  # Convert single input to batch format

        # Forward pass through the layers
        for i, layer in enumerate(self.matching_layers):
            x = linear_forward(x, layer)
            if i < len(self.matching_layers) - 1:  # Apply ReLU except for last layer
                x = relu(x)

        return x  # Single output value per pair


# Helper function: L2 normalization
def l2_normalize(matrix):
    """Normalizes each row in the matrix to have unit L2 norm."""
    normalized_matrix = []
    for row in matrix:
        norm = math.sqrt(sum(x ** 2 for x in row))  # Compute L2 norm
        if norm > 0:
            normalized_matrix.append([x / norm for x in row])  # Normalize
        else:
            normalized_matrix.append(row)  # If norm is 0, return original row
    return normalized_matrix


def sigmoid(x):
    """Computes the sigmoid function: σ(x) = 1 / (1 + exp(-x))"""
    return 1 / (1 + math.exp(-x))


def classify(output):
    """Applies sigmoid activation and returns binary classification (0 or 1)."""
    probability = sigmoid(output)
    return 1 if probability >= 0.5 else 0  # Threshold at 0.5


# === Main script === #
def main():
    print('Start of Raspberry Pi Pico W script to measure system cost.')

    # Configuration: Reduced sampling rate and window size for limited memory
    SR = 100  # Sampling rate (Hz)
    WS = 20  # Window size (seconds)
    FREQ1 = 1.0  # Frequency for device 1 signal
    FREQ2 = 2.0  # Frequency for device 2 signal

    # --- Data Simulation ---
    t0 = time.ticks_ms()
    data1 = simulate_ppg(duration=WS, sampling_rate=SR, frequency=FREQ1)
    # data2 = simulate_ppg(duration=WS, sampling_rate=SR, frequency=FREQ2) # data from other device
    t1 = time.ticks_ms()

    # --- Filtering (Butterworth filter) ---
    data1_filt = ppg_clean_elgendi(data1)
    # data2_filt = ppg_clean_elgendi(data2) # data from other device
    t2 = time.ticks_ms()

    # --- Feature Extraction ---
    # For pico W experiments, let's consider only statistical features
    features1 = compute_statistical_features(data1_filt)
    # features2 = compute_statistical_features(data2_filt)
    t3 = time.ticks_ms()

    # --- Embedding Generation Inference ---
    t_embed_start = time.ticks_ms()
    embgen_model = EmbeddingModel(input_dim=5, emb_dim=16)
    embed_1 = embgen_model.forward(features1)[0]
    # embed_2 = embgen_model.forward(features2)[0]

    # Apply L2 normalization
    embed_1 = l2_normalize([embed_1])[0]
    # embed_2 = l2_normalize([embed_2])[0]

    # print("embed_1:", embed_1)
    # print("embed_1: ", embed_2)
    t_embed_end = time.ticks_ms()

    # embed_pair = array('f', embed_1 + embed_2)  # Flattened array
    embed_pair = array('f', embed_1 + embed_1)  # Self data assuming that it received some other data

    t_match_start = time.ticks_ms()
    matching_model = MatchingModel(emb_dim=16)
    output = matching_model.forward([embed_pair])
    # print("matching output: ", output)

    # Convert to probability using sigmoid
    output_raw = output[0][0]
    probability = sigmoid(output_raw)

    # Convert to binary class
    predicted_class = classify(output_raw)
    t_match_end = time.ticks_ms()

    # print(f"Raw Matching Output: {output_raw}")
    # print(f"Probability: {probability}")
    # print(f"Predicted Class: {predicted_class}")

    # --- Print Timing and Results ---
    print("\n----Timing Information (ms)---:")
    print("Data Simulation: ", time.ticks_diff(t1, t0))
    print("[1] Filtering: ", time.ticks_diff(t2, t1))
    print("[2] Feature Extraction: ", time.ticks_diff(t3, t2))
    print("[3] Embedding Generation: ", time.ticks_diff(t_embed_end, t_embed_start))
    print("[4] Matching Inference: ", time.ticks_diff(t_match_end, t_match_start))
    print("[5] Total Time: ", time.ticks_diff(t_match_end, t0))


if __name__ == '__main__':
    main()
