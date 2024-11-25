import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2

# Load target amplitude image
target_amplitude = plt.imread("D:\\testimage\\O.png")
# Convert to grayscale if needed
if len(target_amplitude.shape) > 2:
    target_amplitude = np.mean(target_amplitude, axis=2)

# Normalize target amplitude to [0, 1] range
target_amplitude /= np.max(target_amplitude)

# Define parameters
source_distance = 0.005  # Source plane distance from target plane in meters
speed_of_sound = 1500  # Speed of sound in m/s
frequency = 2e6  # Excitation frequency in Hz
iterations = 100  # Maximum number of iterations
tolerance = 0.01  # Tolerance for amplitude reconstruction difference

# Calculate wavelength
wavelength = speed_of_sound / frequency

# Initialize source plane phase
source_phase = np.random.rand(*target_amplitude.shape) * 2 * np.pi

# Initialize difference with a large value
difference = float('inf')

# Iterative phase retrieval
for i in range(iterations):
    # Compute transfer function
    transfer_function = np.exp(1j * 2 * np.pi * source_distance / wavelength *
                               np.sqrt(1 - (np.fft.fftfreq(target_amplitude.shape[0])[:, np.newaxis]**2 +
                                            np.fft.fftfreq(target_amplitude.shape[1])**2)))

    # Compute hologram
    hologram = fft2(target_amplitude * np.exp(1j * source_phase))

    # Retrieve phase
    source_phase = np.angle(ifft2(hologram * transfer_function))

    # Check reconstruction difference every 10 iterations
    if (i + 1) % 10 == 0:
        reconstructed_amplitude = np.abs(ifft2(hologram * transfer_function))
        difference = np.mean(np.abs(reconstructed_amplitude - target_amplitude))
        print(f"Iteration {i+1}: Mean absolute difference: {difference}")

    # Check reconstruction convergence
    if difference < tolerance:
        print("Reconstruction converged after", i+1, "iterations.")
        break

# Reconstruct amplitude
reconstructed_amplitude = np.abs(ifft2(hologram * transfer_function))

# Save target amplitude image
plt.figure(figsize=(5, 5))
plt.imshow(target_amplitude, cmap='viridis')
plt.title('Target Amplitude')
plt.axis('off')
plt.savefig('target_amplitude.png', bbox_inches='tight', pad_inches=0)
plt.close()

# Save source phase image
plt.figure(figsize=(5, 5))
plt.imshow(source_phase, cmap='viridis')
plt.title('Source Phase')
plt.axis('off')
plt.savefig('source_phase.png', bbox_inches='tight', pad_inches=0)
plt.close()

# Save reconstructed amplitude image
plt.figure(figsize=(5, 5))
plt.imshow(reconstructed_amplitude, cmap='viridis')
#plt.title('Reconstructed Amplitude')
plt.axis('off')
plt.savefig('reconstructed_amplitude.png', bbox_inches='tight', pad_inches=0)
plt.close()
