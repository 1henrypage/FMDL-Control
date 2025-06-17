import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
SAVE_PATH = './' # Ensure this matches where your .npy files are saved
DT_DATA = 1e-3   # Time step used during data generation
DURATION_DATA = 1.0 # Total duration used during data generation
N_SAMPLES_TO_SHOW = 4 # Number of samples to visualize per class

print("--- Visualizing Generated NPY Files ---")

# --- Load the Data ---
try:
    X = np.load(os.path.join(SAVE_PATH, 'controlX.npy'))
    y = np.load(os.path.join(SAVE_PATH, 'controlY.npy'))
    print(f"Loaded X with shape: {X.shape}")
    print(f"Loaded y with shape: {y.shape}")
except FileNotFoundError:
    print(f"Error: samples_X.npy or samples_y.npy not found in {SAVE_PATH}.")
    print("Please ensure the data generation script has been run and the files exist.")
    exit()

# --- Separate Samples by Class ---
class_0_indices = np.where(y == 0)[0]
class_1_indices = np.where(y == 1)[0]

# --- Select First N_SAMPLES_TO_SHOW from Each Class ---
selected_class_0_X = X[class_0_indices[:N_SAMPLES_TO_SHOW]]
selected_class_1_X = X[class_1_indices[:N_SAMPLES_TO_SHOW]]

print(f"\nDisplaying {N_SAMPLES_TO_SHOW} samples for Class 0 and {N_SAMPLES_TO_SHOW} for Class 1.")

# --- Visualization ---
fig, axes = plt.subplots(N_SAMPLES_TO_SHOW, 2, figsize=(14, N_SAMPLES_TO_SHOW * 2.5), sharex=True, sharey=True)
fig.suptitle(f'Visualizing {N_SAMPLES_TO_SHOW} Samples from Class 0 and Class 1', fontsize=16)

time_vector = np.arange(0, DURATION_DATA, DT_DATA)

# Plot Class 0 Samples
for i in range(N_SAMPLES_TO_SHOW):
    ax = axes[i, 0]
    # Reshape to 1D if necessary (X is (samples, timesteps, 1))
    spike_train = selected_class_0_X[i].flatten()
    ax.stem(time_vector, spike_train, linefmt='b-', markerfmt=' ', basefmt=" ", label='Spikes')
    ax.set_title(f'Class 0 Sample {i+1}')
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([]) # Hide y-axis ticks as it's binary
    ax.grid(True, alpha=0.3)
    if i == N_SAMPLES_TO_SHOW - 1:
        ax.set_xlabel('Time (s)')

# Plot Class 1 Samples
for i in range(N_SAMPLES_TO_SHOW):
    ax = axes[i, 1]
    spike_train = selected_class_1_X[i].flatten()
    ax.stem(time_vector, spike_train, linefmt='r-', markerfmt=' ', basefmt=" ", label='Spikes')
    ax.set_title(f'Class 1 Sample {i+1}')
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([]) # Hide y-axis ticks
    ax.grid(True, alpha=0.3)
    if i == N_SAMPLES_TO_SHOW - 1:
        ax.set_xlabel('Time (s)')

plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
plt.show()

print("\nVisualization complete. Close the plot window to exit.")
