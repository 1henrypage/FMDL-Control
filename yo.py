import numpy as np
import matplotlib.pyplot as plt
import os


class VanillaRFNeuron:
    """A standard Izhikevich Resonate-and-Fire neuron."""
    def __init__(self, omega, b, delta=1e-3, threshold=1.0):
        self.omega = float(omega)
        self.b = float(b)
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.u = 0.0 + 0.0j
        self.spikes = []
        self.membrane_potential_log = []

    def step(self, I):
        udot = (self.b + 1j * self.omega) * self.u + I
        self.u += self.delta * udot
        self.membrane_potential_log.append(self.u)

        spike = 1.0 if self.u.real > self.threshold else 0.0
        self.spikes.append(spike)

        if spike:
            self.u = 0.0 + 0.0j
        return spike

    def reset_state(self):
        self.u = 0.0 + 0.0j
        self.spikes = []
        self.membrane_potential_log = []

class BRFNeuron:
    """A Balanced Resonate-and-Fire neuron."""
    def __init__(self, omega, b_prime, delta=1e-3, threshold_c=1.0, gamma=0.9):
        self.omega = float(omega)
        self.b_prime = float(b_prime)
        self.delta = float(delta)
        self.threshold_c = float(threshold_c)
        self.gamma = float(gamma)

        # State variables
        self.u = 0.0 + 0.0j
        self.q = 0.0  
        self.spikes = []
        self.membrane_potential_log = []

    def p(self, omega):
        radicand = 1 - (self.delta * omega)**2
        if radicand < 0: return -1/self.delta 
        return (-1 + np.sqrt(radicand)) / self.delta

    def step(self, I):
        b_t = self.p(self.omega) - self.b_prime - self.q

        udot = (b_t + 1j * self.omega) * self.u + I
        self.u += self.delta * udot
        self.membrane_potential_log.append(self.u)

        threshold_t = self.threshold_c + self.q

        spike = 1.0 if self.u.real > threshold_t else 0.0
        self.spikes.append(spike)

        self.q = self.gamma * self.q + spike
        return spike

    def reset_state(self):
        self.u = 0.0 + 0.0j
        self.q = 0.0
        self.spikes = []
        self.membrane_potential_log = []



def generate_periodic_train(freq, duration, dt): # binary spike train for freq
    times = np.arange(0, duration, 1.0 / freq)
    binary = np.zeros(int(duration / dt))
    indices = (times / dt).astype(int)
    binary[indices] = 1.0
    return binary

def generate_burst_train(start_time, burst_duration, freq, total_duration, dt): # burst 
    binary = np.zeros(int(total_duration / dt))
    times = np.arange(start_time, start_time + burst_duration, 1.0 / freq)
    indices = (times / dt).astype(int)
    binary[indices] = 1.0
    return binary

def plot_simulation(ax, time_vec, input_train, neuron_u, neuron_spikes, title):
    ax.plot(time_vec, np.real(neuron_u), label='Re(u)', color='C0')
    ax.set_ylabel('Membrane Potential', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax.set_title(title)
    ax.axhline(1.0, ls='--', color='gray', label='Threshold')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.stem(time_vec, input_train, linefmt='k-', markerfmt=' ', basefmt=" ", label='Input Spikes')
    output_times = time_vec[np.array(neuron_spikes) > 0]
    ax2.eventplot(output_times, color='C3', linelengths=0.5, label='Output Spikes', lineoffsets=1.1)
    ax2.set_yticks([])
    ax2.set_ylim([0, 2])
    ax.set_xlabel('Time (s)')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

print("--- Generating Visualizations ---")

# divergence viosualisation
print("Displaying Visualization 1: The Divergence Problem...")
DT_VIZ = 1e-3
DURATION_VIZ = 0.5
time_vector_viz = np.arange(0, DURATION_VIZ, DT_VIZ)
resonant_freq_viz = 20.0
omega_val_viz = 2 * np.pi * resonant_freq_viz
b_divergent = -1.0
b_prime_stable = 0.1
input_signal_div = generate_periodic_train(resonant_freq_viz, DURATION_VIZ, DT_VIZ)

vanilla_rf_div = VanillaRFNeuron(omega=omega_val_viz, b=b_divergent, delta=DT_VIZ)
brf_div = BRFNeuron(omega=omega_val_viz, b_prime=b_prime_stable, delta=DT_VIZ)

for i in range(len(time_vector_viz)):
    vanilla_rf_div.step(input_signal_div[i] * 5.0)
    brf_div.step(input_signal_div[i] * 5.0)

fig1, axs1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig1.suptitle("Visualization 1: The Divergence Problem", fontsize=16)
plot_simulation(axs1[0], time_vector_viz, input_signal_div,
                vanilla_rf_div.membrane_potential_log, vanilla_rf_div.spikes,
                'Problem: Vanilla RF Neuron Diverges')
plot_simulation(axs1[1], time_vector_viz, input_signal_div,
                brf_div.membrane_potential_log, brf_div.spikes,
                'Solution: BRF Neuron Remains Stable')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# spiking visualsation
print("Displaying Visualization 2: The Excessive Spiking Problem...")
b_stable_viz = -1.0
input_signal_burst = generate_burst_train(start_time=0.1, burst_duration=0.05,
                                          freq=100.0, total_duration=DURATION_VIZ, dt=DT_VIZ)

vanilla_rf_burst = VanillaRFNeuron(omega=omega_val_viz, b=b_stable_viz, delta=DT_VIZ)
brf_burst = BRFNeuron(omega=omega_val_viz, b_prime=b_prime_stable, delta=DT_VIZ)

for i in range(len(time_vector_viz)):
    vanilla_rf_burst.step(input_signal_burst[i] * 20.0)
    brf_burst.step(input_signal_burst[i] * 20.0)

fig2, axs2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig2.suptitle("Visualization 2: The Excessive Spiking Problem", fontsize=16)
plot_simulation(axs2[0], time_vector_viz, input_signal_burst,
                vanilla_rf_burst.membrane_potential_log, vanilla_rf_burst.spikes,
                'Problem: Vanilla RF Spikes Excessively to Noise')
plot_simulation(axs2[1], time_vector_viz, input_signal_burst,
                brf_burst.membrane_potential_log, brf_burst.spikes,
                'Solution: BRF Neuron Suppresses Noise')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# DATTASET GEN

SAVE_PATH = './'
os.makedirs(SAVE_PATH, exist_ok=True)

DT_DATA = 1e-3
DURATION_DATA = 1.0
N_STEPS_DATA = int(DURATION_DATA / DT_DATA)
TARGET_FREQS_DATA = [10, 20, 40]
N_SAMPLES_PER_CLASS = 200
np.random.seed(42)


def to_binary(spike_times, duration, dt):
    n_steps = int(duration / dt)
    binary = np.zeros(n_steps)
    indices = np.clip((spike_times / dt).astype(int), 0, n_steps - 1)
    binary[indices] = 1
    return binary

def make_periodic_train(freq, duration, phase=0.0):
    if freq <= 0: return np.array([])
    T = 1.0 / freq
    return np.arange(phase, duration, T)

def make_burst_train(start, burst_dur, freq, total_dur):
    phase = np.random.uniform(0, 1.0 / freq)
    times = np.arange(phase, burst_dur, 1.0/freq)
    return times + start

X = []
y = []

# generaTion of positive samples
for f_target in TARGET_FREQS_DATA:
    for _ in range(N_SAMPLES_PER_CLASS):
        phase = np.random.uniform(0, 1.0 / f_target)
        spike_times = make_periodic_train(f_target, DURATION_DATA, phase)
        X.append(to_binary(spike_times, DURATION_DATA, DT_DATA))
        y.append(1)

# negative
n_negative_per_type = (len(TARGET_FREQS_DATA) * N_SAMPLES_PER_CLASS) // 2

# off
for _ in range(n_negative_per_type):
    f_off = np.random.uniform(5, 50)
    while any(abs(f_off - f_target) < 2.0 for f_target in TARGET_FREQS_DATA):
         f_off = np.random.uniform(5, 50)
    phase = np.random.uniform(0, 1.0 / f_off)
    spike_times = make_periodic_train(f_off, DURATION_DATA, phase)
    X.append(to_binary(spike_times, DURATION_DATA, DT_DATA))
    y.append(0)

# burst
for _ in range(n_negative_per_type):
    start_time = np.random.uniform(0.1, DURATION_DATA - 0.2)
    burst_dur = np.random.uniform(0.05, 0.1)
    burst_freq = np.random.uniform(80, 150)
    spike_times = make_burst_train(start_time, burst_dur, burst_freq, DURATION_DATA)
    X.append(to_binary(spike_times, DURATION_DATA, DT_DATA))
    y.append(0)

# Shuffle and save
indices = np.arange(len(X))
np.random.shuffle(indices)

X = np.array(X)[indices].reshape(-1, N_STEPS_DATA, 1)
y = np.array(y)[indices]

print(f"Generated {len(X)} samples.")
print(f"Dataset shape X: {X.shape}")
print(f"Dataset shape y: {y.shape}")
print(f"Class balance (1s vs 0s): {np.mean(y):.2f}")

np.save(os.path.join(SAVE_PATH, 'controlX.npy'), X)
np.save(os.path.join(SAVE_PATH, 'controlY.npy'), y)

print(f"\nDataset saved to {os.path.join(SAVE_PATH, 'controlX.npy')} and {os.path.join(SAVE_PATH, 'controlY.npy')}")
