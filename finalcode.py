import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries


# ─────────────────────────────────────────────────────────────────────────────
# PLOT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'axes.axisbelow':       True,
    'axes.labelsize':       20,
    'axes.titlesize':       20,
    'axes.titleweight':     'bold',
    'axes.titlepad':        10,
    'axes.grid':            True,
    'grid.linewidth':       0.8,
    'xtick.labelsize':      15,
    'ytick.labelsize':      15,
    'legend.fontsize':      15,
    'legend.framealpha':    0.9,
    'figure.dpi':           120,
    'lines.linewidth':      0.9,
    'font.family':          'sans-serif',
    'figure.figsize':        (12, 6),
})

# Colour palette
C_SIGNAL   = '#1F77B4'   # muted blue  — real data / chirp signal
C_TEMPLATE = '#D62728'   # muted red   — model template
C_CORR     = '#7B2D8B'   # muted purple — correlation / FFT
C_ANNOT    = '#FF7F0E'   # muted orange — annotation lines


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

# Calculates the chirp mass from two component masses (in solar masses)
def Chirp_Mass_calc(m1, m2):
    M0 = 1.989e30  # Solar mass in kg
    chirp_mass = ((m1 * m2 * M0**2)**(3/5)) / (((m1 + m2) * M0)**(1/5))
    return chirp_mass

# GW frequency as a function of chirp mass M and time-to-coalescence tau
def Frequency_calc(M, tau):
    c = 299792458.0
    G = 6.67430e-11
    return 1/np.pi * ((5 * c**5) / (256 * G**(5/3) * M**(5/3) * tau))**(3/8)

# Accumulated GW phase as a function of chirp mass M and time-to-coalescence tau
def Phase_calc(M, tau):
    c = 299792458.0
    G = 6.67430e-11
    return -2 * ((c**3) / (5 * G * M))**(5/8) * tau**(5/8)

# GW strain amplitude as a function of frequency and phase
def Strain_calc(A, frequency, phi):
    return A * frequency**(2/3) * np.cos(phi)

# Schwarzschild radius for a given mass in kg
def Schwarzschild_radius(mass):
    G = 6.67430e-11
    c = 299792458.0
    return (2 * G * mass) / c**2

# Orbital separation from GW frequency using Kepler's Third Law
def Keplers_third_law_separation(m1, m2, frequency):
    G = 6.67430e-11
    M0 = 1.989e30
    # GW frequency is twice the orbital frequency, so period = 2 / f_gw
    period = 2 / frequency
    return (G * (m1 + m2) * M0 * period**2 / (4 * np.pi**2))**(1/3)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: CHIRP SIGNAL GENERATION (Newtonian Toy Model)
# Simulate the gravitational wave strain of a 29-36 solar mass binary inspiral.  This is because the actual GW150914 event had black hole masses of approximately 29 and 36.
# The signal is terminated at when the separation is the sum of Schwarzschild radii
# ─────────────────────────────────────────────────────────────────────────────

m1_p1 = 29  # Component masses in solar masses
m2_p1 = 36
M0 = 1.989e30
mass1_p1 = m1_p1 * M0
mass2_p1 = m2_p1 * M0

M_p1 = Chirp_Mass_calc(m1_p1, m2_p1)
r1_p1 = Schwarzschild_radius(mass1_p1)
r2_p1 = Schwarzschild_radius(mass2_p1)

sample_rate = 4096  # Hz
duration = 2        # seconds
A = 1e-21           # Strain amplitude

t_p1 = np.linspace(0, duration, int(sample_rate * duration))
tau_p1 = np.maximum(duration - t_p1, 1e-4)

f_p1 = Frequency_calc(M_p1, tau_p1)
phi_p1 = Phase_calc(M_p1, tau_p1)
strain_p1 = Strain_calc(A, f_p1, phi_p1)

# Trim at the crash point
sep_p1 = Keplers_third_law_separation(m1_p1, m2_p1, f_p1)
crash_p1 = np.where(sep_p1 <= r1_p1 + r2_p1)[0]
if len(crash_p1) > 0:
    t_clean_p1 = t_p1[:crash_p1[0]]
    strain_clean_p1 = strain_p1[:crash_p1[0]]
else:
    t_clean_p1 = t_p1
    strain_clean_p1 = strain_p1

# Plot clean chirp signal
# plt.plot(t_clean_p1, strain_clean_p1, color=C_SIGNAL)
# plt.title('Phase 1: Clean Newtonian Chirp Signal (29-36 M0, 4096 Hz)')
# plt.xlabel('Time [s]')
# plt.ylabel('Strain')
# plt.grid()
# plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: SYNTHETIC SIGNAL PROCESSING PIPELINE
# Bury the chirp in white noise, apply bandpass filter, run matched filtering.
# Test the pipeline across multiple sample rates to demonstrate the Nyquist limit.
# ─────────────────────────────────────────────────────────────────────────────

sample_rates_to_test = [4096, 2048, 1024, 512, 256, 128, 64]

for rate in sample_rates_to_test:
    t = np.linspace(0, duration, int(rate * duration))
    tau = np.maximum(duration - t, 1e-4)

    f = Frequency_calc(M_p1, tau)
    phi = Phase_calc(M_p1, tau)
    strain = Strain_calc(A, f, phi)

    # Find the crash index
    sep = Keplers_third_law_separation(m1_p1, m2_p1, f)
    crash_idx = np.where(sep <= r1_p1 + r2_p1)[0]
    if len(crash_idx) > 0:
        t_clean = t[:crash_idx[0]]
        strain_clean = strain[:crash_idx[0]]
    else:
        print(f"Sample Rate: {rate} Hz | No crash within time frame.")
        t_clean = t
        strain_clean = strain

    # Add white noise
    static = np.random.normal(0, 1e-21, len(strain_clean))
    strain_noisy = strain_clean + static

    # Nyquist safety check
    nyquist = 0.5 * rate
    if nyquist <= 300.0:
        print(f"Sample Rate: {rate} Hz FAILED: Framerate too low for a 300Hz filter.")
        continue

    # Bandpass filter (20–300 Hz)
    b, a = scipy.signal.butter(4, [20.0 / nyquist, 300.0 / nyquist], btype='band')
    filtered_strain = scipy.signal.filtfilt(b, a, strain_noisy)

    # Matched filter
    correlation_scores = scipy.signal.correlate(filtered_strain, strain_clean, mode='same')
    max_correlation = np.max(correlation_scores)
    print(f"Sample Rate: {rate} Hz | Peak Correlation Score: {max_correlation:.2e}")

# Baseline plots at 4096 Hz
t = np.linspace(0, duration, int(4096 * duration))
tau = np.maximum(duration - t, 1e-4)
f = Frequency_calc(M_p1, tau)
phi = Phase_calc(M_p1, tau)
strain = Strain_calc(A, f, phi)

sep = Keplers_third_law_separation(m1_p1, m2_p1, f)
crash_idx = np.where(sep <= r1_p1 + r2_p1)[0]
if len(crash_idx) > 0:
    t_clean = t[:crash_idx[0]]
    strain_clean = strain[:crash_idx[0]]

static = np.random.normal(0, 1e-21, len(strain_clean))
strain_noisy = strain_clean + static

# FFT plot
fft_strain = scipy.fft.fft(strain_noisy)
dt = 1 / 4096
freqs = scipy.fft.fftfreq(len(strain_clean), dt)
pos_freqs = freqs[freqs >= 0]
pos_fft = fft_strain[freqs >= 0]

plt.figure(figsize=(12, 6))
plt.loglog(pos_freqs, np.abs(pos_fft), color=C_CORR)
plt.title('Phase 2: FFT of Chirp Signal with Noise (4096 Hz)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude of FFT')
plt.xlim(1, 1000)
plt.grid()
plt.show()

# Time domain plot
plt.figure(figsize=(12, 6))
plt.plot(t_clean, strain_noisy, color=C_SIGNAL)
plt.title('Phase 2: Chirp Signal Buried in White Noise (4096 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Strain')
plt.grid()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: REAL OBSERVATORY DATA — GW150914
# Fetch 16s of H1 strain data, whiten to flatten coloured noise,
# apply bandpass filter, run matched filtering with a 29-36 M0 template,
# calculate SNR, and overlay the template on the real signal.
# ─────────────────────────────────────────────────────────────────────────────

# --- Step 1: Fetch real H1 strain data ---
# GW150914 merger GPS time: 1126259462. Fetch 8s either side.
print("Fetching GW150914 H1 strain data from GWOSC...")
real_data = TimeSeries.fetch_open_data(
    'H1', start=1126259462 - 8, end=1126259462 + 8, sample_rate=4096, cache=True
    )
t_plot = real_data.times.value - real_data.times.value[0]  # Time axis: 0 to 16s

# --- Step 2: Whiten ---
# Flattens the 1/f coloured noise floor so the 150 Hz signal isn't swamped by low-frequency seismic noise. Uses 4s FFT windows with 2s overlap.
whitened_real_data = real_data.whiten(fftlength=4, overlap=2)

# --- Step 3: Bandpass filter (20–300 Hz) ---
nyquist = 0.5 * 4096
b, a = scipy.signal.butter(4, [20.0 / nyquist, 300.0 / nyquist], btype='band')
filtered_real_data = scipy.signal.filtfilt(b, a, whitened_real_data.value)

# Plot filtered real data (zoomed in to the merger)
# plt.figure(figsize=(12, 6))
# plt.plot(t_plot, filtered_real_data, label='Filtered Strain Data', color=C_SIGNAL)
# plt.title('Phase 3: GW150914 H1 Strain — Whitened & Bandpassed (20–300 Hz)')
# plt.xlabel('Time [s]')
# plt.ylabel('Strain (whitened)')
# plt.xlim(8, 8.5)
# plt.legend()
# plt.grid()
# plt.show()

# --- Step 4: Build synthetic template (29-36 M0 — actual GW150914 masses) ---
m1_p3, m2_p3 = 29, 36
mass1_p3 = m1_p3 * M0
mass2_p3 = m2_p3 * M0
M_p3 = Chirp_Mass_calc(m1_p3, m2_p3)
r1_p3 = Schwarzschild_radius(mass1_p3)
r2_p3 = Schwarzschild_radius(mass2_p3)

t_tmpl = np.linspace(0, duration, int(4096 * duration))
tau_tmpl = np.maximum(duration - t_tmpl, 1e-4)
f_tmpl = Frequency_calc(M_p3, tau_tmpl)
phi_tmpl = Phase_calc(M_p3, tau_tmpl)
strain_tmpl = Strain_calc(1e-21, f_tmpl, phi_tmpl)

sep_tmpl = Keplers_third_law_separation(m1_p3, m2_p3, f_tmpl)
crash_tmpl = np.where(sep_tmpl <= r1_p3 + r2_p3)[0]
strain_clean_p3 = strain_tmpl[:crash_tmpl[0]]

# --- Step 5: Matched filter (clean template) ---
correlation = scipy.signal.correlate(filtered_real_data, strain_clean_p3, mode='same')

peak = np.max(np.abs(correlation))
snr = peak / np.std(np.abs(correlation))
print(f"SNR (clean template vs real data): {snr:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(t_plot, np.abs(correlation), color=C_CORR)
plt.title(f'Phase 3: Matched Filter Output — Clean Template vs Real Data | SNR: {snr:.2f}')
plt.xlabel('Time [s]')
plt.ylabel('Correlation')
plt.grid()
plt.show()

# --- Step 6: Overlay template on real signal ---
peak_index = int(8.427 * 4096)
time_peak = t_plot[peak_index]
t_tmpl_aligned = np.linspace(time_peak - len(strain_clean_p3)/4096, time_peak, len(strain_clean_p3))
strain_clean_normalised = strain_clean_p3 / np.max(np.abs(strain_clean_p3)) * np.max(np.abs(filtered_real_data))

# plt.figure(figsize=(12, 6))
# plt.plot(t_plot, filtered_real_data, label='Filtered Real Data (H1)', color=C_SIGNAL)
# plt.plot(t_tmpl_aligned, strain_clean_normalised, label='Newtonian Template (29-36 M0)', alpha=0.7, color=C_TEMPLATE)
# plt.title('Phase 3: Newtonian Template Overlaid on GW150914')
# plt.xlabel('Time [s]')
# plt.ylabel('Strain (whitened)')
# plt.xlim(8, 8.5)
# plt.legend()
# plt.grid()
# plt.show()

# --- Step 7: Matched filter (noisy template) for comparison + overlay ---
static_p3 = np.random.normal(0, 1e-21, len(strain_clean_p3))
strain_noisy_p3 = strain_clean_p3 + static_p3
b, a = scipy.signal.butter(4, [20.0 / nyquist, 300.0 / nyquist], btype='band')
filtered_noisy_tmpl = scipy.signal.filtfilt(b, a, strain_noisy_p3)

correlation_noisy = scipy.signal.correlate(filtered_real_data, filtered_noisy_tmpl, mode='same')
peak_noisy = np.max(np.abs(correlation_noisy))
snr_noisy = peak_noisy / np.std(np.abs(correlation_noisy))

# plt.figure(figsize=(12, 6))
# plt.plot(t_plot, np.abs(correlation_noisy), color=C_CORR)
# plt.title(f'Phase 3: Matched Filter Output — Noisy Template vs Real Data | SNR: {snr_noisy:.2f}')
# plt.xlabel('Time [s]')
# plt.ylabel('Correlation')
# plt.grid()
# plt.show()

# Overlay noisy template on real data
strain_noisy_normalised = filtered_noisy_tmpl / np.max(np.abs(filtered_noisy_tmpl)) * np.max(np.abs(filtered_real_data))

# plt.figure(figsize=(12, 6))
# plt.plot(t_plot, filtered_real_data, label='Filtered Real Data (H1)', color=C_SIGNAL)
# plt.plot(t_tmpl_aligned, strain_noisy_normalised, label='Noisy Template (29-36 M0)', alpha=0.7, color=C_TEMPLATE)
# plt.title('Phase 3: Filtered Noisy Template Overlaid on GW150914')
# plt.xlabel('Time [s]')
# plt.ylabel('Strain (whitened)')
# plt.xlim(8, 8.5)
# plt.legend()
# plt.grid()
# plt.show()


print(f"SNR (noisy template vs real data): {snr_noisy:.2f}")
print(f"\nSummary:")
print(f"  Clean template SNR : {snr:.2f}  (stable, reproducible)")
print(f"  Noisy template SNR : {snr_noisy:.2f} (varies each run due to random noise)")
print(f"  LIGO detection threshold: 8.0")

# ─────────────────────────────────────────────────────────────────────────────
# Spectogram
# ─────────────────────────────────────────────────────────────────────────────

# Spectrogram of the real data (whitened & bandpassed)
# plt.figure(figsize=(12, 6))
# plt.specgram(filtered_real_data, Fs=4096, NFFT=256, noverlap=255, pad_to=1024, cmap='viridis', vmin=-45, vmax=-20)
# plt.title('Phase 3: Spectrogram of GW150914 H1 Strain (Whitened & Bandpassed)')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.colorbar(label='Signal Power [dB]')
# plt.xlim(8, 8.5)
# plt.ylim(0, 500)
# plt.grid()
# plt.show()


# Spectrogram of the Filtered Noisy Newtonian template (29-36 M0)
crash_time = len(strain_noisy_normalised) / 4096

# plt.figure(figsize=(12, 6))
# plt.specgram(strain_noisy_normalised, Fs=4096, NFFT=128, noverlap=127, pad_to=1024, cmap='viridis', vmin=-65, vmax=-10)
# plt.title('Phase 3: Spectrogram of the Noisy Newtonian Template (29-36 M0)')
# plt.xlabel('Time [s] (Template model time axis)')
# plt.ylabel('Frequency [Hz]')
# plt.colorbar(label='Signal Power [dB]')
# plt.xlim(crash_time - 0.5, crash_time)
# plt.ylim(0, 500)
# plt.grid()
# plt.show()



# Testing Different masses to show the change in SNR

### THis model didnt work, we get "Best masses: m1 = 64 M0,  m2 = 29 M0, SNR = 19.83".  
# snr_results = []
# for m1_test in range(10, 80, 1):
#     for m2_test in range(10, 80, 1):
#         if m2_test > m1_test: # Avoid duplicates
#             continue
        
#         mass1_p3 = m1_test * M0
#         mass2_p3 = m2_test * M0
#         M_p3 = Chirp_Mass_calc(m1_test, m2_test)
#         r1_p3 = Schwarzschild_radius(mass1_p3)
#         r2_p3 = Schwarzschild_radius(mass2_p3)

#         t_tmpl = np.linspace(0, duration, int(4096 * duration))
#         tau_tmpl = np.maximum(duration - t_tmpl, 1e-4)
#         f_tmpl = Frequency_calc(M_p3, tau_tmpl)
#         phi_tmpl = Phase_calc(M_p3, tau_tmpl)
#         strain_tmpl = Strain_calc(1e-21, f_tmpl, phi_tmpl)

#         sep_tmpl = Keplers_third_law_separation(m1_test, m2_test, f_tmpl)
#         crash_tmpl = np.where(sep_tmpl <= r1_p3 + r2_p3)[0]
#         strain_clean_p3 = strain_tmpl[:crash_tmpl[0]]

#         strain_clean_norm = strain_clean_p3 / np.sqrt(np.sum(strain_clean_p3**2))
#         correlation = scipy.signal.correlate(filtered_real_data, strain_clean_norm, mode='same')
        
#         noise_region = correlation[:int(4 * 4096)]
#         snr_value = np.max(np.abs(correlation)) / np.std(np.abs(noise_region))
#         snr_results.append((m1_test, m2_test, snr_value))

# best_snr = max(snr_results, key=lambda x: x[2])
# print(f"Best masses: m1 = {best_snr[0]} M0,  m2 = {best_snr[1]} M0, SNR = {best_snr[2]:.2f}")


# 2nd Method: Fitting the frequency function to the data to extract the chirp mass
from scipy.signal import hilbert # Hilbert transform to get the instantaneous phase and frequency
analytic_signal = hilbert(filtered_real_data)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = np.diff(instantaneous_phase) * 4096 / (2 * np.pi)

mask = (instantaneous_frequency > 20) & (instantaneous_frequency < 500)
t_inst = t_plot[:-1][mask]
f_inst = instantaneous_frequency[mask]

# plt.figure(figsize=(12, 6))
# plt.plot(t_inst, f_inst, color=C_CORR)
# plt.title('Phase 3: Instantaneous Frequency of GW150914 H1 Strain (Hilbert Transform)')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.xlim(8, 8.45)
# plt.ylim(0, 500)
# plt.grid()
# plt.show()

f_spec, t_spec, Sxx = scipy.signal.spectrogram(filtered_real_data, fs=4096, nperseg=128, noverlap=120)
freq_mask = f_spec > 50
peak_freq = f_spec[freq_mask][np.argmax(Sxx[freq_mask], axis=0)]

time_mask = (t_spec + t_plot[0] > 8.38) & (t_spec + t_plot[0] < 8.43)
t_fit = t_spec[time_mask] + t_plot[0]
f_fit = peak_freq[time_mask]
print(t_fit)
print(f_fit)


def freq_model(t, M, t_merger):
    tau = np.maximum(t_merger - t, 1e-4)
    return Frequency_calc(M, tau)

p0 = [Chirp_Mass_calc(29, 36), 8.43]  # initial guess: known masses, merger time
popt, pcov = curve_fit(freq_model, t_fit, f_fit, p0=p0)
print(f"Fitted chirp mass: {popt[0]/1.989e30:.2f} solar masses")
print(f"Fitted merger time: {popt[1]:.4f} s")

perr = np.sqrt(np.diag(pcov))
print(f"Chirp mass: {popt[0]/1.989e30:.2f} ± {perr[0]/1.989e30:.2f} solar masses")
print(f"Merger time: {popt[1]:.4f} ± {perr[1]:.4f} s")

t_smooth = np.linspace(t_fit[0], t_fit[-1], 200)
f_smooth = freq_model(t_smooth, *popt)


plt.scatter(t_fit, f_fit, color=C_SIGNAL, label='Spectrogram peak frequencies', zorder=5)
plt.plot(t_smooth, f_smooth, color=C_TEMPLATE, lw=1.5, label=f'Newtonian fit  M = {popt[0]/1.989e30:.2f} ± {perr[0]/1.989e30:.2f} M0')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('Chirp Mass Extraction — Frequency Evolution Fit')
plt.legend()
plt.grid()
plt.show()
