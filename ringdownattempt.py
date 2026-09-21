import scipy
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def Chirp_Mass_calc(m1, m2):
    """
    Calculates the chirp mass from two component masses in solar masses.
    Returns chirp mass in kg.
    """
    M0 = 1.989e30
    chirp_mass = ((m1 * m2 * M0**2) ** (3 / 5)) / (((m1 + m2) * M0) ** (1 / 5))
    return chirp_mass


def Frequency_calc(M, tau):
    """
    GW frequency as a function of chirp mass M and time-to-coalescence tau.
    """
    c = 299792458.0
    G = 6.67430e-11

    return (1 / np.pi) * (
        (5 * c**5) / (256 * G ** (5 / 3) * M ** (5 / 3) * tau)
    ) ** (3 / 8)


def Phase_calc(M, tau):
    """
    Accumulated GW phase as a function of chirp mass M and time-to-coalescence tau.
    """
    c = 299792458.0
    G = 6.67430e-11

    return -2 * ((c**3) / (5 * G * M)) ** (5 / 8) * tau ** (5 / 8)


def Strain_calc(A, frequency, phi):
    """
    Simplified GW strain model.
    """
    return A * frequency ** (2 / 3) * np.cos(phi)


def Schwarzschild_radius(mass):
    """
    Schwarzschild radius for a given mass in kg.
    """
    G = 6.67430e-11
    c = 299792458.0

    return (2 * G * mass) / c**2


def Keplers_third_law_separation(m1, m2, frequency):
    """
    Orbital separation from GW frequency using Kepler's Third Law.

    m1, m2 are in solar masses.
    frequency is the GW frequency.
    """
    G = 6.67430e-11
    M0 = 1.989e30

    # GW frequency is twice orbital frequency, so orbital period = 2 / f_gw
    period = 2 / frequency

    return (G * (m1 + m2) * M0 * period**2 / (4 * np.pi**2)) ** (1 / 3)


def Add_ringdown(
    t_clean,
    strain_clean,
    sample_rate,
    ringdown_duration=0.45,
    tau_fast=0.06,
    tau_slow=0.22,
    switch_time=0.08,
    f_ringdown=35,
    amplitude_scale=2.0
):
    """
    Adds a lower-frequency, higher-amplitude two-stage ringdown.

    The ringdown:
    - starts with amplitude similar to the final chirp
    - oscillates slowly
    - drops quickly at first
    - then decays more gradually

    This is a toy model, not full numerical relativity.
    """

    dt = 1 / sample_rate
    t_ringdown = np.arange(dt, ringdown_duration, dt)

    # Use the strongest part of the final chirp to set ringdown amplitude.
    end_points = int(0.10 * sample_rate)
    end_points = min(end_points, len(strain_clean))

    tail = strain_clean[-end_points:]
    A_tail = np.max(np.abs(tail))

    if A_tail == 0:
        A_tail = 1e-30

    A_ringdown = amplitude_scale * A_tail

    # Match the first ringdown value roughly to the final inspiral value.
    y0 = strain_clean[-1]
    y0_clipped = np.clip(y0 / A_ringdown, -1, 1)
    phi0 = np.arccos(y0_clipped)

    # Try to make slope direction continuous.
    if len(strain_clean) >= 2:
        dy_last = strain_clean[-1] - strain_clean[-2]

        derivative_guess = (
            -A_ringdown / tau_fast * np.cos(phi0)
            - 2 * np.pi * f_ringdown * A_ringdown * np.sin(phi0)
        )

        if np.sign(derivative_guess) != np.sign(dy_last):
            phi0 = -phi0

    # Two-stage decay envelope
    envelope = np.zeros_like(t_ringdown)

    early = t_ringdown <= switch_time
    late = t_ringdown > switch_time

    envelope[early] = np.exp(-t_ringdown[early] / tau_fast)

    envelope_at_switch = np.exp(-switch_time / tau_fast)
    envelope[late] = envelope_at_switch * np.exp(
        -(t_ringdown[late] - switch_time) / tau_slow
    )

    ringdown = A_ringdown * envelope * np.cos(
        2 * np.pi * f_ringdown * t_ringdown + phi0
    )

    t_full = np.concatenate([
        t_clean,
        t_clean[-1] + t_ringdown
    ])

    strain_full = np.concatenate([
        strain_clean,
        ringdown
    ])

    return t_full, strain_full, t_ringdown, ringdown


def Generate_chirp_with_ringdown(
    m1,
    m2,
    sample_rate=4096,
    duration=2,
    A=1e-21,
    ringdown_duration=0.45,
    ringdown_tau_fast=0.06,
    ringdown_tau_slow=0.22,
    ringdown_switch_time=0.08,
    f_ringdown=35,
    ringdown_amplitude_scale=2.0
):
    """
    Generates a clean inspiral chirp, trims at the crash point,
    then appends a lower-frequency two-stage ringdown.
    """

    M0 = 1.989e30

    mass1 = m1 * M0
    mass2 = m2 * M0

    M = Chirp_Mass_calc(m1, m2)
    r1 = Schwarzschild_radius(mass1)
    r2 = Schwarzschild_radius(mass2)

    t = np.linspace(0, duration, int(sample_rate * duration))
    tau = np.maximum(duration - t, 1e-4)

    f = Frequency_calc(M, tau)
    phi = Phase_calc(M, tau)
    strain = Strain_calc(A, f, phi)

    # Find crash point
    sep = Keplers_third_law_separation(m1, m2, f)
    crash_idx = np.where(sep <= r1 + r2)[0]

    if len(crash_idx) > 0:
        crash_index = crash_idx[0]
        t_clean = t[:crash_index]
        strain_clean = strain[:crash_index]
        f_clean = f[:crash_index]
    else:
        t_clean = t
        strain_clean = strain
        f_clean = f

    # Add ringdown
    t_full, strain_full, t_ringdown, ringdown = Add_ringdown(
        t_clean=t_clean,
        strain_clean=strain_clean,
        sample_rate=sample_rate,
        ringdown_duration=ringdown_duration,
        tau_fast=ringdown_tau_fast,
        tau_slow=ringdown_tau_slow,
        switch_time=ringdown_switch_time,
        f_ringdown=f_ringdown,
        amplitude_scale=ringdown_amplitude_scale
    )

    return {
        "t_clean": t_clean,
        "strain_clean": strain_clean,
        "f_clean": f_clean,
        "t_full": t_full,
        "strain_full": strain_full,
        "t_ringdown": t_ringdown,
        "ringdown": ringdown,
        "crash_time": t_clean[-1],
    }


def bandpass_filter(signal, sample_rate, lowcut=20.0, highcut=300.0, order=4):
    """
    Applies a Butterworth bandpass filter using second-order sections.
    This is more stable than ordinary b, a filtering.
    """
    nyquist = 0.5 * sample_rate

    if highcut >= nyquist:
        raise ValueError("Highcut frequency must be below the Nyquist frequency.")

    sos = scipy.signal.butter(
        order,
        [lowcut / nyquist, highcut / nyquist],
        btype="band",
        output="sos"
    )

    return scipy.signal.sosfiltfilt(sos, signal)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

m1_p1 = 29
m2_p1 = 36

sample_rate = 4096
duration = 2
A = 1e-21

# Ringdown parameters
ringdown_duration = 0.45
ringdown_tau_fast = 0.06
ringdown_tau_slow = 0.22
ringdown_switch_time = 0.08
f_ringdown = 35
ringdown_amplitude_scale = 2.0

# Use different filters for synthetic and real data
SYNTH_LOW = 5.0
SYNTH_HIGH = 800.0

REAL_LOW = 20.0
REAL_HIGH = 300.0


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: CLEAN CHIRP + RINGDOWN
# ─────────────────────────────────────────────────────────────────────────────

phase1 = Generate_chirp_with_ringdown(
    m1=m1_p1,
    m2=m2_p1,
    sample_rate=sample_rate,
    duration=duration,
    A=A,
    ringdown_duration=ringdown_duration,
    ringdown_tau_fast=ringdown_tau_fast,
    ringdown_tau_slow=ringdown_tau_slow,
    ringdown_switch_time=ringdown_switch_time,
    f_ringdown=f_ringdown,
    ringdown_amplitude_scale=ringdown_amplitude_scale
)

t_clean_p1 = phase1["t_clean"]
strain_clean_p1 = phase1["strain_clean"]

t_full_p1 = phase1["t_full"]
strain_full_p1 = phase1["strain_full"]

plt.figure(figsize=(12, 6))
plt.plot(t_clean_p1, strain_clean_p1, label="Inspiral chirp", linewidth=2)
plt.plot(t_full_p1, strain_full_p1, label="Inspiral + ringdown", linewidth=2, alpha=0.8)
plt.axvline(phase1["crash_time"], color="red", linestyle="--", label="Merger / start of ringdown")

plt.title("Phase 1: Clean Newtonian Chirp With Ringdown")
plt.xlabel("Time [s]")
plt.ylabel("Strain")
plt.legend()
plt.grid()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: SYNTHETIC SIGNAL PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

sample_rates_to_test = [4096, 2048, 1024, 512, 256, 128, 64]

for rate in sample_rates_to_test:

    model = Generate_chirp_with_ringdown(
        m1=m1_p1,
        m2=m2_p1,
        sample_rate=rate,
        duration=duration,
        A=A,
        ringdown_duration=ringdown_duration,
        ringdown_tau_fast=ringdown_tau_fast,
        ringdown_tau_slow=ringdown_tau_slow,
        ringdown_switch_time=ringdown_switch_time,
        f_ringdown=f_ringdown,
        ringdown_amplitude_scale=ringdown_amplitude_scale
    )

    strain_full = model["strain_full"]

    # Add white noise
    static = np.random.normal(0, 1e-21, len(strain_full))
    strain_noisy = strain_full + static

    # Nyquist safety check
    nyquist = 0.5 * rate

    if nyquist <= SYNTH_HIGH:
        print(f"Sample Rate: {rate} Hz FAILED: framerate too low for an {SYNTH_HIGH:.0f} Hz filter.")
        continue

    # Wide filter for synthetic data so the early chirp is not crushed
    filtered_strain = bandpass_filter(
        strain_noisy,
        sample_rate=rate,
        lowcut=SYNTH_LOW,
        highcut=SYNTH_HIGH
    )

    filtered_template = bandpass_filter(
        strain_full,
        sample_rate=rate,
        lowcut=SYNTH_LOW,
        highcut=SYNTH_HIGH
    )

    # Matched filter
    correlation_scores = scipy.signal.correlate(
        filtered_strain,
        filtered_template,
        mode="same"
    )

    max_correlation = np.max(np.abs(correlation_scores))

    print(f"Sample Rate: {rate} Hz | Peak Correlation Score: {max_correlation:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 4096 Hz SYNTHETIC PLOTS
# ─────────────────────────────────────────────────────────────────────────────

baseline = Generate_chirp_with_ringdown(
    m1=m1_p1,
    m2=m2_p1,
    sample_rate=4096,
    duration=duration,
    A=A,
    ringdown_duration=ringdown_duration,
    ringdown_tau_fast=ringdown_tau_fast,
    ringdown_tau_slow=ringdown_tau_slow,
    ringdown_switch_time=ringdown_switch_time,
    f_ringdown=f_ringdown,
    ringdown_amplitude_scale=ringdown_amplitude_scale
)

t_full = baseline["t_full"]
strain_full = baseline["strain_full"]

static = np.random.normal(0, 1e-21, len(strain_full))
strain_noisy = strain_full + static

# FFT plot
fft_strain = scipy.fft.fft(strain_noisy)
dt = 1 / 4096

freqs = scipy.fft.fftfreq(len(strain_full), dt)
pos_freqs = freqs[freqs >= 0]
pos_fft = fft_strain[freqs >= 0]

plt.figure(figsize=(12, 6))
plt.loglog(pos_freqs, np.abs(pos_fft))
plt.title("Phase 2: FFT of Synthetic Chirp + Ringdown With Noise")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude of FFT")
plt.xlim(1, 1000)
plt.grid()
plt.show()

# Time domain noisy signal
plt.figure(figsize=(12, 6))
plt.plot(t_full, strain_noisy)
plt.axvline(baseline["crash_time"], color="red", linestyle="--", label="Merger / start of ringdown")

plt.title("Phase 2: Synthetic Chirp + Ringdown Buried in White Noise")
plt.xlabel("Time [s]")
plt.ylabel("Strain")
plt.legend()
plt.grid()
plt.show()

# Filter baseline synthetic signal with wide synthetic filter
filtered_strain = bandpass_filter(
    strain_noisy,
    sample_rate=4096,
    lowcut=SYNTH_LOW,
    highcut=SYNTH_HIGH
)

filtered_clean_template = bandpass_filter(
    strain_full,
    sample_rate=4096,
    lowcut=SYNTH_LOW,
    highcut=SYNTH_HIGH
)

plt.figure(figsize=(12, 6))
plt.plot(t_full, filtered_clean_template, label="Clean filtered synthetic chirp + ringdown", linewidth=2)
plt.plot(t_full, filtered_strain, label="Noisy filtered synthetic chirp + ringdown", linewidth=2, alpha=0.8)
plt.axvline(baseline["crash_time"], color="red", linestyle="--", label="Merger / start of ringdown")

plt.title("Phase 2: Clean Filtered vs Noisy Filtered Synthetic Chirp")
plt.xlabel("Time [s]")
plt.ylabel("Strain")
plt.legend()
plt.grid()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: REAL OBSERVATORY DATA — GW150914
# ─────────────────────────────────────────────────────────────────────────────

print("Fetching GW150914 H1 strain data from GWOSC...")

real_data = TimeSeries.fetch_open_data(
    "H1",
    start=1126259462 - 8,
    end=1126259462 + 8,
    sample_rate=4096,
    cache=True
)

t_plot = real_data.times.value - real_data.times.value[0]

# Whiten real data
whitened_real_data = real_data.whiten(fftlength=4, overlap=2)

# Filter real data with real-data display band
filtered_real_data = bandpass_filter(
    whitened_real_data.value,
    sample_rate=4096,
    lowcut=REAL_LOW,
    highcut=REAL_HIGH
)

plt.figure(figsize=(12, 6))
plt.plot(t_plot, filtered_real_data, label="Filtered H1 strain data")
plt.title("Phase 3: GW150914 H1 Strain — Whitened & Bandpassed")
plt.xlabel("Time [s]")
plt.ylabel("Strain (whitened)")
plt.xlim(8, 8.5)
plt.legend()
plt.grid()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BUILD SYNTHETIC TEMPLATE FOR OVERLAY USING CROSS-CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

template_model = Generate_chirp_with_ringdown(
    m1=29,
    m2=36,
    sample_rate=4096,
    duration=duration,
    A=1e-21,
    ringdown_duration=ringdown_duration,
    ringdown_tau_fast=ringdown_tau_fast,
    ringdown_tau_slow=ringdown_tau_slow,
    ringdown_switch_time=ringdown_switch_time,
    f_ringdown=f_ringdown,
    ringdown_amplitude_scale=ringdown_amplitude_scale
)

t_template = template_model["t_full"]
strain_template = template_model["strain_full"]
template_merger_time = template_model["crash_time"]

# Filter the template with the same display band as real LIGO data
filtered_template = bandpass_filter(
    strain_template,
    sample_rate=4096,
    lowcut=REAL_LOW,
    highcut=REAL_HIGH
)


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAY MODEL + RINGDOWN ON LIGO CHIRP USING PEAK ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

event_start = 8.30
event_end = 8.50

event_mask = (t_plot >= event_start) & (t_plot <= event_end)
t_event = t_plot[event_mask]
ligo_event = filtered_real_data[event_mask]

# Find the strongest peak in the real LIGO event window
real_peak_index = np.argmax(np.abs(ligo_event))
real_peak_time = t_event[real_peak_index]
real_peak_amplitude = np.max(np.abs(ligo_event))

print(f"Real LIGO peak time: {real_peak_time:.5f} s")
print(f"Real LIGO peak amplitude: {real_peak_amplitude:.3f}")

# Put template time relative to its own merger
t_template_relative = t_template - template_merger_time

# Align model merger/start of ringdown to real LIGO peak
model_merger_time = real_peak_time
t_template_aligned = t_template_relative + model_merger_time

# Interpolate aligned template onto the LIGO event time grid
template_on_event_grid = np.interp(
    t_event,
    t_template_aligned,
    filtered_template,
    left=0,
    right=0
)

# Check overlap
template_energy = np.dot(template_on_event_grid, template_on_event_grid)

print(f"Template overlap energy in event window: {template_energy:.3e}")

if template_energy == 0:
    raise ValueError(
        "Template still has zero overlap. Try increasing event_start/event_end or check template time range."
    )

# Scale model amplitude to match the real LIGO peak
template_peak_amplitude = np.max(np.abs(template_on_event_grid))

if template_peak_amplitude == 0:
    raise ValueError("Template peak amplitude is zero after alignment.")

scale = real_peak_amplitude / template_peak_amplitude

filtered_template_scaled = scale * filtered_template

print(f"Template peak amplitude before scaling: {template_peak_amplitude:.3e}")
print(f"Amplitude scale factor: {scale:.3e}")

# Plot overlay
plt.figure(figsize=(12, 6))

plt.plot(
    t_plot,
    filtered_real_data,
    label="Filtered LIGO H1 data",
    linewidth=1.5
)

plt.plot(
    t_template_aligned,
    filtered_template_scaled,
    label="Model chirp + ringdown",
    linewidth=2,
    alpha=0.85
)

plt.axvline(
    model_merger_time,
    color="red",
    linestyle="--",
    label="Aligned model merger"
)

plt.title("Model Chirp + Ringdown Overlaid on GW150914")
plt.xlabel("Time [s]")
plt.ylabel("Strain (whitened)")
plt.xlim(event_start, event_end)
plt.legend()
plt.grid()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISED SHAPE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

real_norm = ligo_event / np.max(np.abs(ligo_event))

template_event_scaled = np.interp(
    t_event,
    t_template_aligned,
    filtered_template_scaled,
    left=0,
    right=0
)

template_norm_plot = template_event_scaled / np.max(np.abs(template_event_scaled))

plt.figure(figsize=(12, 6))

plt.plot(
    t_event,
    real_norm,
    label="LIGO event normalised",
    linewidth=1.5
)

plt.plot(
    t_event,
    template_norm_plot,
    label="Model normalised",
    linewidth=2,
    alpha=0.85
)

plt.axvline(
    model_merger_time,
    color="red",
    linestyle="--",
    label="Aligned model merger"
)

plt.title("Normalised Overlay: Model Chirp + Ringdown vs GW150914")
plt.xlabel("Time [s]")
plt.ylabel("Normalised strain")
plt.xlim(event_start, event_end)
plt.legend()
plt.grid()
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MATCHED FILTER DEMONSTRATION
# ─────────────────────────────────────────────────────────────────────────────

correlation = scipy.signal.correlate(
    filtered_real_data,
    filtered_template,
    mode="same"
)

plt.figure(figsize=(12, 6))
plt.plot(t_plot, np.abs(correlation))
plt.title("Matched Filter Demonstration — Toy Template vs Real Data")
plt.xlabel("Time [s]")
plt.ylabel("Correlation")
plt.grid()
plt.show()

peak = np.max(np.abs(correlation))
snr = peak / np.std(np.abs(correlation))

print(f"Simplified SNR estimate: {snr:.2f}")
print("Note: this is not the official LIGO matched-filter SNR.")
print("The toy Newtonian model is for qualitative comparison, not exact phase matching.")
