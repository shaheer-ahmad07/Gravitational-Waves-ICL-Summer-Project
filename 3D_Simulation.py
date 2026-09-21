import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# PHYSICS FUNCTIONS FROM YOUR NEWTONIAN MODEL
# ============================================================

def Chirp_Mass_calc(m1, m2):
    M0 = 1.989e30
    return ((m1 * m2 * M0**2) ** (3 / 5)) / (((m1 + m2) * M0) ** (1 / 5))


def Frequency_calc(M, tau):
    c = 299792458.0
    G = 6.67430e-11

    return (1 / np.pi) * (
        (5 * c**5) / (256 * G ** (5 / 3) * M ** (5 / 3) * tau)
    ) ** (3 / 8)


def Phase_calc(M, tau):
    c = 299792458.0
    G = 6.67430e-11

    return -2 * ((c**3) / (5 * G * M)) ** (5 / 8) * tau ** (5 / 8)


def Strain_calc(A, frequency, phi):
    return A * frequency ** (2 / 3) * np.cos(phi)


def Schwarzschild_radius(mass_kg):
    G = 6.67430e-11
    c = 299792458.0

    return (2 * G * mass_kg) / c**2


def Keplers_third_law_separation(m1, m2, frequency):
    G = 6.67430e-11
    M0 = 1.989e30

    # GW frequency is twice orbital frequency
    period = 2 / frequency

    return (G * (m1 + m2) * M0 * period**2 / (4 * np.pi**2)) ** (1 / 3)


def add_ringdown_to_waveform(
    strain_inspiral,
    sample_rate,
    ringdown_duration,
    ringdown_frequency=70,
    tau_fast=0.14,
    tau_slow=2.0,
    switch_time=0.18,
    amplitude_scale=0.9
):
    t_rd = np.arange(0, ringdown_duration, 1 / sample_rate)

    tail_length = min(len(strain_inspiral), int(0.08 * sample_rate))
    tail = strain_inspiral[-tail_length:]

    A_rms = np.sqrt(np.mean(tail**2))
    A_peak = np.max(np.abs(tail))

    A_rd = amplitude_scale * (0.50 * A_peak + 0.50 * A_rms)

    if A_rd == 0:
        A_rd = 1e-30

    y0 = strain_inspiral[-1]
    y0_clipped = np.clip(y0 / A_rd, -1, 1)
    phi0 = np.arccos(y0_clipped)

    envelope = np.zeros_like(t_rd)

    early = t_rd <= switch_time
    late = t_rd > switch_time

    envelope[early] = np.exp(-t_rd[early] / tau_fast)

    envelope_at_switch = np.exp(-switch_time / tau_fast)
    envelope[late] = envelope_at_switch * np.exp(
        -(t_rd[late] - switch_time) / tau_slow
    )

    ringdown = A_rd * envelope * np.cos(
        2 * np.pi * ringdown_frequency * t_rd + phi0
    )

    return t_rd, ringdown


# ============================================================
# MAIN PARAMETERS
# ============================================================

m1 = 36
m2 = 29

M0 = 1.989e30
mass1 = m1 * M0
mass2 = m2 * M0

M_chirp = Chirp_Mass_calc(m1, m2)

r1_s = Schwarzschild_radius(mass1)
r2_s = Schwarzschild_radius(mass2)
r_crash = r1_s + r2_s

animation_duration = 12
fps = 30
frames = animation_duration * fps
wave_sample_rate = 4096

merger_time_video = 8.5
ringdown_video_duration = animation_duration - merger_time_video

model_duration = 2.0

t_model = np.linspace(0, model_duration, int(wave_sample_rate * model_duration))
tau_model = np.maximum(model_duration - t_model, 1e-4)

f_gw_model = Frequency_calc(M_chirp, tau_model)
phi_model = Phase_calc(M_chirp, tau_model)
strain_model = Strain_calc(1e-21, f_gw_model, phi_model)

separation_model = Keplers_third_law_separation(m1, m2, f_gw_model)
crash_indices = np.where(separation_model <= r_crash)[0]

if len(crash_indices) > 0:
    crash_index = crash_indices[0]
else:
    crash_index = len(t_model) - 1

t_inspiral_model = t_model[:crash_index]
f_gw_inspiral = f_gw_model[:crash_index]
strain_inspiral = strain_model[:crash_index]
separation_inspiral = separation_model[:crash_index]

# Normalise inspiral strain
normalisation = np.max(np.abs(strain_inspiral))
if normalisation == 0:
    normalisation = 1

strain_inspiral = strain_inspiral / normalisation

# Add ringdown
t_rd_model, ringdown = add_ringdown_to_waveform(
    strain_inspiral=strain_inspiral,
    sample_rate=wave_sample_rate,
    ringdown_duration=0.75,
    ringdown_frequency=70,
    tau_fast=0.14,
    tau_slow=2.0,
    switch_time=0.18,
    amplitude_scale=0.9
)

strain_full_model = np.concatenate([strain_inspiral, ringdown])
t_full_model = np.concatenate([
    t_inspiral_model,
    t_inspiral_model[-1] + t_rd_model
])


# ============================================================
# MAP MODEL TIME TO VIDEO TIME
# ============================================================

video_t = np.linspace(0, animation_duration, frames)

model_to_video_time = np.zeros_like(t_full_model)

inspiral_mask = t_full_model <= t_inspiral_model[-1]
ringdown_mask = t_full_model > t_inspiral_model[-1]

model_to_video_time[inspiral_mask] = (
    t_full_model[inspiral_mask] / t_inspiral_model[-1]
) * merger_time_video

if np.any(ringdown_mask):
    model_ringdown_time = t_full_model[ringdown_mask] - t_inspiral_model[-1]
    model_to_video_time[ringdown_mask] = (
        merger_time_video
        + model_ringdown_time / np.max(model_ringdown_time) * ringdown_video_duration
    )

chirp_video = np.interp(video_t, model_to_video_time, strain_full_model)

merger_frame = np.argmin(np.abs(video_t - merger_time_video))


# ============================================================
# 3D ORBIT MODEL
# ============================================================

video_t_inspiral = video_t[:merger_frame]

model_inspiral_video_time = np.linspace(
    0,
    merger_time_video,
    len(separation_inspiral)
)

sep_video_physical = np.interp(
    video_t_inspiral,
    model_inspiral_video_time,
    separation_inspiral
)

f_gw_video = np.interp(
    video_t_inspiral,
    model_inspiral_video_time,
    f_gw_inspiral
)

f_orb_video = f_gw_video / 2

# Increase this to make orbit faster
visual_orbit_scale = 0.1
f_orb_visual = f_orb_video * visual_orbit_scale

dt_video = video_t[1] - video_t[0]
theta = 2 * np.pi * np.cumsum(f_orb_visual) * dt_video

sep_visual = sep_video_physical / np.max(sep_video_physical)
sep_visual = 5.2 * sep_visual
sep_visual = np.maximum(sep_visual, 0.45)

M_total = m1 + m2
r1_factor = m2 / M_total
r2_factor = m1 / M_total

x1_inspiral = r1_factor * sep_visual * np.cos(theta)
y1_inspiral = r1_factor * sep_visual * np.sin(theta)
z1_inspiral = 0.15 * np.sin(2 * theta)

x2_inspiral = -r2_factor * sep_visual * np.cos(theta)
y2_inspiral = -r2_factor * sep_visual * np.sin(theta)
z2_inspiral = -0.15 * np.sin(2 * theta)

x1 = np.zeros(frames)
y1 = np.zeros(frames)
z1 = np.zeros(frames)

x2 = np.zeros(frames)
y2 = np.zeros(frames)
z2 = np.zeros(frames)

x1[:merger_frame] = x1_inspiral
y1[:merger_frame] = y1_inspiral
z1[:merger_frame] = z1_inspiral

x2[:merger_frame] = x2_inspiral
y2[:merger_frame] = y2_inspiral
z2[:merger_frame] = z2_inspiral


# ============================================================
# FIGURE SETUP
# ============================================================

fig = plt.figure(figsize=(12, 9))

ax3d = fig.add_subplot(2, 1, 1, projection="3d")
ax_wave = fig.add_subplot(2, 1, 2)


# ============================================================
# DARK / BLACK BACKGROUND STYLE
# ============================================================

fig.patch.set_facecolor("black")

# 3D panel background
ax3d.set_facecolor("black")

# Remove grey 3D panes
ax3d.xaxis.pane.set_facecolor((0, 0, 0, 1))
ax3d.yaxis.pane.set_facecolor((0, 0, 0, 1))
ax3d.zaxis.pane.set_facecolor((0, 0, 0, 1))

# Make pane edges subtle
ax3d.xaxis.pane.set_edgecolor((1, 1, 1, 0.25))
ax3d.yaxis.pane.set_edgecolor((1, 1, 1, 0.25))
ax3d.zaxis.pane.set_edgecolor((1, 1, 1, 0.25))

# Axis labels and ticks white
ax3d.xaxis.label.set_color("white")
ax3d.yaxis.label.set_color("white")
ax3d.zaxis.label.set_color("white")
ax3d.tick_params(colors="white")

# 3D grid lines subtle
ax3d.xaxis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.25)
ax3d.yaxis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.25)
ax3d.zaxis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.25)

# Waveform panel background
ax_wave.set_facecolor("black")
ax_wave.tick_params(colors="white")
ax_wave.xaxis.label.set_color("white")
ax_wave.yaxis.label.set_color("white")

# Waveform grid
ax_wave.grid(True, color="white", alpha=0.15)

# Make plot borders white
for spine in ax_wave.spines.values():
    spine.set_color("white")


fig.suptitle(
    "3D Binary Black Hole Inspiral and Gravitational Wave Chirp",
    fontsize=15,
    fontweight="bold",
    color="white"
)


# ============================================================
# 3D ORBIT PANEL
# ============================================================

ax3d.set_title("3D Inspiral and Merger", color="white")
ax3d.set_xlim(-6, 6)
ax3d.set_ylim(-6, 6)
ax3d.set_zlim(-3, 3)

ax3d.set_xlabel("x", color="white")
ax3d.set_ylabel("y", color="white")
ax3d.set_zlabel("z", color="white")

ax3d.view_init(elev=25, azim=45)

bh1 = ax3d.scatter(
    [],
    [],
    [],
    s=260,
    color="deepskyblue",
    label=f"Black hole 1 ({m1} M☉)"
)

bh2 = ax3d.scatter(
    [],
    [],
    [],
    s=220,
    color="orange",
    label=f"Black hole 2 ({m2} M☉)"
)

merged_bh = ax3d.scatter(
    [],
    [],
    [],
    s=420,
    color="white",
    label="Merged black hole"
)

trail1, = ax3d.plot(
    [],
    [],
    [],
    linewidth=1.2,
    alpha=0.8,
    color="deepskyblue"
)

trail2, = ax3d.plot(
    [],
    [],
    [],
    linewidth=1.2,
    alpha=0.8,
    color="orange"
)

# Gravitational wave rings in 3D
ring_lines = []
angle = np.linspace(0, 2 * np.pi, 200)

for _ in range(8):
    line, = ax3d.plot(
        [],
        [],
        [],
        alpha=0.25,
        color="cyan"
    )
    ring_lines.append(line)

stage_text = ax3d.text2D(
    0.03,
    0.90,
    "",
    transform=ax3d.transAxes,
    fontsize=13,
    color="white",
    bbox=dict(facecolor="black", edgecolor="white", alpha=0.75)
)

legend3d = ax3d.legend(loc="upper right")
legend3d.get_frame().set_facecolor("black")
legend3d.get_frame().set_edgecolor("white")

for text in legend3d.get_texts():
    text.set_color("white")


# ============================================================
# WAVEFORM PANEL
# ============================================================

ax_wave.set_xlim(0, animation_duration)
ax_wave.set_ylim(-1.25, 1.25)
ax_wave.set_title("Newtonian Model Chirp + Ringdown", color="white")
ax_wave.set_xlabel("Time [s]", color="white")
ax_wave.set_ylabel("Normalised strain", color="white")
ax_wave.grid(True, color="white", alpha=0.15)

wave_line, = ax_wave.plot(
    [],
    [],
    linewidth=2,
    color="lime"
)

current_time_line = ax_wave.axvline(
    0,
    linestyle="--",
    alpha=0.7,
    color="white"
)

ax_wave.axvline(
    merger_time_video,
    color="red",
    linestyle="--",
    alpha=0.8,
    label="Merger"
)

legend_wave = ax_wave.legend(loc="upper right")
legend_wave.get_frame().set_facecolor("black")
legend_wave.get_frame().set_edgecolor("white")

for text in legend_wave.get_texts():
    text.set_color("white")


# ============================================================
# UPDATE FUNCTION
# ============================================================

def update(frame):
    current_time = video_t[frame]

    if frame < merger_frame:
        bh1._offsets3d = ([x1[frame]], [y1[frame]], [z1[frame]])
        bh2._offsets3d = ([x2[frame]], [y2[frame]], [z2[frame]])
        merged_bh._offsets3d = ([], [], [])

        stage_text.set_text("Inspiral")

    else:
        bh1._offsets3d = ([], [], [])
        bh2._offsets3d = ([], [], [])
        merged_bh._offsets3d = ([0], [0], [0])

        if frame < merger_frame + int(1.2 * fps):
            stage_text.set_text("Merger")
        else:
            stage_text.set_text("Ringdown")

    # Trails
    trail_length = 100
    start = max(0, frame - trail_length)

    if frame < merger_frame:
        trail1.set_data(x1[start:frame], y1[start:frame])
        trail1.set_3d_properties(z1[start:frame])

        trail2.set_data(x2[start:frame], y2[start:frame])
        trail2.set_3d_properties(z2[start:frame])
    else:
        trail_start = max(0, merger_frame - trail_length)

        trail1.set_data(x1[trail_start:merger_frame], y1[trail_start:merger_frame])
        trail1.set_3d_properties(z1[trail_start:merger_frame])

        trail2.set_data(x2[trail_start:merger_frame], y2[trail_start:merger_frame])
        trail2.set_3d_properties(z2[trail_start:merger_frame])

    # Gravitational wave rings
    for i, line in enumerate(ring_lines):
        if frame < merger_frame:
            r = ((frame / frames) * 13 - i * 1.4) % 8
            alpha = max(0.04, 0.22 * (1 - r / 8))
        else:
            time_since_merger = current_time - merger_time_video
            r = (time_since_merger * 3.2 - i * 0.75) % 8
            alpha = max(0.04, 0.34 * (1 - r / 8))

        x_ring = r * np.cos(angle)
        y_ring = r * np.sin(angle)
        z_ring = np.zeros_like(angle)

        line.set_data(x_ring, y_ring)
        line.set_3d_properties(z_ring)
        line.set_alpha(alpha)

    # Slowly rotate camera
    ax3d.view_init(elev=25, azim=45 + 0.12 * frame)

    # Waveform
    wave_line.set_data(video_t[:frame], chirp_video[:frame])
    current_time_line.set_xdata([current_time, current_time])

    return [
        bh1,
        bh2,
        merged_bh,
        trail1,
        trail2,
        wave_line,
        current_time_line,
        stage_text,
        *ring_lines
    ]


# ============================================================
# CREATE ANIMATION
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=1000 / fps,
    blit=False
)

plt.show()


# ============================================================
# OPTIONAL SAVE
# ============================================================

# To save as MP4, install ffmpeg.
#
# If ffmpeg does not work, save as GIF instead:
#
# animation.save(
#     "black_hole_3d_inspiral_chirp.gif",
#     writer="pillow",
#     fps=fps,
#     dpi=120
# )
#
# For MP4 after installing ffmpeg:
#
# animation.save(
#     "black_hole_3d_inspiral_chirp.mp4",
#     writer="ffmpeg",
#     fps=fps,
#     dpi=200
# )
