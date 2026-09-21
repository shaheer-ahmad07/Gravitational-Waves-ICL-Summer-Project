import numpy as np
import pandas as pd
import plotly.graph_objects as go


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
fps = 20
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

# Ringdown
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

# Increase this for faster visible orbit
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
z1_inspiral = 0.18 * np.sin(2 * theta)

x2_inspiral = -r2_factor * sep_visual * np.cos(theta)
y2_inspiral = -r2_factor * sep_visual * np.sin(theta)
z2_inspiral = -0.18 * np.sin(2 * theta)

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

# After merger, one object at centre
x1[merger_frame:] = 0
y1[merger_frame:] = 0
z1[merger_frame:] = 0

x2[merger_frame:] = 0
y2[merger_frame:] = 0
z2[merger_frame:] = 0


# ============================================================
# BUILD PLOTLY FRAMES
# ============================================================

plotly_frames = []

trail_length = 45

angle_ring = np.linspace(0, 2 * np.pi, 120)

for frame in range(frames):
    start = max(0, frame - trail_length)

    current_time = video_t[frame]

    # Stage
    if frame < merger_frame:
        stage = "Inspiral"
        bh1_visible_size = 9
        bh2_visible_size = 8
        merged_size = 0
    else:
        stage = "Merger / Ringdown"
        bh1_visible_size = 0
        bh2_visible_size = 0
        merged_size = 14

    # Wave rings
    ring_traces = []

    for i in range(5):
        if frame < merger_frame:
            r = ((frame / frames) * 13 - i * 1.5) % 8
            opacity = max(0.05, 0.22 * (1 - r / 8))
        else:
            time_since_merger = current_time - merger_time_video
            r = (time_since_merger * 3.2 - i * 0.9) % 8
            opacity = max(0.05, 0.35 * (1 - r / 8))

        x_ring = r * np.cos(angle_ring)
        y_ring = r * np.sin(angle_ring)
        z_ring = np.zeros_like(angle_ring)

        ring_traces.append(
            go.Scatter3d(
                x=x_ring,
                y=y_ring,
                z=z_ring,
                mode="lines",
                line=dict(color=f"rgba(0, 220, 255, {opacity})", width=3),
                showlegend=False
            )
        )

    data = [
        # Black hole 1
        go.Scatter3d(
            x=[x1[frame]],
            y=[y1[frame]],
            z=[z1[frame]],
            mode="markers",
            marker=dict(size=bh1_visible_size, color="deepskyblue"),
            name=f"Black hole 1 ({m1} M☉)",
            showlegend=True
        ),

        # Black hole 2
        go.Scatter3d(
            x=[x2[frame]],
            y=[y2[frame]],
            z=[z2[frame]],
            mode="markers",
            marker=dict(size=bh2_visible_size, color="orange"),
            name=f"Black hole 2 ({m2} M☉)",
            showlegend=True
        ),

        # Merged black hole
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(size=merged_size, color="white"),
            name="Merged black hole",
            showlegend=True
        ),

        # Trail 1
        go.Scatter3d(
            x=x1[start:frame + 1],
            y=y1[start:frame + 1],
            z=z1[start:frame + 1],
            mode="lines",
            line=dict(color="deepskyblue", width=4),
            name="BH1 trail",
            showlegend=False
        ),

        # Trail 2
        go.Scatter3d(
            x=x2[start:frame + 1],
            y=y2[start:frame + 1],
            z=z2[start:frame + 1],
            mode="lines",
            line=dict(color="orange", width=4),
            name="BH2 trail",
            showlegend=False
        ),

        # Waveform line
        go.Scatter(
            x=video_t[:frame + 1],
            y=chirp_video[:frame + 1],
            mode="lines",
            line=dict(color="lime", width=2),
            xaxis="x2",
            yaxis="y2",
            name="Chirp waveform",
            showlegend=False
        ),

        # Current time marker on waveform
        go.Scatter(
            x=[current_time, current_time],
            y=[-1.2, 1.2],
            mode="lines",
            line=dict(color="white", dash="dash"),
            xaxis="x2",
            yaxis="y2",
            showlegend=False
        ),
    ]

    data.extend(ring_traces)

    plotly_frames.append(
        go.Frame(
            data=data,
            name=str(frame),
            layout=go.Layout(
                annotations=[
                    dict(
                        text=f"<b>{stage}</b>",
                        x=0.02,
                        y=0.96,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(color="white", size=18)
                    )
                ]
            )
        )
    )


# ============================================================
# INITIAL FIGURE
# ============================================================

fig = go.Figure(
    data=plotly_frames[0].data,
    frames=plotly_frames
)

fig.update_layout(
    title=dict(
        text="Interactive 3D Binary Black Hole Inspiral and Gravitational Wave Chirp",
        font=dict(color="white", size=20),
        x=0.5
    ),

    paper_bgcolor="black",
    plot_bgcolor="black",

    height=850,

    scene=dict(
        domain=dict(x=[0.0, 1.0], y=[0.35, 1.0]),
        bgcolor="black",
        xaxis=dict(
            range=[-6, 6],
            color="white",
            gridcolor="rgba(255,255,255,0.15)",
            zerolinecolor="rgba(255,255,255,0.25)"
        ),
        yaxis=dict(
            range=[-6, 6],
            color="white",
            gridcolor="rgba(255,255,255,0.15)",
            zerolinecolor="rgba(255,255,255,0.25)"
        ),
        zaxis=dict(
            range=[-3, 3],
            color="white",
            gridcolor="rgba(255,255,255,0.15)",
            zerolinecolor="rgba(255,255,255,0.25)"
        ),
        aspectmode="cube",
        camera=dict(
            eye=dict(x=1.5, y=1.8, z=1.2)
        )
    ),

    xaxis2=dict(
        domain=[0.08, 0.95],
        anchor="y2",
        range=[0, animation_duration],
        title=dict(text="Time [s]", font=dict(color="white")),
        color="white",
        gridcolor="rgba(255,255,255,0.15)"
    ),

    yaxis2=dict(
        domain=[0.05, 0.28],
        anchor="x2",
        range=[-1.25, 1.25],
        title=dict(text="Normalised strain", font=dict(color="white")),
        color="white",
        gridcolor="rgba(255,255,255,0.15)"
    ),

    legend=dict(
        font=dict(color="white"),
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="white",
        borderwidth=1
    ),

    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            x=0.1,
            y=0.32,
            xanchor="right",
            yanchor="top",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=40, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate"
                        )
                    ]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate"
                        )
                    ]
                )
            ]
        )
    ],

    sliders=[
        dict(
            active=0,
            x=0.15,
            y=0.32,
            len=0.75,
            currentvalue=dict(
                prefix="Frame: ",
                font=dict(color="white")
            ),
            steps=[
                dict(
                    method="animate",
                    args=[
                        [str(k)],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0)
                        )
                    ],
                    label=str(k)
                )
                for k in range(frames)
            ]
        )
    ]
)

# Save interactive HTML
fig.write_html("black_hole_plotly_simulation.html")

print("Saved interactive simulation as black_hole_plotly_simulation.html")

fig.show()
