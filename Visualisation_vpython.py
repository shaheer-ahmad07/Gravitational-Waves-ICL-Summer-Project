from vpython import sphere, vector, color, rate, canvas, curve, ring, label
import numpy as np


# ============================================================
# SCENE SETUP
# ============================================================

scene = canvas(
    title="Binary Black Hole Inspiral and Gravitational Waves",
    width=1000,
    height=700,
    background=color.black
)

scene.camera.pos = vector(0, -12, 7)
scene.camera.axis = vector(0, 12, -7)


# ============================================================
# PARAMETERS
# ============================================================

m1 = 36
m2 = 29
M = m1 + m2

r1_factor = m2 / M
r2_factor = m1 / M

duration = 12
fps = 60
dt = 1 / fps

merger_time = 8.5

r_initial = 5.0
r_final = 0.45

f_initial = 0.35
f_final = 3.0


# ============================================================
# OBJECTS
# ============================================================

bh1 = sphere(
    pos=vector(0, 0, 0),
    radius=0.35,
    color=color.blue,
    emissive=True,
    make_trail=True,
    trail_radius=0.025,
    retain=200
)

bh2 = sphere(
    pos=vector(0, 0, 0),
    radius=0.30,
    color=color.red,
    emissive=True,
    make_trail=True,
    trail_radius=0.025,
    retain=200
)

merged_bh = sphere(
    pos=vector(0, 0, 0),
    radius=0.55,
    color=color.white,
    emissive=True,
    visible=False
)

centre = sphere(
    pos=vector(0, 0, 0),
    radius=0.06,
    color=color.yellow,
    emissive=True
)

stage_label = label(
    pos=vector(-5.5, 5.5, 0),
    text="Inspiral",
    height=18,
    color=color.white,
    box=False
)


# ============================================================
# GRAVITATIONAL WAVE RINGS
# ============================================================

wave_rings = []

for i in range(8):
    wave = ring(
        pos=vector(0, 0, 0),
        axis=vector(0, 0, 1),
        radius=0.1,
        thickness=0.015,
        color=color.cyan,
        opacity=0.25
    )
    wave_rings.append(wave)


# ============================================================
# CHIRP CURVE
# ============================================================

waveform = curve(color=color.green, radius=0.015)

wave_origin_x = -5.5
wave_origin_y = -5.5
wave_scale_x = 0.9
wave_scale_y = 0.8


# ============================================================
# SIMULATION LOOP
# ============================================================

theta = 0
time = 0

while time < duration:
    rate(fps)

    if time < merger_time:
        progress = time / merger_time

        # Radius shrinks with time
        radius = r_initial * (1 - progress) + r_final * progress

        # Frequency increases with time
        orbital_frequency = f_initial * (1 - progress) + f_final * progress
        orbital_frequency = orbital_frequency ** 1.4

        theta += 2 * np.pi * orbital_frequency * dt

        x1 = r1_factor * radius * np.cos(theta)
        y1 = r1_factor * radius * np.sin(theta)

        x2 = -r2_factor * radius * np.cos(theta)
        y2 = -r2_factor * radius * np.sin(theta)

        bh1.pos = vector(x1, y1, 0)
        bh2.pos = vector(x2, y2, 0)

        bh1.visible = True
        bh2.visible = True
        merged_bh.visible = False

        stage_label.text = "Inspiral"

        # Chirp signal: GW frequency is roughly twice orbital frequency
        gw_phase = 2 * theta
        amplitude = 1 / radius
        amplitude = min(amplitude / 1.5, 1.2)

        strain = amplitude * np.sin(gw_phase)

    else:
        bh1.visible = False
        bh2.visible = False
        merged_bh.visible = True

        t_ringdown = time - merger_time

        stage_label.text = "Merger / Ringdown"

        # Ringdown: decaying oscillation
        strain = np.exp(-t_ringdown / 1.5) * np.cos(2 * np.pi * 3.0 * t_ringdown)

    # Update gravitational wave rings
    for i, wave in enumerate(wave_rings):
        if time < merger_time:
            r_wave = ((time * 1.8) - i * 0.8) % 7
        else:
            r_wave = (((time - merger_time) * 3.0) - i * 0.6) % 7

        wave.radius = max(r_wave, 0.1)
        wave.opacity = max(0.04, 0.25 * (1 - r_wave / 7))

    # Draw waveform in 3D scene
    waveform.append(
        pos=vector(
            wave_origin_x + wave_scale_x * time,
            wave_origin_y + wave_scale_y * strain,
            0
        )
    )

    time += dt
