import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import time

# -------------------------------
# Simulation Parameters
# -------------------------------
nx, ny = 200, 200       # Total grid dimensions
dx = 1.0                # Grid spacing
dt = 0.1                # Time step per frame
D = 1.0                 # Diffusion constant
epsilon = 0.5           # Controls interface sharpness

# -------------------------------
# Physical Parameters for Overlays
# -------------------------------
T0 = -5.0  # Base temperature (°C) when liquid

# -------------------------------
# Define the bottle mask
# -------------------------------
def create_bottle_mask(nx, ny):
    X, Y = np.meshgrid(np.arange(ny), np.arange(nx))
    mask_rect = (X >= 60) & (X <= 140) & (Y >= 25) & (Y <= 150)
    left_boundary = 60 + ((90 - 60) / (165 - 150)) * (Y - 150)
    right_boundary = 140 + ((110 - 140) / (165 - 150)) * (Y - 150)
    mask_trap = (Y >= 150) & (Y <= 165) & (X >= left_boundary) & (X <= right_boundary)
    mask_square = (X >= 90) & (X <= 110) & (Y >= 165) & (Y <= 180)
    return mask_rect | mask_trap | mask_square

mask = create_bottle_mask(nx, ny)
phi0 = -np.ones((nx, ny))  # Liquid everywhere (φ = -1)

# -------------------------------
# Laplacian kernel (finite-difference)
# -------------------------------
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]]) / (dx * dx)

def update_field(phi):
    lap = convolve(phi, laplacian_kernel, mode='constant', cval=-1.0)
    dphi = dt * (D * lap - (1.0/epsilon**2) * (phi - phi**3))
    phi += dphi
    phi = np.clip(phi, -1.1, 1.1)
    phi[~mask] = -1.0
    return phi

# -------------------------------
# Initialize simulation state in session_state
# -------------------------------
if 'phi' not in st.session_state:
    st.session_state.phi = phi0.copy()
if 'nucleation_occurred' not in st.session_state:
    st.session_state.nucleation_occurred = False
if 'simulation_time' not in st.session_state:
    st.session_state.simulation_time = 0.0
if 'simulation_finished' not in st.session_state:
    st.session_state.simulation_finished = False
if 'prev_frozen_fraction' not in st.session_state:
    st.session_state.prev_frozen_fraction = 0.0
if 'nucleation_time' not in st.session_state:
    st.session_state.nucleation_time = None

# -------------------------------
# Define functions to interact with simulation
# -------------------------------
def nucleate():
    """Simulate nucleation at the center of the bottle."""
    i, j = nx // 2, ny // 2  # Use the center for simplicity
    r = 3  # nucleation radius in grid units
    for di in range(-r, r+1):
        for dj in range(-r, r+1):
            if di**2 + dj**2 <= r**2:
                if 0 <= i+di < nx and 0 <= j+dj < ny:
                    if mask[i+di, j+dj]:
                        st.session_state.phi[i+di, j+dj] = 1.0
    if not st.session_state.nucleation_occurred:
        st.session_state.nucleation_occurred = True
        st.session_state.nucleation_time = st.session_state.simulation_time

def reset_simulation():
    st.session_state.phi = phi0.copy()
    st.session_state.nucleation_occurred = False
    st.session_state.nucleation_time = None
    st.session_state.simulation_time = 0.0
    st.session_state.simulation_finished = False
    st.session_state.prev_frozen_fraction = 0.0

# -------------------------------
# Streamlit UI Controls
# -------------------------------
st.title("Supercooled Water in a Bottle Simulation")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Nucleate Ice"):
        nucleate()
with col2:
    if st.button("Reset"):
        reset_simulation()
with col3:
    run_simulation = st.button("Run Simulation")

# Create a placeholder for the plot
placeholder = st.empty()

# -------------------------------
# Set up the matplotlib figure
# -------------------------------
fig, ax = plt.subplots(figsize=(6,6))
ax.set_facecolor('black')
cmap = plt.cm.coolwarm
im = ax.imshow(np.ma.array(st.session_state.phi, mask=~mask), cmap=cmap, vmin=-1, vmax=1,
               origin='lower', extent=[0, ny, 0, nx])
ax.set_title('Supercooled Water in a Bottle', fontsize=12, color='black')
ax.set_xticks([])
ax.set_yticks([])

# -------------------------------
# Simulation Loop
# -------------------------------
if run_simulation:
    for frame in range(200):  # Run for 200 frames (adjust as needed)
        if st.session_state.nucleation_occurred and not st.session_state.simulation_finished:
            st.session_state.simulation_time += dt
        if not st.session_state.simulation_finished:
            for _ in range(2):  # Update simulation twice per frame for speed
                st.session_state.phi = update_field(st.session_state.phi)
        avg_phi = np.mean(st.session_state.phi[mask])
        frozen_fraction = (avg_phi + 1) * 100
        if frozen_fraction >= 99.1:
            frozen_fraction = 100.0
            st.session_state.simulation_finished = True
        freeze_rate = (frozen_fraction - st.session_state.prev_frozen_fraction) / dt
        st.session_state.prev_frozen_fraction = frozen_fraction
        if frozen_fraction >= 99.1:
            freeze_rate = 0.0
        T_eff = T0 + (frozen_fraction/100)*(-T0)
        if T_eff > 0:
            T_eff = 0

        # Update the plot
        im.set_array(np.ma.array(st.session_state.phi, mask=~mask))
        # Instead of setting ax.texts = [], remove texts individually:
        for txt in ax.texts:
            txt.remove()
        ax.text(5, 185, f'Frozen: {frozen_fraction:.1f}%', color='white', fontsize=12,
                bbox=dict(facecolor='gray', alpha=0.6))
        ax.text(135, 185, f'Temp: {T_eff:.1f}°C', color='white', fontsize=12,
                bbox=dict(facecolor='gray', alpha=0.6))
        ax.text(5, 10, f'Freeze Rate: {freeze_rate:.1f}%/s', color='white', fontsize=12,
                bbox=dict(facecolor='gray', alpha=0.6))
        if st.session_state.nucleation_occurred:
            elapsed_time = st.session_state.simulation_time - st.session_state.nucleation_time
            ax.text(140, 10, f'Time: {elapsed_time:.1f}s', color='white', fontsize=12,
                    bbox=dict(facecolor='gray', alpha=0.6))
        else:
            ax.text(140, 10, 'Time: -', color='white', fontsize=12,
                    bbox=dict(facecolor='gray', alpha=0.6))
        if not st.session_state.nucleation_occurred:
            ax.text(100, 100, 'Click "Nucleate Ice" to start!', color='white', fontsize=14,
                    ha='center', bbox=dict(facecolor='black', alpha=0.8))
        placeholder.pyplot(fig)
        time.sleep(0.05)  # Control frame rate
