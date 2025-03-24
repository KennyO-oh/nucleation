import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import convolve
from matplotlib.widgets import Button

# -------------------------------
# Simulation Parameters
# -------------------------------
nx, ny = 200, 200  # Total grid dimensions
dx = 1.0  # Grid spacing
dt = 0.1  # Time step per frame (real time per frame)
D = 1.0  # Diffusion constant
epsilon = 0.5  # Controls interface sharpness

# -------------------------------
# Physical Parameters for Overlays
# -------------------------------
T0 = -5.0  # Base temperature (°C) when liquid


# -------------------------------
# Define the bottle mask
# -------------------------------
def create_bottle_mask(nx, ny):
    """
    Create a binary mask representing the interior of a bottle.

    The mask is constructed from three parts:
      - A rectangular region.
      - A trapezoidal region representing the curved transition.
      - A square region at the top.
    """
    X, Y = np.meshgrid(np.arange(ny), np.arange(nx))
    mask_rect = (X >= 60) & (X <= 140) & (Y >= 25) & (Y <= 150)
    left_boundary = 60 + ((90 - 60) / (165 - 150)) * (Y - 150)
    right_boundary = 140 + ((110 - 140) / (165 - 150)) * (Y - 150)
    mask_trap = (Y >= 150) & (Y <= 165) & (X >= left_boundary) & (X <= right_boundary)
    mask_square = (X >= 90) & (X <= 110) & (Y >= 165) & (Y <= 180)
    mask = mask_rect | mask_trap | mask_square
    return mask


mask = create_bottle_mask(nx, ny)
total_pixels = np.sum(mask)

# -------------------------------
# Initialize the phase field
# -------------------------------
phi = -np.ones((nx, ny))  # Start with liquid everywhere (φ = -1)

# Global variables for simulation state and timer.
nucleation_occurred = False
nucleation_time = None  # Time when nucleation first occurs.
simulation_time = 0.0
simulation_finished = False  # Flag to stop updates.
prev_frozen_fraction = 0.0  # For computing freeze rate.

# -------------------------------
# Laplacian kernel (finite-difference)
# -------------------------------
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]]) / (dx * dx)


def update_field(phi):
    """
    Update the phase field (φ) using a finite-difference approximation of diffusion and a double-well potential.

    The evolution is given by:
      φ_new = φ + dt * [D * ∇²φ - (1/ε²) * (φ - φ³)]

    The field is clipped to remain within [-1.1, 1.1] and cells outside the bottle mask are forced to liquid (φ = -1).
    """
    lap = convolve(phi, laplacian_kernel, mode='constant', cval=-1.0)
    dphi = dt * (D * lap - (1.0 / epsilon ** 2) * (phi - phi ** 3))
    phi += dphi
    phi = np.clip(phi, -1.1, 1.1)
    # Force cells outside the mask to remain liquid.
    phi[~mask] = -1.0
    return phi


# -------------------------------
# Setting up the figure and display
# -------------------------------
fig, ax = plt.subplots()
ax.set_facecolor('black')
cmap = plt.cm.coolwarm
im = ax.imshow(np.ma.array(phi, mask=~mask), cmap=cmap, vmin=-1, vmax=1,
               origin='lower', extent=[0, ny, 0, nx])
ax.set_title('Supercooled Water in a Bottle', fontweight='bold', fontsize=12, color='black')
ax.set_xticks([])
ax.set_yticks([])

# -------------------------------
# Setting persistent text overlays
# -------------------------------
text_freeze = ax.text(5, 185, "", color='white', fontsize=12,
                      bbox=dict(facecolor='gray', alpha=0.6), animated=True)
text_temp = ax.text(135, 185, "", color='white', fontsize=12,
                    bbox=dict(facecolor='gray', alpha=0.6), animated=True)
text_freeze_rate = ax.text(5, 10, "", color='white', fontsize=12,
                           bbox=dict(facecolor='gray', alpha=0.6), animated=True)
text_time = ax.text(140, 10, "", color='white', fontsize=12,
                    bbox=dict(facecolor='gray', alpha=0.6), animated=True)
text_instruction = ax.text(100, 100, "Click to Nucleate Ice!", color='white', fontsize=14,
                           ha='center', bbox=dict(facecolor='black', alpha=0.8), animated=True)


# -------------------------------
# Setting nucleation event on mouse click
# -------------------------------
def on_click(event):
    """
    Handle a mouse click event to nucleate ice.

    If the click occurs within the bottle mask, a circular region (radius 3 grid units)
    around the click location is set to ice (φ = 1.0). The nucleation time is recorded
    if this is the first nucleation event.
    """
    global phi, nucleation_occurred, nucleation_time, simulation_time
    if event.inaxes != ax:
        return
    j = int(event.xdata)
    i = int(event.ydata)
    if not mask[i, j]:
        return
    r = 3  # Nucleation radius (grid units)
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di ** 2 + dj ** 2 <= r ** 2:
                if 0 <= i + di < nx and 0 <= j + dj < ny:
                    if mask[i + di, j + dj]:
                        phi[i + di, j + dj] = 1.0
    if not nucleation_occurred:
        nucleation_occurred = True
        nucleation_time = simulation_time


fig.canvas.mpl_connect('button_press_event', on_click)


# -------------------------------
# Animation function
# -------------------------------
def animate(frame):
    """
    Update the simulation for each animation frame.

    This function advances the simulation (if nucleation has occurred and the simulation
    is not finished), updates the phase field, computes the frozen fraction and effective temperature,
    updates the persistent overlays, and stops the simulation if the frozen fraction reaches 100%.
    """
    global phi, simulation_time, prev_frozen_fraction, simulation_finished, ani

    # Advance simulation time by dt if nucleation has occurred and simulation is still running.
    if nucleation_occurred and not simulation_finished:
        simulation_time += dt

    # Update simulation twice per frame for faster dynamics.
    if not simulation_finished:
        for _ in range(2):
            phi = update_field(phi)

    # Compute the average phase value in the bottle and convert to a frozen percentage.
    avg_phi = np.mean(phi[mask])
    frozen_fraction = (avg_phi + 1) * 100

    if frozen_fraction >= 99.1:
        frozen_fraction = 100.0
        simulation_finished = True
        ani.event_source.stop()

    freeze_rate = (frozen_fraction - prev_frozen_fraction) / dt
    prev_frozen_fraction = frozen_fraction
    if frozen_fraction >= 99.1:
        freeze_rate = 0.0

    # Compute the effective temperature based on the frozen fraction.
    T_eff = T0 + (frozen_fraction / 100) * (-T0)
    if T_eff > 0:
        T_eff = 0

    # Update the displayed phase field.
    im.set_array(np.ma.array(phi, mask=~mask))

    display_fraction = min(frozen_fraction, 100.0)
    text_freeze.set_text(f'Frozen: {display_fraction:.1f}%')
    text_temp.set_text(f'Temp: {T_eff:.1f}°C')
    text_freeze_rate.set_text(f'Freeze Rate: {freeze_rate:.1f}%/s')
    if nucleation_occurred:
        elapsed_time = simulation_time - nucleation_time
        text_time.set_text(f'Time: {elapsed_time:.1f}s')
    else:
        text_time.set_text('Time: -')

    if nucleation_occurred:
        text_instruction.set_text('')
    else:
        text_instruction.set_text('Click to Nucleate Ice!')

    return [im, text_freeze, text_temp, text_freeze_rate, text_time, text_instruction]


ani = animation.FuncAnimation(fig, animate, frames=2000, interval=50, blit=True)

# -------------------------------
# Reset button
# -------------------------------
ax_button = plt.axes([0.4, 0.02, 0.2, 0.06])
button = Button(ax_button, "Reset", color='gray', hovercolor='lightgray')


def reset(event):
    """
    Reset the simulation to its initial state.

    This function resets the phase field, global simulation variables,
    and updates the displayed text overlays, then restarts the animation.
    """
    global phi, nucleation_occurred, nucleation_time, simulation_time, simulation_finished, prev_frozen_fraction, ani
    phi = -np.ones((nx, ny))
    nucleation_occurred = False
    nucleation_time = None
    simulation_time = 0.0
    simulation_finished = False
    prev_frozen_fraction = 0.0
    im.set_array(np.ma.array(phi, mask=~mask))
    text_freeze.set_text('Frozen: 0.0%')
    text_temp.set_text(f'Temp: {T0:.1f}°C')
    text_freeze_rate.set_text('Freeze Rate: 0.0%/s')
    text_time.set_text('Time: -')
    text_instruction.set_text('Click to Nucleate Ice!')
    ani.event_source.start()


button.on_clicked(reset)

plt.show()