# Code for creating the initial conditions
import numpy as np

def circle(x, y, x0, y0, radius):
    """
    Create a phase field model where the central circular region is 1 and the rest of the domain is 0.
    
    Parameters:
    x, y: 2D arrays of x and y coordinates.
    x0, y0: Center of the circular region.
    radius: Radius of the central region.
    
    Returns:
    phase_field: 2D array where the region inside the circle is 1, and outside is 0.
    """
    x = np.array(x)
    y = np.array(y)
    
    # Circular region equation
    circle = (x - x0)**2 + (y - y0)**2
    
    # Create the phase field: inside the circle is 1, outside is 0
    phase_field = np.zeros_like(circle)
    phase_field[circle <= radius**2] = 1
    
    return phase_field

def circle_with_noise(x, y, x0, y0, radius):
    """
    Create a phase field model where the central circular region is 1 with added noise and the rest of the domain is 0.
    
    Parameters:
    x, y: 2D arrays of x and y coordinates.
    x0, y0: Center of the circular region.
    radius: Radius of the central region.
    noise_amplitude: The amplitude of the noise to be added in the region where the value is 1.
    
    Returns:
    phase_field: 2D array where the region inside the circle is 1 with noise, and outside is 0.
    """
    noise_amplitude=0.1 #0.5
    x = np.array(x)
    y = np.array(y)
    
    # Circular region equation
    circle = (x - x0)**2 + (y - y0)**2
    
    # Create the phase field: inside the circle is 1, outside is 0
    phase_field = np.zeros_like(circle)
    phase_field[circle <= radius**2] = 1.0
    
    # Add random noise to the region where the phase field is 1, but keep values between 0 and 1
    noise = noise_amplitude * np.random.randn(*phase_field.shape)
    phase_field[circle <= radius**2] += noise[circle <= radius**2]
    
    # Clip the values to keep them within the range [0, 1]
    phase_field = np.clip(phase_field, 0, 1)
    
    return phase_field

# Define the function with a wavy boundary
def circle_with_wavy_boundary(x, y, x0, y0, radius, wave_amplitude=0.2, wave_frequency=5):
    """
    Create a phase field model where the central circular region has a wavy boundary with noise,
    and the rest of the domain is 0.
    
    Parameters:
    x, y: 2D arrays of x and y coordinates.
    x0, y0: Center of the circular region.
    radius: Average radius of the central region.
    wave_amplitude: Amplitude of the waviness on the boundary.
    wave_frequency: Frequency of the waviness on the boundary.
    
    Returns:
    phase_field: 2D array where the region inside the wavy circle is 1 with noise, and outside is 0.
    """
    noise_amplitude = 0.1  # Noise amplitude
    x = np.array(x)
    y = np.array(y)

    # Calculate the angle at each (x, y) point relative to the center
    angles = np.arctan2(y - y0, x - x0)

    # Calculate the wavy radius as a function of the angle
    wavy_radius = radius + wave_amplitude * np.sin(wave_frequency * angles)
    
    # Circular region with wavy boundary equation
    circle = (x - x0)**2 + (y - y0)**2
    
    # Create the phase field: inside the wavy circle is 1, outside is 0
    phase_field = np.zeros_like(circle)
    phase_field[circle <= wavy_radius**2] = 1
    
    # Add random noise to the region where the phase field is 1
    noise = noise_amplitude * np.random.randn(*phase_field.shape)
    phase_field[circle <= wavy_radius**2] += noise[circle <= wavy_radius**2]
    
    # Clip the values to keep them within the range [0, 1]
    phase_field = np.clip(phase_field, 0, 1)
    
    return phase_field

def initialize_fields(x, y, Lx, Ly, Nx, Ny, phi, ep, ep_prime, sig, r, theta, l_f, l_phi, lr_phi, l1_sig, l2_sig, lr_sig, k2):
    # Ellipsoid parameters
    x0, y0 = 0, 0  # Center of the ellipsoid
    x1, y1 = 0.0, (Ly - 0.5)
    a1, b1 = 0.25, 0.15  # Semi-axes
    a2, b2 = 0.2, 0.3  # Semi-axes
    interface_width = 0.1  # Interface width
    radius = 3.0
    vartheta = 25.0

    # Compute the phase field
    sig['g'] = circle_with_noise(x, y, x0, y0, radius) 
    r['g'] = - np.arctan((k2 * sig['g']))
    ep_prime['g'] = -np.sin(theta['g'])
    ep['g'] = (vartheta) + (np.cos(theta['g']))