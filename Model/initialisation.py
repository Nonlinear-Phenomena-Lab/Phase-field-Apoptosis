import numpy as np

def circle(x, y, x0, y0, radius):
    x = np.array(x)
    y = np.array(y)
    
    circle = (x - x0)**2 + (y - y0)**2
    
    phase_field = np.zeros_like(circle)
    phase_field[circle <= radius**2] = 1
    
    return phase_field

def circle_with_noise(x, y, x0, y0, radius):
    noise_amplitude=0.1 
    x = np.array(x)
    y = np.array(y)
    
    circle = (x - x0)**2 + (y - y0)**2
    
    phase_field = np.zeros_like(circle)
    phase_field[circle <= radius**2] = 1
    
    noise = noise_amplitude * np.random.randn(*phase_field.shape)
    phase_field[circle <= radius**2] += noise[circle <= radius**2]
    
    phase_field = np.clip(phase_field, 0, 1)
    
    return phase_field

def circle_with_wavy_boundary(x, y, x0, y0, radius, wave_amplitude=0.2, wave_frequency=5):
    noise_amplitude = 0.1 
    x = np.array(x)
    y = np.array(y)

    angles = np.arctan2(y - y0, x - x0)

    wavy_radius = radius + wave_amplitude * np.sin(wave_frequency * angles)

    circle = (x - x0)**2 + (y - y0)**2

    phase_field = np.zeros_like(circle)
    phase_field[circle <= wavy_radius**2] = 1

    noise = noise_amplitude * np.random.randn(*phase_field.shape)
    phase_field[circle <= wavy_radius**2] += noise[circle <= wavy_radius**2]

    phase_field = np.clip(phase_field, 0, 1)
    
    return phase_field

def initialize_fields(x, y, Lx, Ly, Nx, Ny, phi, ep, ep_prime, sig, r, theta, beta, ebar, alpha, k1, k2, gamma, tau):
    x0, y0 = 0, 0  
    x1, y1 = 0.0, (Ly - 0.5)
    a1, b1 = 0.25, 0.15  
    a2, b2 = 0.2, 0.3  
    interface_width = 0.1 
    radius = 3.0

    sig['g'] = circle_with_noise(x, y, x0, y0, radius) 
    r['g'] = - (k1/np.pi) * np.arctan((k2 * sig['g']))
    ep_prime['g'] = -ebar*np.sin(theta['g'])
    ep['g'] = (ebar/4e-02) + (ebar * np.cos(theta['g']))