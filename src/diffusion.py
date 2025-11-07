import math
import numpy as np

k_B = 8.617e-5  # eV/K

def D_cm2_s(D0, Ea_eV, T_C):
    T_K = T_C + 273.15
    return D0 * math.exp(-Ea_eV/(k_B*T_K))

def const_source_erfc(Cs, D, t_s, x_um):
    x_cm = x_um * 1e-4
    z = x_cm / (2*math.sqrt(D*t_s))
    # numpy.erf handles arrays nicely
    from scipy.special import erf
    return Cs * 0.5 * (1 - erf(z))

def limited_source_gaussian(Q, D, t_s, x_um):
    x_cm = x_um * 1e-4
    denom = math.sqrt(math.pi*D*t_s)
    return (Q/denom) * np.exp(-(x_cm**2)/(4*D*t_s))

def implant_gaussian(dose_cm2, Rp_um, dR_um, x_um):
    denom = math.sqrt(2*math.pi) * dR_um
    return (dose_cm2/denom) * np.exp(-0.5*((x_um - Rp_um)/dR_um)**2) * 1e4  # → /cm^3

def anneal_broaden(Cx, D, t_s, x_um):
    # Convolve with Gaussian kernel of sigma = sqrt(2 D t). FFT approach.
    from numpy.fft import fft, ifft
    x_cm = x_um * 1e-4
    dx = x_cm[1]-x_cm[0]
    sigma = math.sqrt(2*D*t_s)
    # Build kernel in real space (wrap-around ok for long arrays)
    L = len(x_um)
    kx = (np.arange(L) - L//2) * dx
    kern = np.exp(-0.5*(kx/sigma)**2)
    kern /= (kern.sum()*dx)
    return np.real(ifft(fft(Cx)*fft(kern)))

def junction_depth(x_um, NA, ND):
    # First depth where ND <= NA (simple p–n cross)
    idx = np.where(ND <= NA)[0]
    return float(x_um[idx[0]]) if len(idx) else float("nan")
