import numpy as np
from typing import Optional

def compute_sigwave_height_frequency_band(spectra, frequency, f_low, f_high):
    # Find indices between the frequency limits
    Hs = np.empty(spectra.shape[1])
    for n in range(spectra.shape[1]):
        inds = np.where((frequency >= f_low) & (frequency <= f_high))[0] 
        Hs[n] = 4 * np.sqrt(np.trapz(spectra[inds, n], frequency[inds]))
    return Hs

def energy_weighted_direction(
    energy_density: np.ndarray,
    frequency: np.ndarray,
    a1: np.ndarray,
    b1: np.ndarray,
    min_frequency: Optional[float] = None,
    max_frequency: Optional[float] = None,
):
    a1_weighted = energy_weighted_mean(a1,
                                        energy_density,
                                        frequency,
                                        min_frequency=min_frequency,
                                        max_frequency=max_frequency)
    b1_weighted = energy_weighted_mean(b1,
                                        energy_density,
                                        frequency,
                                        min_frequency=min_frequency,
                                        max_frequency=max_frequency)
    mean_direction = (direction(a1_weighted, b1_weighted))

    return mean_direction



def energy_weighted_mean(
    X: np.ndarray,
    energy_density: np.ndarray,
    frequency: np.ndarray,
    min_frequency: Optional[float] = None,
    max_frequency: Optional[float] = None,
) -> np.ndarray:
    """TODO:"""
    if min_frequency is None:
        min_frequency = frequency.min()

    if max_frequency is None:
        max_frequency = frequency.max()

    # Mask frequencies outside of the specified range.  Must be 0 and not NaN.
    frequency_mask = np.logical_and(frequency >= min_frequency,
                                    frequency <= max_frequency)
    frequency = frequency[frequency_mask]
    energy_density = energy_density[..., frequency_mask]
    X = X[..., frequency_mask]

    # Compute energy-weighted mean.
    m0 = spectral_moment(energy_density, frequency, n=0)
    weighted_integral = np.trapz(y=energy_density * X, x=frequency)
    return weighted_integral / m0


def spectral_moment(
    energy_density: np.ndarray,
    frequency: np.ndarray,
    n: float = 0,
    min_frequency: Optional[float] = None,
    max_frequency: Optional[float] = None,
) -> np.ndarray:
    """
    Function to compute 'nth' spectral moment

    Note:
        This function requires that the frequency dimension is along the last
        axis of `energy_density`.  If energy is empty or null, NaN is returned.

    Args:
        energy_density (np.ndarray): 1-D energy density frequency spectrum with
            shape (f,) or (..., f).
        frequency (np.ndarray): 1-D frequencies with shape (f,).
        n (float): Moment order (e.g., `n=1` is returns the first moment).
        min_frequency (float, optional): lower frequency bound.
        max_frequency (float, optional): upper frequency bound.

    Output:
        Union[float, np.ndarray]: nth spectral moment

    Example:

    Compute 4th spectral moment:
        m4 = spectral_moment(energy, frequency, n=4)
    """
    # Assign default min and max frequencies.
    if min_frequency is None:
        min_frequency = frequency.min()

    if max_frequency is None:
        max_frequency = frequency.max()

    # Mask frequencies outside of the specified range. Must be 0 and not
    # NaN to avoid null output from trapz.
    frequency_mask = np.logical_and(frequency >= min_frequency,
                                    frequency <= max_frequency)
    frequency = frequency[frequency_mask]
    energy_density = energy_density[..., frequency_mask]
    # frequency = np.where(frequency_mask, frequency, 0)
    # energy_density = np.where(frequency_mask, energy_density, 0)

    # Compute nth spectral moment.
    fn = frequency ** n
    mn = np.trapz(energy_density * fn, x=frequency, axis=-1)
    return mn


def direction(a1: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """ Return the frequency-dependent direction from the directional moments.

    Calculate the direction at each frequency from the first two Fourier
    coefficients of the directional spectrum (see Sofar and Kuik et al.).

    References:
        Sofar (n.d.) Spotter Technical Reference Manual

        A J Kuik, G P Van Vledder, and L H Holthuijsen (1988) "A method for the
        routine analysis of pitch-and-roll buoy wave data" JPO, 18(7), 1020-
        1034, 1988.

    Args:
        a1 (np.ndarray): Normalized spectral directional moment (+E).
        b1 (np.ndarray): Normalized spectral directional moment (+N).

    Returns:
        np.ndarray: Direction at each spectral frequency in the metereological
            convention (degrees clockwise from North).
    """
    return (270 - np.rad2deg(np.arctan2(b1, a1))) % 360

