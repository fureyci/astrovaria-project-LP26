"""
Script that reads in model lines and prints out
parameters of different models in the grid that
are needed as inputs to FASTWIND.

Also contains functions that calculate mass
loss rates for different formulae

author: @Ciarán Furey
"""

import numpy as np
import matplotlib.pyplot as plt

Z_LP26 = 0.03

def vink_mdot(stellar_params, metallicity, out="log"):
    """Equation 24 of Vink et al for log(Mdot/[Msun/yr]).
    calculated for different stars for a given metallicity Z.

    Parameters
    ----------
    stellar_params : np.ndarray
        Teff, logg (not used), logL, R (not used), M

    """
    # Lines are:
    # Constant
    # L (already in log(Lsun))
    # M,
    # vinf/vesc, assuming 2.6 for O type,
    # T,
    # T**2
    # Z
    log_Mdot = -6.697 \
         + 2.194 * (stellar_params[:,2] - 5) \
         - 1.313 * np.log10(stellar_params[:,4] / 30.0) \
         - 1.226 * np.log10(2.6/2.0) \
         + 0.933 * np.log10(stellar_params[:,0]/ 4e4) \
         - 10.92 * (np.log10(stellar_params[:,0]/ 4e4))**2 \
         + 0.85 * np.log10(metallicity)
    if out=="log":
        mdot_out = log_Mdot
    elif out=="linear":
        mdot_out = 10**log_Mdot # [Msun/yr]
    return mdot_out

def bjorklund_mdot(stellar_params, metallicity, out="log"):
    """Equation 20 of Bjorklund et al for log(Mdot/[Msun/yr]).
    calculated for different stars for a given metallicity Z.

    Parameters
    ----------
    stellar_params : np.ndarray
        Teff, logg (not used), logL, R (not used), M

    """
    # Lines are:
    # Constant
    # L (already in log(Lsun))
    # M,
    # vinf/vesc, assuming 2.6 for O type,
    # T,
    # T**2
    # Z
    log_Mdot = -5.55 \
         + 0.79*np.log10(metallicity) \
         + (2.16 - 0.32*np.log10(metallicity))*(stellar_params[:,2] - 6)
    if out=="log":
        mdot_out = log_Mdot
    elif out=="linear":
        mdot_out = 10**log_Mdot # [Msun/yr]
    return mdot_out


def krticka_mdot(stellar_params, metallicity, out="log"):
    """Equation 20 of Krticka, J., & Kubat, J. 2017
    for log(Mdot/[Msun/yr]).
    calculated for different stars for a given metallicity Z.

    Parameters
    ----------
    stellar_params : np.ndarray
        Teff, logg (not used), logL, R (not used), M

    """
    # Lines are:
    # Constant
    # L (already in log(Lsun))
    # M,
    # vinf/vesc, assuming 2.6 for O type,
    # T,
    # T**2
    # Z
    log_Mdot = -5.70 \
         + 0.50*np.log10(metallicity) \
         + (1.61 - 0.12*np.log10(metallicity))*(stellar_params[:,2] - 6)
    if out=="log":
        mdot_out = log_Mdot
    elif out=="linear":
        mdot_out = 10**log_Mdot # [Msun/yr]
    return mdot_out

def vinf(stellar_params, gamma=0):
    """Based off lamers calculateion (need reference to paper)

    Parameters
    ----------
    stellar_params : array like
        Teff (not used), logg (not used), logL(not used), R, M

    """
    G = 6.674e-11 # [m3/kg/s2]
    M = stellar_params[:,4] * 1.989e30 # [kg]
    R = stellar_params[:,3] * 6.957e8  # [m]
    vinf_SI = 2.6 * np.sqrt((2 * G * M * (1-gamma))/(R))
    vinf_kms = vinf_SI * 1e-3

    return vinf_kms

def t_r_logg_grid(temps, mag, zero_point, wl, log_g, band_name):
    """Get radii from photometry, log(g), and grid of temperatures."""

    # Convert wavelength to cm.
    wl_cm = wl*1e-8 # 1cm / 1e8 angstrom.

    # Distance to Leo P.
    distance_cm = 1.62 * 3.086e24  # 3.086e24 cm / 1 Mpc

    # Flux at this wavelength from magnitude, using magnitude
    # formula in Vega system.
    # Multiply by 1e8 to convert units from
    # erg/s/cm2/Ang -> erg/s/cm2/cm (since 1e8 A/cm).
    f_lambda = zero_point * (10**(-0.4*mag)) * 1e8 # erg/s/cm2/cm

    # Constrain with Rayleigh jeans approx of blackbody.
    # F = F0 * 10 ^{-0.4m} = (R/d)^2 * pi * B_nu(T_eff)
    radius_cm = np.sqrt((wl_cm**4  * distance_cm**2 * f_lambda)/(2*np.pi*2.998e10*1.381e-16*temps))
    radius_solar = radius_cm / 6.957e10 # 1 Rsun / 6.957e10 cm

    # Bolometric luminosity using this (Teff, R) pair.
    L_cgs = 4*np.pi*(radius_cm**2)*5.67037441e-5*(temps**4)
    logL_solar = np.log10(L_cgs / 3.828e33) # 1Lsun / 3.828e33 erg/s

    g_km = 10**(log_g) * 1e-5 # 1km/s / 1e5cm/s
    # Terminal velocity in terms of g and R, converting R to km.
    vinf_kms = 2.6*np.sqrt(2*g_km*(radius_cm*1e-5))

    print(f"Constraining T and R from magnitude in {band_name} band (= {mag}),\nwhere log(g [cm/s^2]) = {log_g}:")
    print("%-10s %-15s %-15s %-15s" % \
        ("T [K]",  "log(L/Lsun)", "R [Rsun]", "vinf [km/s]"))
    for paramset in zip(temps, logL_solar, radius_solar, vinf_kms):
        print("%-10i %-15.5f %-15.5f %-15.5f" % \
            (paramset[0], paramset[1], paramset[2], paramset[3]))

if __name__ == "__main__":

    temps_in = np.linspace(33e3, 44e3, 8)

    t_r_logg_grid(temps_in, 21.929, 1.1132e-9, 8100.44, 3.42, "F814W")
    t_r_logg_grid(temps_in, 21.929, 1.1132e-9, 8100.44, 4.42, "F814W")
