#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:27:56 2025

@author: aurelien
"""

import numpy as np
"""
chemin_modules = "/Users/aurelien/Desktop/PhD/Codes/Package/Perso/"
sys.path.append(chemin_modules)
import read as read 

"""

# Constantes utiles
c = 299792458  # Vitesse de la lumière (m/s)
G = 6.67430e-11  # Constante gravitationnelle (m^3 kg^-1 s^-2)
au_to__m = 1.49597870700e11  # Unité astronomique en mètres
pc_to__m = 3.085677581491367e16  # Parsec en mètres
arcsec_to__rad = np.pi / (180 * 3600)  # Arcseconde en radians

M_sun_kg = 1.98847e30  # Masse du Soleil (kg)
R_sun_m = 6.957e8  # Rayon solaire en mètres
L_sun_si = 3.828e26  # Luminosité solaire en Watts (unités SI)

clum_cgs= 2.99792458e10 #speed of light in cm/s
h_cgs = 6.62607015e-27 #Planck constant in cgs g.cm2/s
k_cgs = 1.380649e-16   #Boltzmann constant in cgs erg/K
#-------------------------Conversion with Intensity----------------------------

def W_to_erg_s(intensite_si):
    """Convertit l'intensité d'unités SI (W/m^2) en unités CGS (erg/s/cm^2)."""
    return intensite_si * 1e7 * 1e4  # 1 W = 10^7 erg/s, 1 m^2 = 10^4 cm^2

def ergs_to_cm_simple_cgs(energy_ergs):
  wavenumber = energy_ergs / (h_cgs * clum_cgs)
  return wavenumber

#-------------------------Conversion with distances----------------------------

def m_to_UA(distance_m):
    """Convertit une distance en mètres en unités astronomiques (UA)."""
    return distance_m / au_to__m

def m_to_pc(distance_m):
    """Convertit une distance en mètres en parsecs (pc)."""
    return distance_m / pc_to__m

def pc_to_UA(distance_pc):
    """Convertit une distance en parsecs (pc) en unités astronomiques (UA)."""
    return distance_pc * (pc_to__m / au_to__m)

cm_to_A = 1e8

#-------------------------Conversion with angles ------------------------------

def cm_to_arcsec(largeur_cm, distance_pc):
    """Convertit une distance linéaire en centimètres en une mesure angulaire en arcsecondes
    en connaissant la distance à l'étoile en parsecs.
    """
    distance_en_m_etoile = distance_pc * pc_to__m
    angle_rad = largeur_cm*100 / distance_en_m_etoile
    angle_arcsec = np.degrees(angle_rad) * 3600
    return angle_arcsec


def arcsec_to_rad(angle_arcsec):
    """Convertit un angle en arcsecondes en radians."""
    return angle_arcsec * arcsec_to__rad

def rad_to_deg(angle_rad):
    """Convertit un angle en radians en degrés."""
    return np.degrees(angle_rad)



#-------------------------Conversion % Sun ------------------------------------
def Rsun_to_m(rayon_solaire_rs):
    """Convertit un rayon exprimé en rayons solaires (Rs) en mètres (m).
    """
    return rayon_solaire_rs * R_sun_m

def Lsun_to_W(luminosite_solaire_ls):
    """Convertit une luminosité exprimée en luminosités solaires (Ls) en Watts (unités SI).

    """
    return luminosite_solaire_ls * L_sun_si

def Msun_to_kg(masse_solaire):
    """Convertit une masse en masses solaires en kilogrammes."""
    return masse_solaire * M_sun_kg


#------------------------Complex conversion ----------------------------------
def Nh_to_I (Nh,g,nu,Aij):
    '''Converts column density to intensity in cgs'''
    return ( Nh * h_cgs * nu * Aij / ( 4 * np.pi ) )


def I_to_Nh (I,nu,Aij):
    '''Converts intensity to column density'''
    return (( 4 * np.pi * I * nu) / (h_cgs * clum_cgs * Aij ) )

def Nh_to_excitation (Nh,gu):
    return (Nh/gu)



def energy_to_kelvin(energy_value, unit='eV'):
    """
    Convertit une énergie donnée en température en Kelvin.

    Args:
        energy_value (float): La valeur de l'énergie.
        unit (str): L'unité de l'énergie. Peut être 'eV' (électronvolts)
                    ou 'meV' (milli-électronvolts). Par défaut 'eV'.

    Returns:
        float: La température équivalente en Kelvin.
    """
    # Constante de Boltzmann en eV/K
    kB_eV = 8.617333262e-5 # eV/K

    if unit == 'meV':
        # Convertir meV en eV
        energy_value_ev = energy_value / 1000
    elif unit == 'eV':
        energy_value_ev = energy_value
    else:
        raise ValueError("L'unité doit être 'eV' ou 'meV'.")

    temperature_k = energy_value_ev / kB_eV
    return temperature_k

