#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:28:53 2025

@author: aurelien
"""

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
import re as re 
import os 


import sys
chemin_modules = "/Users/aurelien/Desktop/PhD/Codes/Package/Perso/"
sys.path.append(chemin_modules)
import read as read 



# --- 1. Définition des constantes physiques ---
c_m_s = 2.99792458e8    # Vitesse de la lumière en m/s
c_cm_s = 2.99792458e10  # Vitesse de la lumière en cm/s
h_J_s = 6.62607015e-34  # Constante de Planck en J.s
k_J_K = 1.380649e-23    # Constante de Boltzmann en J/K

h = 6.62607015e-34  # Constante de Planck en J.s
k = 1.380649e-23    # Constante de Boltzmann en J/K
c = 2.99792458e10 # cm/s
def lambda_to_nu(wavelength_cm):
    return c / wavelength_cm*1e-2

# Fonction de conversion inverse
# nu (Hz) -> lambda (cm) : lambda = c / nu
def nu_to_lambda(frequency_hz):
    return c / frequency_hz

def nu_to_temp(frequency_hz):
    return (h * frequency_hz) / k

def temp_to_nu(temperature_k):
    return (k * temperature_k) / h

def lambda_cm_to_temp_k(wavelength_cm):
    nu_hz = c / wavelength_cm *1e-2 # cm -> Hz
    return (h * nu_hz) / k     # Hz -> K

def temp_k_to_lambda_cm(temperature_k):
    nu_hz = (k * temperature_k) / h # K -> Hz
    return c / nu_hz                # Hz -> cm





# =============================================================================
#                                Pour l'étude du spectre en sortie polyvalent 
# =============================================================================
def parse_spectrum_file(filepath):
    """
    Parse le fichier de spectre d'émission et extrait les intensités des raies.
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            
            # Utiliser une expression régulière pour extraire la valeur et la description de la raie
            match = re.match(r'\s*\d+\s+([\d.Ee-]+)\s+#\s*erg cm-2 s-1 sr-1\s*I\((CO|H2)\s+v=(\d+),J=(\d+)->v=(\d+),J=(\d+).*', line)
            if match:
                intensity_value = float(match.group(1))
                molecule = match.group(2)
                vu = int(match.group(3))
                Ju = int(match.group(4))
                vl = int(match.group(5))
                Jl = int(match.group(6))
                
                data.append({
                    'molecule': molecule,
                    'intensity': intensity_value, # La nouvelle colonne d'intensité
                    'vu': vu,
                    'Ju': Ju,
                    'vl': vl,
                    'Jl': Jl
                })
            else:
                print(f"Ligne non parsée (Orion_Spectrum): {line.strip()}")
    return pd.DataFrame(data)



def parse_line_data_file(filepath, molecule_type):
    """
    Parse les fichiers de données de raies (CO_line.dat, H2_line.dat)
    et extrait les énergies, Aij, et les transitions J.
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            if molecule_type == 'CO':
                match = re.match(r'\s*\d+\s+\d+\s+\d+\s+([\d.]+)\s+([\d.Ee-]+)\s+quant:\s+(\d+);\s+(\d+);\s+(\d+);\s+(\d+);.*(\d+\.\d+)\s+GHz', line)
                match = re.match(r'\s*\d+\s+\d+\s+\d+\s+([\d.]+)\s+([\d.Ee-]+)\s+quant:\s+(\d+);\s+(\d+);\s+(\d+);\s+(\d+);\s+info:\s+([\d.]+)\s+GHz', line)
                if match:
                    E_K = float(match.group(1))
                    Aij = float(match.group(2))
                    vu = int(match.group(3))
                    Ju = int(match.group(4))
                    vl = int(match.group(5))
                    Jl = int(match.group(6))
                    frequency_GHz = float(match.group(7))
                    
                    data.append({
                        'molecule': molecule_type,
                        'vu': vu,
                        'Ju': Ju,
                        'vl': vl,
                        'Jl': Jl,
                        'energy_K': E_K,
                        'Aij_s-1': Aij,
                        'frequency_GHz': frequency_GHz
                    })
                else:
                    print(f"Ligne non parsée (lineCO): {line.strip()}")

            elif molecule_type == 'H2':
                match = re.match(r'\s*\d+\s+\d+\s+\d+\s+([\d.]+)\s+([\d.Ee-]+)\s+quant:\s+(\d+);\s+(\d+);\s+(\d+);\s+(\d+);.*(\d+\.\d+)\s+micrometres', line)
                
                if match:
                    E_K = float(match.group(1))
                    Aij = float(match.group(2))
                    vu = int(match.group(3))
                    Ju = int(match.group(4))
                    vl = int(match.group(5))
                    Jl = int(match.group(6))
                    wavelength_micrometers = float(match.group(7))
                    
                    frequency_GHz = (299792458 / (wavelength_micrometers * 1e-6)) * 1e-9 
                    
                    data.append({
                        'molecule': molecule_type,
                        'vu': vu,
                        'Ju': Ju,
                        'vl': vl,
                        'Jl': Jl,
                        'energy_K': E_K,
                        'Aij_s-1': Aij,
                        'wavelength_nm': wavelength_micrometers * 1000, 
                        'frequency_GHz': frequency_GHz
                    })
                else:
                    print(f"Ligne non parsée (lineH2): {line.strip()}")
    return pd.DataFrame(data)





def parse_format1(filepath, mol_type, source_file_name):
    """
    Parse les fichiers de type lines_h2.dat, lines_co.dat, lines_co_A.dat
    """
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start_data = False
    for line in lines:
        if line.strip() == '#---data------------------------------------------------------------------------------------------':
            start_data = True
            continue
        if not start_data or line.strip() == '':
            continue

        parts = line.strip().split()
        if len(parts) < 11: # S'assurer qu'il y a assez de colonnes
            continue

        try:
            # Extraction des données brutes
            # n, nu, nl (non utilisés directement ici mais présents dans le format)
            e_k_val = float(parts[3]) # E(K)
            aij = float(parts[4])     # Aij(s-1)
            vu = int(parts[5])
            ju = int(parts[6])
            # pu = int(parts[7]) # pu non demandé, mais présent
            vl = int(parts[8])
            jl = int(parts[9])

            # Wavelength est dans la description (ex: 1544.4482952688606 Angstrom)
            description_str = " ".join(parts[10:])
            # Regex pour extraire le nombre en Angstrom
            match = re.search(r'(\d+\.?\d*)\s*Angstrom', description_str)
            if match:
                wavelength_angstrom = float(match.group(1))
                wavelength_m = wavelength_angstrom * 1e-10 # Angstrom -> m
            else:
                print(f"Warning: Wavelength not found in description for line: {line.strip()} in {source_file_name}")
                continue # Skip if wavelength cannot be extracted

            # Calculs
            wavelength_cm = wavelength_m * 100 # m -> cm (pour cohérence avec c_cm_s)
            frequency_hz = c_m_s / wavelength_m
            frequency_ghz = frequency_hz / 1e9
            teq_k = (h_J_s * frequency_hz) / k_J_K

            # E(cm-1) à partir de E(K) = E/k
            # E(cm-1) = E_K * k / (h * c_cm_s)
            e_cm_1 = e_k_val * k_J_K / (h_J_s * c_cm_s)

            data.append({
                'Molecule': mol_type,
                'vu': vu, 'Ju': ju, 'vl': vl, 'Jl': jl,
                'Aij_s-1': aij,
                'E_cm-1': e_cm_1,
                'E_K': e_k_val,
                'Wavelength_m': wavelength_m,
                'Frequency_GHz': frequency_ghz,
                'Teq_K': teq_k,
                'Source_File': source_file_name
            })
        except ValueError as e:
            print(f"Error parsing line in {source_file_name}: {line.strip()} - {e}")
            continue
    return pd.DataFrame(data)

def parse_format2(filepath, mol_type, source_file_name):
    """
    Parse les fichiers de type UV_H2_B.dat
    """
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start_data = False
    for line in lines:
        if line.strip() and not line.strip().startswith('#') and not line.strip().isdigit():
            start_data = True
        if not start_data or line.strip() == '':
            continue
        if line.strip().startswith('#'): # Skip comment lines after first data detected
            continue

        # Utiliser regex pour split sur '!' et ensuite sur les espaces
        match = re.match(r'\s*(\d+)\s*(\d+)\s*([\d\.E\+\-]+)\s*([\d\.E\+\-]+)\s*!\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)', line)
        if match:
            try:
                # levu, levl = int(match.group(1)), int(match.group(2)) # Niveaux d'énergie (non demandés en output)
                aij = float(match.group(3))      # Aij(s-1)
                e_cm_1_val = float(match.group(4)) # EE(cm-1)
                vu = int(match.group(5))
                ju = int(match.group(6))
                vl = int(match.group(7))
                jl = int(match.group(8))

                # Calculs
                # Wavelength_cm = 1 / E_cm-1
                wavelength_cm = 1 / e_cm_1_val
                wavelength_m = wavelength_cm * 1e-2 # cm -> m
                frequency_hz = c_m_s / wavelength_m
                frequency_ghz = frequency_hz / 1e9
                teq_k = (h_J_s * frequency_hz) / k_J_K

                # E_K à partir de E_cm-1
                e_k = e_cm_1_val * h_J_s * c_cm_s / k_J_K

                data.append({
                    'Molecule': mol_type,
                    'vu': vu, 'Ju': ju, 'vl': vl, 'Jl': jl,
                    'Aij_s-1': aij,
                    'E_cm-1': e_cm_1_val,
                    'E_K': e_k,
                    'Wavelength_m': wavelength_m,
                    'Frequency_GHz': frequency_ghz,
                    'Teq_K': teq_k,
                    'Source_File': source_file_name
                })
            except ValueError as e:
                print(f"Error parsing line in {source_file_name}: {line.strip()} - {e}")
                continue
        else:
            # print(f"Warning: Line format not recognized in {source_file_name}: {line.strip()}")
            pass # Many lines might not match, e.g., first line number
    return pd.DataFrame(data)

# NOUVELLE FONCTION: parse_format3 (pour le nouveau lines_h2.dat)
def parse_format3(filepath, mol_type, source_file_name):
    """
    Parse le fichier de type lines_h2.dat avec le nouveau format
    (où la longueur d'onde est en micromètres et les quantums num sont après 'quant:')
    """
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start_data = False
    # Regex pour capturer les champs importants
    # S'adapte à des espaces variables
    line_pattern = re.compile(
        r'^\s*\d+\s+\d+\s+\d+\s+'                     # n, nu, nl (non capturés ici mais parsés)
        r'(\S+)\s+'                                  # E(K)
        r'([\d\.E\+\-]+)\s+'                         # Aij(s-1)
        r'quant:\s*(\d+);\s*(\d+);\s*(\d+);\s*(\d+);\s+' # vu, Ju, vl, Jl
        r'info:\s*([\d.]+)\s*micrometres;.*$'         # Wavelength (micrometres) et le reste
    )

    for line in lines:
        if line.strip() == '#---data------------------------------------------------------------------------------------------------------------------------------------------':
            start_data = True
            continue
        if not start_data or line.strip() == '':
            continue

        match = line_pattern.search(line)
        if not match:
            # print(f"Warning: Line format not matched in {source_file_name}: {line.strip()}")
            continue

        try:
            e_k_val = float(match.group(1))
            aij = float(match.group(2))
            vu = int(match.group(3))
            ju = int(match.group(4))
            vl = int(match.group(5))
            jl = int(match.group(6))
            wavelength_micrometres = float(match.group(7))

            wavelength_m = wavelength_micrometres * 1e-6 # micrometres -> m
            frequency_hz = c_m_s / wavelength_m
            frequency_ghz = frequency_hz / 1e9
            teq_k = (h_J_s * frequency_hz) / k_J_K

            # E(cm-1) à partir de E(K) = E/k
            e_cm_1 = e_k_val * k_J_K / (h_J_s * c_cm_s)

            data.append({
                'Molecule': mol_type,
                'vu': vu, 'Ju': ju, 'vl': vl, 'Jl': jl,
                'Aij_s-1': aij,
                'E_cm-1': e_cm_1,
                'E_K': e_k_val,
                'Wavelength_m': wavelength_m,
                'Frequency_GHz': frequency_ghz,
                'Teq_K': teq_k,
                'Source_File': source_file_name
            })
        except ValueError as e:
            print(f"Error parsing line in {source_file_name}: {line.strip()} - {e}")
            continue
    return pd.DataFrame(data)

def parse_format4(filepath, mol_type, source_file_name):
    """
    Parse le fichier de type line_co.dat avec le nouveau format
    (où la fréquence est en GHz après 'info:', et les quantums num sont après 'quant:')
    """
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start_data = False
    # Regex pour capturer les champs importants
    # S'adapte à des espaces variables
    line_pattern = re.compile(
        r'^\s*\d+\s+\d+\s+\d+\s+'                     # n, nu, nl (non capturés ici mais parsés pour sauter les colonnes)
        r'(\S+)\s+'                                  # E(K)
        r'([\d\.E\+\-]+)\s+'                         # Aij(s-1)
        r'quant:\s*(\d+);\s*(\d+);\s*(\d+);\s*(\d+);\s*' # vu, Ju, vl, Jl
        r'info:\s*([\d.]+)\s*GHz.*$'                 # Frequency (GHz)
    )

    for line in lines:
        if line.strip() == '#---data--------------------------------------------------------------------------------------------------------------------------':
            start_data = True
            continue
        if not start_data or line.strip() == '':
            continue

        match = line_pattern.search(line)
        if not match:
            # print(f"Warning: Line format not matched in {source_file_name}: {line.strip()}")
            continue

        try:
            e_k_val = float(match.group(1))
            aij = float(match.group(2))
            vu = int(match.group(3))
            ju = int(match.group(4))
            vl = int(match.group(5))
            jl = int(match.group(6))
            frequency_ghz = float(match.group(7))

            frequency_hz = frequency_ghz * 1e9 # GHz -> Hz
            wavelength_m = c_m_s / frequency_hz # Hz -> m
            teq_k = (h_J_s * frequency_hz) / k_J_K

            # E(cm-1) à partir de l'énergie en K ou de la fréquence
            # E(cm-1) = (E_K * k) / (h * c_cm_s)
            e_cm_1 = e_k_val * k_J_K / (h_J_s * c_cm_s)

            data.append({
                'Molecule': mol_type,
                'vu': vu, 'Ju': ju, 'vl': vl, 'Jl': jl,
                'Aij_s-1': aij,
                'E_cm-1': e_cm_1,
                'E_K': e_k_val,
                'Wavelength_m': wavelength_m,
                'Frequency_GHz': frequency_ghz,
                'Teq_K': teq_k,
                'Source_File': source_file_name
            })
        except ValueError as e:
            print(f"Error parsing line in {source_file_name}: {line.strip()} - {e}")
            continue
    return pd.DataFrame(data)









def plot_emission_spectrum(dataframe):
    """
    Trace le spectre d'émission à partir du DataFrame.
    Les raies de H2 et CO sont affichées avec des couleurs et des styles de marqueurs différents.

    Args:
        dataframe (pd.DataFrame): DataFrame contenant les données du spectre,
                                  avec au minimum 'molecule', 'wavelength_nm' et 'intensity'.
    """
    fig, ax1 = plt.subplots(figsize=(15, 6))
    # Filtrer les données pour CO et H2
    # S'assurer que 'wavelength_nm' et 'intensity' ne sont pas NaN
    df_co = dataframe[dataframe['molecule'] == 'CO'].dropna(subset=['wavelength_nm', 'intensity'])
    df_h2 = dataframe[dataframe['molecule'] == 'H2'].dropna(subset=['wavelength_nm', 'intensity'])

    # Tracer les raies de CO
    if not df_co.empty:
        ax1.scatter(df_co['wavelength_nm']*1e-9, df_co["intensity"],#['Aij_s-1'], 
            color='red', marker='o', s=10,
            label='CO Emission Lines')
 
# Tracer les raies de H2
    if not df_h2.empty:
        ax1.scatter(df_h2['wavelength_nm']*1e-9, df_h2["intensity"],#['Aij_s-1'], 
                color='black', marker='o', s=10,
                label='H2 Emission Lines')

    ax1.set_xlabel('Longueur d\'onde (m)', fontsize=25)
    ax1.set_ylabel('Intensité (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)', fontsize=25)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.invert_xaxis()
    ax1.tick_params(axis='x', labelsize=25)
    ax1.tick_params(axis='y', labelsize=25)

    plt.title('Spectre d\'émission de CO et H2', fontsize=25)
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend()
    

# Créer l'axe secondaire en haut (location='top')
# en utilisant les fonctions de conversion
    secax_nu = ax1.secondary_xaxis('top', functions=(lambda_to_nu, nu_to_lambda))
    secax_nu.set_xlabel('Fréquence (Hz)')
    secax_nu.tick_params(axis='x', labelsize=25)


    '''
    secax_temp = ax1.secondary_xaxis('top', functions=(temp_to_nu, nu_to_temp)) # Corrigé: la fonction inverse doit être nu_to_temp, pas nu_to_nu
    secax_temp = ax1.secondary_xaxis('top', functions=(lambda_cm_to_temp_k, temp_k_to_lambda_cm))

    secax_temp.set_xlabel('Température (K)', fontsize=25)
    secax_temp.set_xscale('log') # Doit aussi être en log
    '''    
    plt.tight_layout()
    plt.show()
    
    
def process_molecular_data(stat_file, level_dat_file, lines_dat_file,mol="co"):
    """
    Reads molecular intensity data and energy levels, combines them, calculates
    column density for each transition, and returns a comprehensive DataFrame.

    Args:
        stat_file (str): Path to the .stat file containing line intensities.
        level_dat_file (str): Path to the level_mol.dat file containing energy levels and degeneracies.
        lines_dat_file (str): Path to the lines.dat file containing transition coefficients (Aij) and frequencies.

    Returns:
        pd.DataFrame: A DataFrame containing combined data, calculated column densities,
                      and column densities normalized by degeneracy and energy levels.
    """

    # Constants
    H = 6.62607015e-34  # Planck constant (J*s)
    K_B = 1.380649e-23  # Boltzmann constant (J/K)

    # --- Part 1: Read and process data ---

    # 1. Read .stat file
    # We use a custom parser because the format is non-standard and we only need specific lines.
    stat_data = []
    with open(stat_file, 'r') as f:
        for line in f:
            if 'value' in line and f'inta_00_{mol}' in line:
                parts = [p.strip() for p in line.split('#')]
                if len(parts) >= 3:
                    # Extract transition name and intensity
                    transition_name = parts[1]
                    try:
                        intensity_erg = float(parts[2])
                        stat_data.append([transition_name, intensity_erg])
                    except ValueError:
                        continue  # Skip lines where intensity cannot be parsed

    df_stat = pd.DataFrame(stat_data, columns=['Transition_Spectro', 'Intensity_erg_cm2_s_sr'])
    #print(df_stat)
    # 2. Extract vibrational and rotational quantum numbers (v, J) from .stat data
    # We use regex to parse the transition name (e.g., 'inta_00_co_v10_j10__v9_j11')
    
    
    def parse_quantum_numbers(transition):
        # Regex to find high (v_u, J_u) and low (v_l, J_l) levels
        match = re.match(rf'inta_00_{mol}_v(\d+)_j(\d+)__v(\d+)_j(\d+)', transition)
        if match:
            # We assume the first pair is the upper level (u) and the second is the lower level (l)
            v_u, j_u, v_l, j_l = map(int, match.groups())
            return pd.Series([v_u, j_u, v_l, j_l])
        return pd.Series([None, None, None, None])

    df_stat[['v_u', 'J_u', 'v_l', 'J_l']] = df_stat['Transition_Spectro'].apply(parse_quantum_numbers)

    # 3. Read level_mol.dat file (energy levels and degeneracy)
    # We read this file skipping the header lines that start with '#' or '@' until we hit the data section.
    # The data columns are n, g, E(K), OP, v, J
    df_levels = pd.read_csv(level_dat_file, sep='\s+', comment='#', header=None, skiprows=12, usecols=[0, 1, 2, 4, 5], 
                            names=['n_level', 'g_degeneracy', 'E_K', 'v', 'J'])
    # 4. Read lines.dat file (Aij, E(K), frequency, quant numbers)
    # We read this file skipping the header lines. The relevant columns are Aij, vu, Ju, vl, Jl, and E(K)
    
    df_lines = pd.read_csv(lines_dat_file, sep='\s+', comment='#', header=None, skiprows=4, 
                               usecols=[0, 1, 2, 3, 4, 5,6,7, 8, 9, 10, 11,12],
                               names=['n_transition', 'n_u', 'n_l', 'E_transition_K', 'Aij', 
                                      'quant_prefix', 'v_u', 'J_u', 'v_l', 'J_l','info','Freq','unitFreq'],
                               skipinitialspace=True, dtype={'quant_prefix': str, 'v_u': str, 'J_u': str, 'v_l': str, 'J_l': str})
        

    # Remove the 'quant:' and trailing ';' from the columns if present
    df_lines['v_u'] = df_lines['v_u'].astype(str).str.replace(';', '').astype(int)
    df_lines['J_u'] = df_lines['J_u'].astype(str).str.replace(';', '').astype(int)
    df_lines['v_l'] = df_lines['v_l'].astype(str).str.replace(';', '').astype(int)
    df_lines['J_l'] = df_lines['J_l'].astype(str).str.replace(';', '').astype(int)

    # 5. Combine DataFrames
    # We merge df_stat with df_lines using (v_u, J_u, v_l, J_l) to get Aij and E_transition_K
    df_combined = pd.merge(df_stat, df_lines[['n_u','n_l','v_u', 'J_u', 'v_l', 'J_l', 'Aij', 'E_transition_K']],
                           on=['v_u', 'J_u', 'v_l', 'J_l'], how='left')

    df_levels_upper = df_levels.rename(columns={
        'n_level': 'n_u', 
        'g_degeneracy': 'g_u', 
        'E_K': 'E_u_K', 
        'v': 'v_u', 
        'J': 'J_u'
    })
    
    # Select only the relevant columns from df_levels_upper for the merge
    df_combined = pd.merge(df_combined, df_levels_upper[['n_u', 'g_u', 'E_u_K', 'v_u', 'J_u']],
                           on='n_u', how='left', suffixes=('', '_level_u'))

    # Merge with df_levels for the lower level (l) properties
    # We merge using 'n_l' from df_combined (originally from df_lines) and 'n_level' from df_levels.
    # We rename columns of df_levels with '_l' suffix for the merge.
    df_levels_lower = df_levels.rename(columns={
        'n_level': 'n_l', 
        'g_degeneracy': 'g_l', 
        'E_K': 'E_l_K', 
        'v': 'v_l', 
        'J': 'J_l'
    })
    # Select only the relevant columns from df_levels_lower for the merge
    df_combined = pd.merge(df_combined, df_levels_lower[['n_l', 'g_l', 'E_l_K', 'v_l', 'J_l']],
                           on='n_l', how='left', suffixes=('', '_level_l'))

    
    #df_combined = df_combined.drop(columns=['n_u', 'n_l'])

    # --- Part 2: Calculate Column Density ---

    # Formula: $I_{ij} = (\frac{N_i}{g_i})\frac{g_i h\nu A_{ij}}{4\pi}$ 
    # We want $N_i$ (N_u in this case)
    # $N_u = \frac{I_{ij} \cdot 4\pi}{h\nu A_{ij}}$ 

    df_combined['Frequency_Hz'] = (df_combined['E_transition_K'] * K_B) / H

    # 2. Convert Intensity from erg cm^-2 s^-1 sr^-1 to J m^-2 s^-1 sr^-1
    # 1 erg = 1e-7 J
    # 1 cm^2 = 1e-4 m^2
    # I_SI = I_erg * (1e-7 J / 1 erg) * (1e4 cm^2 / 1 m^2) = I_erg * 1e-3
    df_combined['Intensity_SI'] = df_combined['Intensity_erg_cm2_s_sr'] * 1e-3

    # 3. Calculate Column Density ($N_u$)
    # $N_u = \frac{I_{SI} \cdot 4\pi}{h \cdot \nu \cdot A_{ij}}$
    
    # Ensure Aij and Frequency are not zero to avoid division errors
    #df_combined['Aij'] = df_combined['Aij'].replace(0, np.nan)
    #df_combined['Frequency_Hz'] = df_combined['Frequency_Hz'].replace(0, np.nan)
    """
    numerator = df_combined['Intensity_SI'] * 4 * np.pi
    denominator = H * df_combined['Frequency_Hz'] * df_combined['Aij']
    
    df_combined['N_u_Column_Density'] = numerator / denominator
    """
    
    df_combined['N_u_Column_Density'] = 4 * np.pi * df_combined["Intensity_erg_cm2_s_sr"]/\
        (1e7 * k * df_combined["E_transition_K"] * df_combined["Aij"])

    # 4. Calculate Column Density normalized by degeneracy ($N_u / g_u$)
    df_combined['N_u_over_g_u'] = df_combined['N_u_Column_Density'] / df_combined['g_u']

    # --- Part 3: Final Formatting and Output ---
    # Select and rename columns for the final output DataFrame
    final_df = df_combined[[
        'Transition_Spectro', 
        'v_u', 
        'J_u', 
        'v_l', 
        'J_l', 
        'Intensity_erg_cm2_s_sr', 
        'N_u_Column_Density', 
        'N_u_over_g_u', 
        'g_u', 
        'g_l', 
        'E_u_K', 
        'E_l_K'
    ]].copy()

    # Rename columns to match the requested output format (v_u, J_u, v_l, J_l were retained in the temporary df_combined after the initial merge for clarity)
    final_df.rename(columns={
        'Intensity_erg_cm2_s_sr': 'Intensity_erg_cm-2_s-1_sr-1',
        'N_u_Column_Density': 'Column_Density_Nu',
        'N_u_over_g_u': 'Column_Density_Normalized_by_Degeneracy',
        'E_u_K': 'E_upper_K',
        'E_l_K': 'E_lower_K',
        'g_u': 'Degeneracy_upper_gu',
        'g_l': 'Degeneracy_lower_gl'
    }, inplace=True)
    
    # Add back the v_u, J_u, v_l, J_l columns which were dropped for simplicity during the merge process
    #df_stat_retained = df_stat[['Transition_Spectro', 'v_u', 'J_u', 'v_l', 'J_l']]
    #final_df = pd.merge(final_df, df_stat_retained, on='Transition_Spectro', how='left')
    print(final_df.columns)

    # Reorder columns for final presentation
    final_df = final_df[[
        'Transition_Spectro', 
        'v_u', 
        'J_u', 
        'v_l', 
        'J_l', 
        'Intensity_erg_cm-2_s-1_sr-1', 
        'Column_Density_Nu', 
        'Column_Density_Normalized_by_Degeneracy',
        'Degeneracy_upper_gu',
        'Degeneracy_lower_gl',
        'E_upper_K', 
        'E_lower_K'
    ]]

    return final_df