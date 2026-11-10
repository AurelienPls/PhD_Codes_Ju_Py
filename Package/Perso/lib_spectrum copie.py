#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 10:54:17 2026

@author: aurelien
"""
import to_PDR
import read
import viz
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import math
import sys
import plotly.colors as pc
from matplotlib.ticker import MultipleLocator

chemin_modules = "/Users/aurelien/Desktop/PhD/Codes/Package/Perso/"
sys.path.append(chemin_modules)

########################################################
# Data Processing and importation
########################################################


def filter_fitted_lines(df, col_vrad="v_rad", min_vel=None, max_vel=None, lines_to_remove=None, col_snr="snr", min_snr=None):
    """
    Filtre un dataframe de raies fittées selon la vitesse radiale, le nom de la raie ou un SNR minimal.
    """
    df_filtered = df.copy()
    
    # 1. Filtrage par vitesse radiale
    if min_vel is not None and col_vrad in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[col_vrad] >= min_vel]
    if max_vel is not None and col_vrad in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[col_vrad] <= max_vel]
        
    # 2. Filtrage par exclusion manuelle (nom de la raie)
    if lines_to_remove is not None:
        df_filtered = df_filtered[~df_filtered["Line"].isin(lines_to_remove)]
        
    # 3. Filtrage par rapport au signal sur bruit (SNR)
    if min_snr is not None and col_snr in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[col_snr] >= min_snr]
        
    return df_filtered



def get_co_lines_with_overlap(file_path="tableb1.dat"):
    """
    Lit tableb1.dat en incluant les informations sur le chevauchement (overlapping)
    et filtre pour les raies de CO.
    """
    colspecs = [
        (0, 19),    # Line_ID (Nom de la raie)
        (19, 20),   # n_Line (Note overlap: '+' ou vide) -> Bytes 20
        (20, 22),   # NLine (Numéro séquentiel overlap) -> Bytes 21-22
        (23, 29),   # Wave_um
        # HII Region
        (37, 45), (46, 54),
        # Atomic PDR
        (62, 70), (71, 79),
        # DF1
        (87, 95), (96, 104),
        # DF2
        (112, 120), (121, 129),
        # DF3
        (137, 145), (146, 154)]

    names = [
        "Line_ID", "Overlap_Note", "Overlap_Num", "Wave_um",
        "HII_Int", "HII_Err",
        "APDR_Int", "APDR_Err",
        "DF1_Int", "DF1_Err",
        "DF2_Int", "DF2_Err",
        "DF3_Int", "DF3_Err"
    ]

    try:
        df = pd.read_fwf(file_path, colspecs=colspecs,
                         names=names, header=None)
        df['Line_ID'] = df['Line_ID'].astype(str).str.strip()
        df['Overlap_Note'] = df['Overlap_Note'].astype(str).str.strip()

        df_co = df[df['Line_ID'].str.contains(
            "CO", case=False, na=False)].copy()

        cols_numeriques = names[2:]
        for col in cols_numeriques:
            df_co[col] = pd.to_numeric(df_co[col], errors='coerce')
        df_co = df_co.reset_index(drop=True)
        return df_co

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file_path}' est introuvable.")
        return None


def read_obs(path, name):
    ColNames = ["1", "2", "Int", "Err"]
    df_observation = read.dat_to_pd(
        path, name, line_data=6, ColNames=ColNames, tofloat=False)
    df_observation["Line"] = df_observation["1"] + \
        " " + df_observation["2"].str.rstrip(',')
    liste = df_observation["Line"].tolist()
    return liste, df_observation


# Fit And processing
def Analyse_Spectrum_SingleGaussian(df_spectrum, levels_config, R=2700):
    """Appelle analyse lines custom et fait une petite routine avec """
    # df_level_co = to_PDR.Level_to_pd("co")
    df_line_co = to_PDR.Lines_to_pd("co", version="HITRAN")
    df_line_co["Frequency"] = df_line_co["Description"].str.split(
        ' GHz').str[0].astype(float)
    # Vitesse de la lumière en Mégamètres par seconde (10^6 m/s)
    C_MM_PER_S = 299.792458
    df_line_co['wavelength'] = C_MM_PER_S / df_line_co['Frequency'] * 1000

    print("Traitement des niveaux...")

    processed_levels = []

    for name, line_list, params, color in levels_config:
        df_subset = df_line_co[df_line_co['Line'].isin(line_list)].copy()

        if not df_subset.empty:
            df_res = analyze_lines_custom(df_spectrum, df_subset, params, R=R)

            processed_levels.append({
                "name": name,
                "df_res": df_res,
                "df_theo": df_subset,
                "params": params,
                "color": color})
            print(
                f" -> {name} traité ({len(df_res[df_res['success']==True])} fits réussis)")

    list_dfs = []
    for level in processed_levels:
        df_res = level["df_res"]
        df_theo = level["df_theo"]
        df_success = df_res[df_res['success'] == True].copy()
        merged = pd.merge(df_success, df_theo[['wavelength', 'Line']],
                          left_on='wavelength_th', right_on='wavelength',
                          how='left')
        if 'wavelength' in merged.columns:
            merged = merged.drop(columns=['wavelength'])
        list_dfs.append(merged)

    if list_dfs:
        df_final_fits = pd.concat(list_dfs, ignore_index=True)
    else:
        df_final_fits = pd.DataFrame()

    c_kms = 299792.458
    df_final_fits['Delta_Lambda'] = df_final_fits['wavelength_obs'] -  df_final_fits['wavelength_th']
    df_final_fits['Velocity_km_s'] = c_kms * (df_final_fits['Delta_Lambda'] / df_final_fits['wavelength_th'])
    df_final_fits = df_final_fits.sort_values(
        'wavelength_th').reset_index(drop=True)

    dfv1 = df_final_fits[df_final_fits['Line'].str.contains('1-0')].copy()
    dfv2 = df_final_fits[df_final_fits['Line'].str.contains('2-1')].copy()

    # print(f"Nombre de raies 1-0 : {len(dfv1)}")
    # print(f"Nombre de raies 2-1 : {len(dfv2)}")
    return (df_final_fits, dfv1, dfv2, processed_levels)


def apply_overrides(original_config, overrides_dict):
    """
    Sépare les raies spécifiques de leurs groupes d'origine et crée 
    des groupes individuels pour elles avec les nouveaux paramètres.
    """
    new_config = []
    
    # --- Étape 1 : Nettoyer les groupes existants ---
    for name, line_list, params, color in original_config:
        # On crée une nouvelle liste sans les raies qui sont dans le dictionnaire d'exceptions
        cleaned_list = [line for line in line_list if line not in overrides_dict]
        
        # Si la liste n'est pas vide après nettoyage, on garde le groupe
        if cleaned_list:
            new_config.append((name, cleaned_list, params, color))
            
    # --- Étape 2 : Créer les groupes individuels pour les exceptions ---
    for line_name, new_params in overrides_dict.items():
        
        # Astuce : On essaye de retrouver la couleur d'origine pour garder la cohérence graphique
        line_color = "red" # Couleur par défaut si non trouvée (ou pour marquer le changement)
        for _, old_list, _, old_color in original_config:
            if line_name in old_list:
                line_color = old_color # On reprend la couleur du groupe d'origine
                break
        
        # On crée un tuple de config pour cette raie seule
        # Nom du groupe = "Custom_" + nom de la raie
        single_group = (f"Custom_{line_name}", [line_name], new_params, line_color)
        new_config.append(single_group)
        
    return new_config


def model_gauss_linear(x, A, mu, sigma, m, c):
    """
    Modèle : Gaussienne + Continuum Linéaire
    F(x) = A * exp(-(x-mu)^2 / (2*sigma^2)) + mx + c
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + m * x + c


def model_2gauss_linear(x, A1, mu1, sigma1, A2, mu2, sigma2, m, c):
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    g2 = A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    return g1 + g2 + m * x + c


def analyze_lines(df_spectrum, df_lines, R=2700, window_width=0.04, width_guess=1, nedge=10):
    wave = df_spectrum["Wavelength"].values
    flux = df_spectrum["Int"].values

    results = []
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))

    # Constante C en microns/s pour la conversion
    C_MICRON_S = 2.99792458e14

    print("Démarrage de l'ajustement des raies...")

    for idx, row in tqdm(df_lines.iterrows(), total=df_lines.shape[0], desc="Fitting Lines"):
        lambda_0 = row['wavelength']

        # 1. Découpage fenêtre
        mask = (wave >= lambda_0 - window_width /
                2) & (wave <= lambda_0 + window_width/2)
        x_local = wave[mask]
        y_local = flux[mask]

        if len(x_local) < 5:
            results.append({'wavelength_th': lambda_0, 'success': False})
            continue

        # 2. Guess
        sigma_guess = (lambda_0 / R / fwhm_factor) * width_guess
        amp_guess = np.max(y_local) - np.min(y_local)

        n_edge = max(2, len(x_local) // nedge)
        y_left = np.mean(y_local[:n_edge])
        y_right = np.mean(y_local[-n_edge:])
        # Pente = (y_droite - y_gauche) / largeur_totale
        slope_guess = (y_right - y_left) / (x_local[-1] - x_local[0])

        # 2. Estimation de l'ordonnée à l'origine (Intercept)
        # Formule : c = y - mx
        x_mean = np.mean(x_local)
        y_base = np.percentile(y_local, 20)  # Le niveau de référence bas
        intercept_guess = y_base - (slope_guess * x_mean)

        # 3. Création du vecteur p0 avec pente et ordonnée dynamiques
        p0 = [amp_guess, lambda_0, sigma_guess, slope_guess, intercept_guess]

        try:
            popt, pcov = curve_fit(
                model_gauss_linear, x_local, y_local, p0=p0, maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
            amp_fit, mu_fit, sigma_fit = popt[0], popt[1], popt[2]
            err_amp, err_mu, err_sigma = perr[0], perr[1], perr[2]

            # 4. Flux MJy * um
            line_flux = amp_fit * abs(sigma_fit) * np.sqrt(2 * np.pi)
            # Erreur Flux MJy * um
            if amp_fit != 0 and sigma_fit != 0:
                rel_err = np.sqrt((err_amp/amp_fit)**2 +
                                  (err_sigma/sigma_fit)**2)
                err_flux = abs(line_flux * rel_err)
            else:
                err_flux = 0

            # --- 5. CONVERSION CGS (erg s-1 cm-2 sr-1) ---
            # Facteur de conversion local dépendant de la longueur d'onde observée
            # 1 MJy = 1e-17 erg s-1 cm-2 Hz-1
            # I = Integral(S_nu d_lambda) * (c / lambda^2)
            conv_factor = (C_MICRON_S / (mu_fit**2)) * 1e-17
            int_cgs = line_flux * conv_factor
            err_cgs = err_flux * conv_factor
            delta_lambda = mu_fit - lambda_0
            snr = amp_fit / err_amp if err_amp > 0 else 0
            
            margin = max(3, int(0.05 * len(y_local)))
            y_continuum = np.concatenate((y_local[:margin], y_local[-margin:]))
            rms_cont = np.std(y_continuum)
            snr2 = amp_fit / rms_cont if rms_cont > 0 else 0
            
            results.append({
                'wavelength_th': lambda_0,
                'wavelength_obs': mu_fit,
                'delta_lambda': delta_lambda,
                'amplitude': amp_fit,
                'err_amplitude': err_amp,
                'sigma': sigma_fit,
                'err_sigma': err_sigma,
                'flux_integral': line_flux,       # Unité : MJy * um
                'err_flux_integral': err_flux,
                'Int_cgs': int_cgs,               # Unité : erg s-1 cm-2 sr-1
                'Err_cgs': err_cgs,               # Erreur associée
                'continuum_m': popt[3],
                'continuum_c': popt[4],
                'snr': snr,
                'snr2': snr2,
                'success': True
            })

        except Exception:
            results.append({'wavelength_th': lambda_0, 'success': False})

    return pd.DataFrame(results)


def analyze_lines_custom(df_spectrum, df_lines, params, R=2700):
    """
    Version adaptée qui prend la liste 'params' [win_width, guess_width, nedge]
    """
    window_width, width_guess, nedge = params[0], params[1], int(params[2])

    wave = df_spectrum["Wavelength"].values
    flux = df_spectrum["Int"].values
    results = []
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    C_MICRON_S = 2.99792458e14

    # Pas de tqdm ici pour éviter 6 barres de chargement imbriquées, on le fait au niveau global
    for idx, row in df_lines.iterrows():
        lambda_0 = row['wavelength']

        # 1. Découpage fenêtre
        mask = (wave >= lambda_0 - window_width /
                2) & (wave <= lambda_0 + window_width/2)
        x_local = wave[mask]
        y_local = flux[mask]

        if len(x_local) < 5:
            results.append({'wavelength_th': lambda_0, 'success': False})
            continue

        # 2. Guess
        sigma_guess = (lambda_0 / R / fwhm_factor) * width_guess
        amp_guess = np.max(y_local) - np.min(y_local)

        # Continuum Guess
        n_edge_pts = max(1, len(x_local) // nedge)
        y_left = np.mean(y_local[:n_edge_pts])
        y_right = np.mean(y_local[-n_edge_pts:])

        slope_guess = (y_right - y_left) / \
            (x_local[-1] - x_local[0]
             ) if (x_local[-1] - x_local[0]) != 0 else 0
        intercept_guess = np.percentile(
            y_local, 20) - (slope_guess * np.mean(x_local))

        p0 = [amp_guess, lambda_0, sigma_guess, slope_guess, intercept_guess]

        try:
            popt, pcov = curve_fit(
                model_gauss_linear, x_local, y_local, p0=p0, maxfev=5000)
            perr = np.sqrt(np.diag(pcov))

            amp_fit, mu_fit, sigma_fit = popt[0], popt[1], abs(popt[2])

            # Flux et Conversion
            line_flux = amp_fit * abs(sigma_fit) * np.sqrt(2 * np.pi)
            conv_factor = (C_MICRON_S / (mu_fit**2)) * 1e-17
            int_cgs = line_flux * conv_factor

            # SNR
            snr = amp_fit / perr[0] if perr[0] > 0 else 0

            margin = max(3, int(0.05 * len(y_local)))
            y_continuum = np.concatenate((y_local[:margin], y_local[-margin:]))
            rms_cont = np.std(y_continuum)
            snr2 = amp_fit / rms_cont if rms_cont > 0 else 0
            
            results.append({
                'wavelength_th': lambda_0,
                'wavelength_obs': mu_fit,
                'amplitude': amp_fit,
                'sigma': sigma_fit,
                'flux_integral': line_flux,
                'Int_cgs': int_cgs,
                'continuum_m': popt[3],
                'continuum_c': popt[4],
                'snr': snr,
                'snr2': snr2,
                'success': True
            })

        except Exception:
            results.append({'wavelength_th': lambda_0, 'success': False})

    return pd.DataFrame(results)


def fit_manual_lines(df_spectrum, df_lines_ref, target_names, manual_guesses={}, R=2700, window_width=0.04,plot=True):
    mask_names = df_lines_ref['Line'].isin(target_names)
    df_subset = df_lines_ref[mask_names].copy()

    if df_subset.empty:
        return pd.DataFrame(), None, None

    wave = df_spectrum["Wavelength"].values
    flux = df_spectrum["Int"].values
    results = []
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    C_MICRON_S = 2.99792458e14

    for idx, row in df_subset.iterrows():
        line_id = row['Line']
        lambda_0 = row['wavelength']

        mask = (wave >= lambda_0 - window_width /
                2) & (wave <= lambda_0 + window_width/2)
        x_local = wave[mask]
        y_local = flux[mask]

        if len(x_local) < 5:
            continue

        sigma_guess = (lambda_0 / R / fwhm_factor)
        amp_guess = np.max(y_local) - np.min(y_local)

        s_g = (y_local[-1] - y_local[0]) / (x_local[-1] - x_local[0])
        i_g = np.mean(y_local) - (s_g * np.mean(x_local))

        is_fixed_continuum = False
        fixed_m = 0.0
        fixed_c = 0.0

        if line_id in manual_guesses:
            user_p = manual_guesses[line_id]
            if 'amp' in user_p:
                amp_guess = user_p['amp']
            if 'sigma' in user_p:
                sigma_guess = user_p['sigma']

            if 'slope' in user_p or 'intercept' in user_p:
                is_fixed_continuum = True
                fixed_m = user_p.get('slope', 0.0)
                fixed_c = user_p.get('intercept', np.mean(y_local))

        try:
            if is_fixed_continuum:
                def model_frozen_cont(x, A, mu, sigma):
                    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + fixed_m * x + fixed_c

                p0 = [amp_guess, lambda_0, sigma_guess]
                popt, pcov = curve_fit(
                    model_frozen_cont, x_local, y_local, p0=p0, maxfev=10000)
                perr = np.sqrt(np.diag(pcov))

                amp_fit, mu_fit, sigma_fit = popt[0], popt[1], abs(popt[2])
                m_fit, c_fit = fixed_m, fixed_c
                err_amp = perr[0]

            else:
                p0 = [amp_guess, lambda_0, sigma_guess, s_g, i_g]
                popt, pcov = curve_fit(
                    model_gauss_linear, x_local, y_local, p0=p0, maxfev=10000)
                perr = np.sqrt(np.diag(pcov))

                amp_fit, mu_fit, sigma_fit = popt[0], popt[1], abs(popt[2])
                m_fit, c_fit = popt[3], popt[4]
                err_amp = perr[0]

            line_flux = amp_fit * sigma_fit * np.sqrt(2 * np.pi)
            conv_factor = (C_MICRON_S / (mu_fit**2)) * 1e-17
            int_cgs = line_flux * conv_factor

            c_kms = 299792.458
            delta_l = mu_fit - lambda_0
            velocity = c_kms * (delta_l / lambda_0)
            snr = amp_fit / err_amp if err_amp > 0 else 0

            margin = max(3, int(0.05 * len(y_local)))
            y_continuum = np.concatenate((y_local[:margin], y_local[-margin:]))
            rms_cont = np.std(y_continuum)
            
            # SNR2 = amp_fit / rms_cont
            snr2 = amp_fit / rms_cont if rms_cont > 0 else 0
            
            results.append({
                'Line': line_id,
                'wavelength_th': lambda_0,
                'wavelength_obs': mu_fit,
                'amplitude': amp_fit,
                'sigma': sigma_fit,
                'Int_cgs': int_cgs,
                'continuum_m': m_fit,
                'continuum_c': c_fit,
                'Velocity_km_s': velocity,
                'snr': snr,
                'snr2': snr2,
                'success': True
            })
        except Exception as e:
            print(f"Echec fit pour {line_id}: {e}")

    df_res = pd.DataFrame(results)

    if not df_res.empty and plot== True :
        fig, axes = plot_gallery_fits(df_spectrum, df_res, n_cols=min(
            4, len(df_res)), window_view=window_width)
        return df_res, fig, axes
    else:
        return df_res, None, None



def fit_manual_lines2(df_spectrum, df_lines_ref, target_names, manual_guesses={}, R=2700, window_width=0.04, plot=True):
    """"UTILISEE ACTUELLEMENT MARS 2026 pour les calculs avec continuum"""
    # 1. Filtrer pour ne garder que les raies demandées
    mask_names = df_lines_ref['Line'].isin(target_names)
    df_subset = df_lines_ref[mask_names].copy()

    if df_subset.empty:
        return pd.DataFrame(), None, None

    # --- DÉBUT DE LA MODIFICATION : FORCER L'ORDRE ---
    # On transforme la colonne 'Line' en catégorie avec l'ordre exact de target_names
    df_subset['Line'] = pd.Categorical(df_subset['Line'], categories=target_names, ordered=True)
    # On trie le dataframe pour respecter cet ordre et on réinitialise l'index
    df_subset = df_subset.sort_values('Line').reset_index(drop=True)
    # --- FIN DE LA MODIFICATION ---

    wave = df_spectrum["Wavelength"].values
    flux = df_spectrum["Int"].values
    results = []
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    C_MICRON_S = 2.99792458e14

    # La boucle itère maintenant dans l'ordre strict de target_names
    for idx, row in df_subset.iterrows():
        line_id = row['Line']
        lambda_0 = row['wavelength']

        mask = (wave >= lambda_0 - window_width / 2) & (wave <= lambda_0 + window_width/2)
        x_local = wave[mask]
        y_local = flux[mask]

        if len(x_local) < 5:
            continue

        sigma_guess = (lambda_0 / R / fwhm_factor)
        amp_guess = np.max(y_local) - np.min(y_local)

        s_g = (y_local[-1] - y_local[0]) / (x_local[-1] - x_local[0])
        i_g = np.mean(y_local) - (s_g * np.mean(x_local))

        is_fixed_continuum = False
        fixed_m = 0.0
        fixed_c = 0.0

        if line_id in manual_guesses:
            user_p = manual_guesses[line_id]
            if 'amp' in user_p:
                amp_guess = user_p['amp']
            if 'sigma' in user_p:
                sigma_guess = user_p['sigma']

            if 'slope' in user_p or 'intercept' in user_p:
                is_fixed_continuum = True
                fixed_m = user_p.get('slope', 0.0)
                
                # Conversion de l'intercept central en intercept à x=0
                valeur_centrale_y = user_p.get('intercept', np.mean(y_local))
                fixed_c = valeur_centrale_y - (fixed_m * lambda_0)

        try:
            if is_fixed_continuum:
                def model_frozen_cont(x, A, mu, sigma):
                    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + fixed_m * x + fixed_c

                p0 = [amp_guess, lambda_0, sigma_guess]
                popt, pcov = curve_fit(
                    model_frozen_cont, x_local, y_local, p0=p0, maxfev=10000)
                perr = np.sqrt(np.diag(pcov))

                amp_fit, mu_fit, sigma_fit = popt[0], popt[1], abs(popt[2])
                m_fit, c_fit = fixed_m, fixed_c
                err_amp = perr[0]

            else:
                p0 = [amp_guess, lambda_0, sigma_guess, s_g, i_g]
                popt, pcov = curve_fit(
                    model_gauss_linear, x_local, y_local, p0=p0, maxfev=10000)
                perr = np.sqrt(np.diag(pcov))

                amp_fit, mu_fit, sigma_fit = popt[0], popt[1], abs(popt[2])
                m_fit, c_fit = popt[3], popt[4]
                err_amp = perr[0]

            line_flux = amp_fit * sigma_fit * np.sqrt(2 * np.pi)
            conv_factor = (C_MICRON_S / (mu_fit**2)) * 1e-17
            int_cgs = line_flux * conv_factor

            c_kms = 299792.458
            delta_l = mu_fit - lambda_0
            velocity = c_kms * (delta_l / lambda_0)
            snr = amp_fit / err_amp if err_amp > 0 else 0

            margin = max(3, int(0.05 * len(y_local)))
            y_continuum = np.concatenate((y_local[:margin], y_local[-margin:]))
            rms_cont = np.std(y_continuum)
            
            snr2 = amp_fit / rms_cont if rms_cont > 0 else 0
            
            results.append({
                'Line': line_id,
                'wavelength_th': lambda_0,
                'wavelength_obs': mu_fit,
                'amplitude': amp_fit,
                'sigma': sigma_fit,
                'Int_cgs': int_cgs,
                'continuum_m': m_fit,
                'continuum_c': c_fit, 
                'Velocity_km_s': velocity,
                'snr': snr,
                'snr2': snr2,
                'success': True
            })
        except Exception as e:
            print(f"Echec fit pour {line_id}: {e}")

    df_res = pd.DataFrame(results)

    if not df_res.empty and plot == True:
        # La galerie conservera l'ordre car df_res est construit dans l'ordre de target_names
        fig, axes = plot_gallery_fits(df_spectrum, df_res, n_cols=min(4, len(df_res)), window_view=window_width)
        return df_res, fig, axes
    else:
        return df_res, None, None

def fit_unresolved_blendsV2(df_spectrum, df_lines_ref, unresolved_pairs, manual_guesses={}, R=2700, window_width=0.04):
    """
    Ajuste une Gaussienne unique forcée POSITIVE et CONFINÉE dans sa fenêtre.
    """
    wave = df_spectrum["Wavelength"].values
    flux = df_spectrum["Int"].values
    results = []
    
    lines_dict = df_lines_ref.set_index('Line').to_dict('index')
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    C_MICRON_S = 2.99792458e14
    C_KMS = 299792.458

    for pair in unresolved_pairs:
        line1, line2 = pair

        if line1 not in lines_dict or line2 not in lines_dict:
            continue

        lth1 = lines_dict[line1]['wavelength']
        lth2 = lines_dict[line2]['wavelength']
        lambda_center = (lth1 + lth2) / 2.0

        mask = (wave >= lambda_center - window_width / 2) & (wave <= lambda_center + window_width / 2)
        x_local = wave[mask]
        y_local = flux[mask]

        if len(x_local) < 5:
            continue

        sigma_guess = (lambda_center / R / fwhm_factor)
        amp_guess = np.max(y_local) - np.min(y_local)
        if amp_guess <= 0: amp_guess = 1.0

        s_g = (y_local[-1] - y_local[0]) / (x_local[-1] - x_local[0])
        i_g = np.mean(y_local) - (s_g * np.mean(x_local))

        is_fixed_continuum = False
        fixed_m, fixed_c = 0.0, 0.0

        if line1 in manual_guesses:
            user_p = manual_guesses[line1]
            if 'amp' in user_p: amp_guess = user_p['amp']
            if 'sigma' in user_p: sigma_guess = user_p['sigma']
            if 'slope' in user_p or 'intercept' in user_p:
                is_fixed_continuum = True
                fixed_m = user_p.get('slope', 0.0)
                fixed_c = user_p.get('intercept', np.mean(y_local)) - (fixed_m * lambda_center)

        # --- NOUVEAUTÉ : BORNES STRICTES ---
        # On empêche le centre de sortir de la fenêtre locale et l'amplitude d'être négative.
        b_low_m  = lambda_center - window_width/2
        b_high_m = lambda_center + window_width/2

        try:
            if is_fixed_continuum:
                def model_frozen_cont(x, A, mu, sigma):
                    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + fixed_m * x + fixed_c

                p0 = [amp_guess, lambda_center, sigma_guess]
                b_low  = [0, b_low_m, 1e-4]
                b_high = [np.inf, b_high_m, window_width]
                
                popt, pcov = curve_fit(model_frozen_cont, x_local, y_local, p0=p0, bounds=(b_low, b_high), maxfev=10000)
                perr = np.sqrt(np.diag(pcov))

                amp_fit, mu_fit, sigma_fit = popt[0], popt[1], popt[2]
                m_fit, c_fit = fixed_m, fixed_c
                err_amp = perr[0]

            else:
                p0 = [amp_guess, lambda_center, sigma_guess, s_g, i_g]
                b_low  = [0, b_low_m, 1e-4, -np.inf, -np.inf]
                b_high = [np.inf, b_high_m, window_width, np.inf, np.inf]

                popt, pcov = curve_fit(model_gauss_linear, x_local, y_local, p0=p0, bounds=(b_low, b_high), maxfev=10000)
                perr = np.sqrt(np.diag(pcov))

                amp_fit, mu_fit, sigma_fit = popt[0], popt[1], popt[2]
                m_fit, c_fit = popt[3], popt[4]
                err_amp = perr[0]

            # Calculs
            line_flux = amp_fit * sigma_fit * np.sqrt(2 * np.pi)
            int_cgs = line_flux * (C_MICRON_S / (mu_fit**2)) * 1e-17
            snr = amp_fit / err_amp if err_amp > 0 else 0

            margin = max(3, int(0.05 * len(y_local)))
            y_continuum = np.concatenate((y_local[:margin], y_local[-margin:]))
            rms_cont = np.std(y_continuum)
            snr2 = amp_fit / rms_cont if rms_cont > 0 else 0

            # Ligne Raie 1
            results.append({'Line': line1, 'blended_with': line2, 'fit_type': 'UNRESOLVED',
                            'wavelength_th': lth1, 'wavelength_obs': mu_fit, 'amplitude': amp_fit,
                            'sigma': sigma_fit, 'Int_cgs': int_cgs, 'continuum_m': m_fit, 'continuum_c': c_fit, 
                            'Velocity_km_s': C_KMS * ((mu_fit - lth1) / lth1), 'snr': snr, 'snr2': snr2, 'success': True})

            # Ligne Raie 2
            results.append({'Line': line2, 'blended_with': line1, 'fit_type': 'UNRESOLVED',
                            'wavelength_th': lth2, 'wavelength_obs': mu_fit, 'amplitude': amp_fit,
                            'sigma': sigma_fit, 'Int_cgs': int_cgs, 'continuum_m': m_fit, 'continuum_c': c_fit, 
                            'Velocity_km_s': C_KMS * ((mu_fit - lth2) / lth2), 'snr': snr, 'snr2': snr2, 'success': True})

        except Exception as e:
            print(f"Echec fit {line1}/{line2}: {e}")

    df_res = pd.DataFrame(results)
    for c in ['amp_blend', 'mu_blend', 'sig_blend']:
        if c not in df_res.columns: df_res[c] = np.nan
    return df_res




def analyze_lines_flexible(df_spectrum, df_lines, default_params=[0.08, 2.0, 2], R=2700):
    """
    Analyse simple gaussienne. Utilise les colonnes 'win_width', 'width_guess', 
    'n_edge' si elles existent dans df_lines, sinon utilise default_params.
    """
    wave, flux = df_spectrum["Wavelength"].values, df_spectrum["Int"].values
    results = []
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    C_MICRON_S = 2.99792458e14

    for idx, row in df_lines.iterrows():
        l0 = row['wavelength']
        # Priorité aux colonnes du DF, sinon paramètre par défaut
        ww = row['win_width'] if 'win_width' in row else default_params[0]
        wg = row['width_guess'] if 'width_guess' in row else default_params[1]
        ne = int(row['n_edge']) if 'n_edge' in row else int(default_params[2])

        mask = (wave >= l0 - ww/2) & (wave <= l0 + ww/2)
        x_l, y_l = wave[mask], flux[mask]

        if len(x_l) < 5:
            results.append({'Line': row['Line'], 'success': False})
            continue

        sig_g = (l0 / R / fwhm_factor) * wg
        amp_g = np.max(y_l) - np.min(y_l)

        # Continuum Guess
        pts = max(1, len(x_l) // ne)
        slope_g = (np.mean(y_l[-pts:]) -
                   np.mean(y_l[:pts])) / (x_l[-1] - x_l[0])
        int_g = np.percentile(y_l, 20) - (slope_g * np.mean(x_l))

        # AJoiut pour forcer une raie d'émissio
        lower_bounds = [0, -np.inf, -np.inf, -np.inf, -np.inf]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]
        try:
            popt, pcov = curve_fit(
                model_gauss_linear,
                x_l, y_l,
                p0=[amp_g, l0, sig_g, slope_g, int_g],
                bounds=(lower_bounds, upper_bounds),  # Restriction ajoutée ici
                maxfev=5000
            )

            # popt, pcov = curve_fit(model_gauss_linear, x_l, y_l, p0=[amp_g, l0, sig_g, slope_g, int_g], maxfev=5000)
            amp_f, mu_f, sig_f = popt[0], popt[1], abs(popt[2])

            results.append({
                'Line': row['Line'], 'wavelength_th': l0, 'wavelength_obs': mu_f,
                'amplitude': amp_f, 'sigma': sig_f, 'continuum_m': popt[3], 'continuum_c': popt[4],
                'Velocity_km_s': 299792.458 * (mu_f - l0) / l0,
                'Int_cgs': amp_f * sig_f * np.sqrt(2 * np.pi) * (C_MICRON_S / (mu_f**2)) * 1e-17,
                'snr': amp_f / np.sqrt(np.diag(pcov))[0], 'success': True
            })
        except:
            results.append({'Line': row['Line'], 'success': False})

    return pd.DataFrame(results)

"""
def analyze_double_lines_flexible(df_spectrum, df_lines, default_params=[0.06, 1.0, 10], R=2700):
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    wave, flux = df_spectrum["Wavelength"].values, df_spectrum["Int"].values
    results = []
    C_cgs = 2.99792458e10

    def model_2g_lin(x, a1, m1, s1, a2, m2, s2, m, c):
        return a1*np.exp(-(x-m1)**2/(2*s1**2)) + a2*np.exp(-(x-m2)**2/(2*s2**2)) + m*x + c

    for _, row in df_lines.iterrows():
        l0, line_id = row['wavelength'], row['Line']
        ww = row.get('win_width', default_params[0])
        mask = (wave >= l0 - ww/2) & (wave <= l0 + ww/2)
        x_l, y_l = wave[mask], flux[mask]

        if len(x_l) < 12:
            results.append({'Line': line_id, 'success': False})
            continue

        sig_init = (l0 / R / 2.355) * row.get('width_guess', default_params[1])
        f_m, f_c = row.get('fixed_m'), row.get('fixed_c')
        is_fixed = pd.notna(f_m) and pd.notna(f_c)

        try:
            # 1. Estimation Continuum + Gaussienne Principale
            m_g = (y_l[-1] - y_l[0]) / (x_l[-1] -
                                        x_l[0]) if not is_fixed else f_m
            c_g = y_l[0] - m_g * x_l[0] if not is_fixed else f_c
            cont = m_g * x_l + c_g

            amp1 = np.max(y_l - cont)
            if amp1 < 0:
                amp1 = np.max(y_l) * 0.1  # Fallback si continuum surestimé

            # 2. Recherche intelligente du 2ème pic (Blend) dans les résidus
            g1_temp = amp1 * np.exp(-(x_l - l0)**2 / (2 * sig_init**2))
            residuals = y_l - (cont + g1_temp)
            # On cherche le pic positif restant
            idx_max_res = np.argmax(residuals)
            mu2_guess = x_l[idx_max_res]
            amp2_guess = residuals[idx_max_res]
            if amp2_guess < 0:
                amp2_guess = amp1 * 0.3  # Fallback

            # 3. Configuration du Fit Final
            p0 = [amp1, l0, sig_init, amp2_guess,
                  mu2_guess, sig_init, m_g, c_g]

            # Bornes: Amp > 0, Sigma > 0, Position dans la fenêtre
            b_low = [0, x_l[0], 1e-4, 0, x_l[0], 1e-4, -np.inf, -np.inf]
            b_high = [np.inf, x_l[-1], ww, np.inf, x_l[-1], ww, np.inf, np.inf]

            if is_fixed:
                p0 = p0[:6]  # On garde que les gaussiennes
                b_low, b_high = b_low[:6], b_high[:6]
                func = lambda x, *p: model_2g_lin(x, *p, f_m, f_c)
                popt_sub, pcov = curve_fit(
                    func, x_l, y_l, p0=p0, bounds=(b_low, b_high), maxfev=10000)
                popt = [*popt_sub, f_m, f_c]
                perr = np.sqrt(np.diag(pcov))
            else:
                popt, pcov = curve_fit(
                    model_2g_lin, x_l, y_l, p0=p0, bounds=(b_low, b_high), maxfev=10000)
                perr = np.sqrt(np.diag(pcov))

            # Identification Target (le plus proche de l0) vs Blend
            dist1, dist2 = abs(popt[1] - l0), abs(popt[4] - l0)
            idx_m, idx_b = (0, 3) if dist1 <= dist2 else (3, 0)
            # --- CALCUL SNR2 ---
            margin = max(3, int(0.15 * len(y_l)))
            y_continuum = np.concatenate((y_l[:margin], y_l[-margin:]))
            rms_cont = np.std(y_continuum)
            
            # SNR2 = amp_target / rms_cont
            snr2 = popt[idx_m] / rms_cont if rms_cont > 0 else 0
            
            results.append({
                'Line': line_id, 'wavelength_th': l0, 'wavelength_obs': popt[idx_m+1],
                'amplitude': popt[idx_m], 'sigma': popt[idx_m+2],
                'amp_blend': popt[idx_b], 'mu_blend': popt[idx_b+1], 'sig_blend': popt[idx_b+2],
                'continuum_m': popt[6], 'continuum_c': popt[7],
                'Velocity_km_s': 299792.458 * (popt[idx_m+1] - l0) / l0,
                'snr': popt[idx_m] / perr[idx_m] if len(perr) > idx_m and perr[idx_m] > 0 else 0,
                'snr2': snr2,
                'success': True
            })

        except Exception:
            results.append({'Line': line_id, 'success': False})

    return pd.DataFrame(results)



def analyze_double_lines_flexible_v2(df_spectrum, df_lines, default_params=[0.06, 1.0, 10], R=2700):
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    wave, flux = df_spectrum["Wavelength"].values, df_spectrum["Int"].values
    results = []
    C_cgs = 2.99792458e10

    def model_2g_lin(x, a1, m1, s1, a2, m2, s2, m, c):
        return a1*np.exp(-(x-m1)**2/(2*s1**2)) + a2*np.exp(-(x-m2)**2/(2*s2**2)) + m*x + c

    for _, row in df_lines.iterrows():
        l0, line_id = row['wavelength'], row['Line']
        ww = row.get('win_width', default_params[0])
        mask = (wave >= l0 - ww/2) & (wave <= l0 + ww/2)
        x_l, y_l = wave[mask], flux[mask]

        if len(x_l) < 12:
            results.append({'Line': line_id, 'success': False})
            continue

        sig_init = (l0 / R / 2.355) * row.get('width_guess', default_params[1])
        f_m, f_c = row.get('fixed_m'), row.get('fixed_c')
        is_fixed = pd.notna(f_m) and pd.notna(f_c)
        
        # NOUVEAU : On cherche si l'utilisateur a imposé une position pour le 2ème pic
        blend_wave_guess = row.get('blend_wave', np.nan)

        try:
            # 1. Estimation Continuum + Gaussienne Principale
            m_g = (y_l[-1] - y_l[0]) / (x_l[-1] - x_l[0]) if not is_fixed else f_m
            c_g = y_l[0] - m_g * x_l[0] if not is_fixed else f_c
            cont = m_g * x_l + c_g

            amp1 = np.max(y_l - cont)
            if amp1 < 0:
                amp1 = np.max(y_l) * 0.1  # Fallback si continuum surestimé

            # 2. Recherche intelligente du 2ème pic (Blend)
            g1_temp = amp1 * np.exp(-(x_l - l0)**2 / (2 * sig_init**2))
            residuals = y_l - (cont + g1_temp)
            
            # --- MODIFICATION ICI ---
            if pd.notna(blend_wave_guess):
                # Si l'utilisateur a donné une valeur, on l'utilise directement
                mu2_guess = blend_wave_guess
                # On regarde la hauteur des résidus à cet endroit précis pour aider l'algo
                idx_blend = np.argmin(np.abs(x_l - mu2_guess))
                amp2_guess = residuals[idx_blend]
                if amp2_guess < 0:
                    amp2_guess = amp1 * 0.3
            else:
                # Sinon, on cherche le pic positif restant comme avant
                idx_max_res = np.argmax(residuals)
                mu2_guess = x_l[idx_max_res]
                amp2_guess = residuals[idx_max_res]
                if amp2_guess < 0:
                    amp2_guess = amp1 * 0.3  # Fallback
            # ------------------------

            # 3. Configuration du Fit Final
            p0 = [amp1, l0, sig_init, amp2_guess, mu2_guess, sig_init, m_g, c_g]

            # Bornes: Amp > 0, Sigma > 0, Position dans la fenêtre
            b_low = [0, x_l[0], 1e-4, 0, x_l[0], 1e-4, -np.inf, -np.inf]
            b_high = [np.inf, x_l[-1], ww, np.inf, x_l[-1], ww, np.inf, np.inf]

            if is_fixed:
                p0 = p0[:6]  # On garde que les gaussiennes
                b_low, b_high = b_low[:6], b_high[:6]
                func = lambda x, *p: model_2g_lin(x, *p, f_m, f_c)
                popt_sub, pcov = curve_fit(
                    func, x_l, y_l, p0=p0, bounds=(b_low, b_high), maxfev=10000)
                popt = [*popt_sub, f_m, f_c]
                perr = np.sqrt(np.diag(pcov))
            else:
                popt, pcov = curve_fit(
                    model_2g_lin, x_l, y_l, p0=p0, bounds=(b_low, b_high), maxfev=10000)
                perr = np.sqrt(np.diag(pcov))

            # Identification Target (le plus proche de l0) vs Blend
            dist1, dist2 = abs(popt[1] - l0), abs(popt[4] - l0)
            idx_m, idx_b = (0, 3) if dist1 <= dist2 else (3, 0)
            
            # --- CALCUL SNR2 ---
            margin = max(3, int(0.15 * len(y_l)))
            y_continuum = np.concatenate((y_l[:margin], y_l[-margin:]))
            rms_cont = np.std(y_continuum)
            
            # SNR2 = amp_target / rms_cont
            snr2 = popt[idx_m] / rms_cont if rms_cont > 0 else 0
            
            results.append({
                'Line': line_id, 'wavelength_th': l0, 'wavelength_obs': popt[idx_m+1],
                'amplitude': popt[idx_m], 'sigma': popt[idx_m+2],
                'amp_blend': popt[idx_b], 'mu_blend': popt[idx_b+1], 'sig_blend': popt[idx_b+2],
                'continuum_m': popt[6], 'continuum_c': popt[7],
                'Velocity_km_s': 299792.458 * (popt[idx_m+1] - l0) / l0,
                'snr': popt[idx_m] / perr[idx_m] if len(perr) > idx_m and perr[idx_m] > 0 else 0,
                'snr2': snr2,
                'success': True
            })

        except Exception:
            results.append({'Line': line_id, 'success': False})

    return pd.DataFrame(results)


def analyze_double_lines_flexible_v3(df_spectrum, df_lines, default_params=[0.06, 1.0, 10], R=2700):
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    wave, flux = df_spectrum["Wavelength"].values, df_spectrum["Int"].values
    results = []

    for _, row in df_lines.iterrows():
        l0, line_id = row['wavelength'], row['Line']

        # ---------------------------------------------------------
        # 1. GESTION DE LA FENÊTRE D'EXPLORATION
        # ---------------------------------------------------------
        custom_win = row.get('custom_window', None)
        
        # Si une fenêtre personnalisée [min, max] est fournie
        if isinstance(custom_win, (list, tuple)) and len(custom_win) == 2:
            w_min, w_max = custom_win[0], custom_win[1]
            ww_eff = w_max - w_min
        # Sinon, comportement par défaut centré sur l0
        else:
            ww = row.get('win_width', default_params[0])
            w_min, w_max = l0 - ww/2, l0 + ww/2
            ww_eff = ww

        mask = (wave >= w_min) & (wave <= w_max)
        x_l, y_l = wave[mask], flux[mask]

        if len(x_l) < 12:
            results.append({'Line': line_id, 'success': False})
            continue

        sig_init = (l0 / R / 2.355) * row.get('width_guess', default_params[1])

        # ---------------------------------------------------------
        # 2. IDENTIFICATION DES PARAMÈTRES FIGÉS (GELÉS)
        # ---------------------------------------------------------
        f_m, f_c = row.get('fixed_m'), row.get('fixed_c')
        is_fixed_cont = pd.notna(f_m) and pd.notna(f_c)

        # NOUVEAU: On récupère la longueur d'onde EXACTE figée du blend
        f_m2 = row.get('fixed_blend_wave', np.nan)
        is_fixed_blend = pd.notna(f_m2)

        try:
            # Estimation Initiale du Continuum
            m_g = (y_l[-1] - y_l[0]) / (x_l[-1] - x_l[0]) if not is_fixed_cont else f_m
            c_g = y_l[0] - m_g * x_l[0] if not is_fixed_cont else f_c
            cont = m_g * x_l + c_g

            amp1 = np.max(y_l - cont)
            if amp1 < 0: amp1 = np.max(y_l) * 0.1

            # Estimation Initiale du 2ème pic (Blend)
            g1_temp = amp1 * np.exp(-(x_l - l0)**2 / (2 * sig_init**2))
            residuals = y_l - (cont + g1_temp)

            if is_fixed_blend:
                mu2_guess = f_m2
                idx_blend = np.argmin(np.abs(x_l - mu2_guess))
                amp2_guess = residuals[idx_blend]
            else:
                idx_max_res = np.argmax(residuals)
                mu2_guess = x_l[idx_max_res]
                amp2_guess = residuals[idx_max_res]

            if amp2_guess < 0: amp2_guess = amp1 * 0.3

            # ---------------------------------------------------------
            # 3. CONSTRUCTION DYNAMIQUE DES VARIABLES POUR LE FIT
            # ---------------------------------------------------------
            # On n'ajoute dans p0 (guesses) que les paramètres qui ont le droit de bouger
            p0 = [amp1, l0, sig_init, amp2_guess]
            b_low = [0, x_l[0], 1e-4, 0]
            b_high = [np.inf, x_l[-1], ww_eff, np.inf]

            if not is_fixed_blend:
                p0.append(mu2_guess)
                b_low.append(x_l[0])
                b_high.append(x_l[-1])

            p0.append(sig_init)
            b_low.append(1e-4)
            b_high.append(ww_eff)

            if not is_fixed_cont:
                p0.extend([m_g, c_g])
                b_low.extend([-np.inf, -np.inf])
                b_high.extend([np.inf, np.inf])

            # Le modèle s'adapte en direct selon ce qui est figé
            def model_dynamic(x, *args):
                idx = 0
                a1_d, m1_d, s1_d = args[idx], args[idx+1], args[idx+2]; idx += 3
                a2_d = args[idx]; idx += 1
                
                m2_d = f_m2 if is_fixed_blend else args[idx]
                if not is_fixed_blend: idx += 1
                
                s2_d = args[idx]; idx += 1

                m_d = f_m if is_fixed_cont else args[idx]
                if not is_fixed_cont: idx += 1
                c_d = f_c if is_fixed_cont else args[idx]

                return a1_d*np.exp(-(x-m1_d)**2/(2*s1_d**2)) + a2_d*np.exp(-(x-m2_d)**2/(2*s2_d**2)) + m_d*x + c_d

            # FIT !
            popt, pcov = curve_fit(model_dynamic, x_l, y_l, p0=p0, bounds=(b_low, b_high), maxfev=10000)
            perr_partial = np.sqrt(np.diag(pcov))

            # ---------------------------------------------------------
            # 4. RECONSTRUCTION DU TABLEAU DE RÉSULTATS (Les 8 valeurs)
            # ---------------------------------------------------------
            # On réintègre les valeurs figées dans les résultats pour la cohérence
            full_popt = np.zeros(8)
            full_perr = np.zeros(8)
            idx = 0

            full_popt[0:3] = popt[idx:idx+3]; full_perr[0:3] = perr_partial[idx:idx+3]; idx += 3
            full_popt[3] = popt[idx]; full_perr[3] = perr_partial[idx]; idx += 1

            if is_fixed_blend:
                full_popt[4] = f_m2; full_perr[4] = 0.0 # L'erreur est de 0 car la valeur est exacte
            else:
                full_popt[4] = popt[idx]; full_perr[4] = perr_partial[idx]; idx += 1

            full_popt[5] = popt[idx]; full_perr[5] = perr_partial[idx]; idx += 1

            if is_fixed_cont:
                full_popt[6], full_popt[7] = f_m, f_c
                full_perr[6], full_perr[7] = 0.0, 0.0
            else:
                full_popt[6], full_popt[7] = popt[idx], popt[idx+1]
                full_perr[6], full_perr[7] = perr_partial[idx], perr_partial[idx+1]

            # ---------------------------------------------------------
            # 5. REMPLISSAGE FINAL
            # ---------------------------------------------------------
            dist1, dist2 = abs(full_popt[1] - l0), abs(full_popt[4] - l0)
            idx_m, idx_b = (0, 3) if dist1 <= dist2 else (3, 0)

            margin = max(3, int(0.15 * len(y_l)))
            y_continuum = np.concatenate((y_l[:margin], y_l[-margin:]))
            rms_cont = np.std(y_continuum)
            
            snr2 = full_popt[idx_m] / rms_cont if rms_cont > 0 else 0
            snr = full_popt[idx_m] / full_perr[idx_m] if full_perr[idx_m] > 0 else 0

            results.append({
                'Line': line_id, 'wavelength_th': l0, 'wavelength_obs': full_popt[idx_m+1],
                'amplitude': full_popt[idx_m], 'sigma': full_popt[idx_m+2],
                'amp_blend': full_popt[idx_b], 'mu_blend': full_popt[idx_b+1], 'sig_blend': full_popt[idx_b+2],
                'continuum_m': full_popt[6], 'continuum_c': full_popt[7],
                'Velocity_km_s': 299792.458 * (full_popt[idx_m+1] - l0) / l0,
                'snr': snr,
                'snr2': snr2,
                'success': True
            })

        except Exception as e:
            results.append({'Line': line_id, 'success': False})

    return pd.DataFrame(results)



"""
def analyze_double_lines_flexible_v4(df_spectrum, df_lines, default_params=[0.06, 1.0, 10], R=2700):
    """UTILISEE ACTUELLEMENT MARS 2026"""
    import numpy as np
    import pandas as pd
    from scipy.optimize import curve_fit

    wave, flux = df_spectrum["Wavelength"].values, df_spectrum["Int"].values
    results = []

    for _, row in df_lines.iterrows():
        l0, line_id = row['wavelength'], row['Line']

        # ---------------------------------------------------------
        # 1. GESTION DE LA FENÊTRE D'EXPLORATION
        # ---------------------------------------------------------
        custom_win = row.get('custom_window', None)
        
        if isinstance(custom_win, (list, tuple)) and len(custom_win) == 2:
            w_min, w_max = custom_win[0], custom_win[1]
            ww_eff = w_max - w_min
        else:
            ww = row.get('win_width', default_params[0])
            w_min, w_max = l0 - ww/2, l0 + ww/2
            ww_eff = ww

        mask = (wave >= w_min) & (wave <= w_max)
        x_l, y_l = wave[mask], flux[mask]

        if len(x_l) < 12:
            results.append({'Line': line_id, 'success': False})
            continue

        sig_init = (l0 / R / 2.355) * row.get('width_guess', default_params[1])

        # ---------------------------------------------------------
        # 2. IDENTIFICATION DES PARAMÈTRES FIGÉS (GELÉS)
        # ---------------------------------------------------------
        f_m_in, f_c_in = row.get('fixed_m'), row.get('fixed_c')
        is_fixed_cont = pd.notna(f_m_in) and pd.notna(f_c_in)

        # --- MODIFICATION ICI : CONVERSION DU CONTINUUM ---
        if is_fixed_cont:
            f_m = f_m_in
            # Conversion de l'intercept central en intercept mathématique à x=0
            f_c = f_c_in - (f_m * l0)
        else:
            f_m, f_c = np.nan, np.nan

        # Longueur d'onde EXACTE figée du blend
        f_m2 = row.get('fixed_blend_wave', np.nan)
        is_fixed_blend = pd.notna(f_m2)

        try:
            # Estimation Initiale du Continuum (aligné sur fit_manual_lines2)
            if not is_fixed_cont:
                m_g = (y_l[-1] - y_l[0]) / (x_l[-1] - x_l[0])
                c_g = np.mean(y_l) - (m_g * np.mean(x_l))
            else:
                m_g, c_g = f_m, f_c
                
            cont = m_g * x_l + c_g

            amp1 = np.max(y_l - cont)
            if amp1 < 0: amp1 = np.max(y_l) * 0.1

            # Estimation Initiale du 2ème pic (Blend)
            g1_temp = amp1 * np.exp(-(x_l - l0)**2 / (2 * sig_init**2))
            residuals = y_l - (cont + g1_temp)

            if is_fixed_blend:
                mu2_guess = f_m2
                idx_blend = np.argmin(np.abs(x_l - mu2_guess))
                amp2_guess = residuals[idx_blend]
            else:
                idx_max_res = np.argmax(residuals)
                mu2_guess = x_l[idx_max_res]
                amp2_guess = residuals[idx_max_res]

            if amp2_guess < 0: amp2_guess = amp1 * 0.3

            # ---------------------------------------------------------
            # 3. CONSTRUCTION DYNAMIQUE DES VARIABLES POUR LE FIT
            # ---------------------------------------------------------
            p0 = [amp1, l0, sig_init, amp2_guess]
            b_low = [0, x_l[0], 1e-4, 0]
            b_high = [np.inf, x_l[-1], ww_eff, np.inf]

            if not is_fixed_blend:
                p0.append(mu2_guess)
                b_low.append(x_l[0])
                b_high.append(x_l[-1])

            p0.append(sig_init)
            b_low.append(1e-4)
            b_high.append(ww_eff)

            if not is_fixed_cont:
                p0.extend([m_g, c_g])
                b_low.extend([-np.inf, -np.inf])
                b_high.extend([np.inf, np.inf])

            # Le modèle s'adapte en direct selon ce qui est figé (équation y = mx + c stricte)
            def model_dynamic(x, *args):
                idx = 0
                a1_d, m1_d, s1_d = args[idx], args[idx+1], args[idx+2]; idx += 3
                a2_d = args[idx]; idx += 1
                
                m2_d = f_m2 if is_fixed_blend else args[idx]
                if not is_fixed_blend: idx += 1
                
                s2_d = args[idx]; idx += 1

                m_d = f_m if is_fixed_cont else args[idx]
                if not is_fixed_cont: idx += 1
                c_d = f_c if is_fixed_cont else args[idx]

                return a1_d*np.exp(-(x-m1_d)**2/(2*s1_d**2)) + a2_d*np.exp(-(x-m2_d)**2/(2*s2_d**2)) + m_d*x + c_d

            # FIT !
            popt, pcov = curve_fit(model_dynamic, x_l, y_l, p0=p0, bounds=(b_low, b_high), maxfev=10000)
            perr_partial = np.sqrt(np.diag(pcov))

            # ---------------------------------------------------------
            # 4. RECONSTRUCTION DU TABLEAU DE RÉSULTATS (Les 8 valeurs)
            # ---------------------------------------------------------
            full_popt = np.zeros(8)
            full_perr = np.zeros(8)
            idx = 0

            full_popt[0:3] = popt[idx:idx+3]; full_perr[0:3] = perr_partial[idx:idx+3]; idx += 3
            full_popt[3] = popt[idx]; full_perr[3] = perr_partial[idx]; idx += 1

            if is_fixed_blend:
                full_popt[4] = f_m2; full_perr[4] = 0.0 
            else:
                full_popt[4] = popt[idx]; full_perr[4] = perr_partial[idx]; idx += 1

            full_popt[5] = popt[idx]; full_perr[5] = perr_partial[idx]; idx += 1

            if is_fixed_cont:
                full_popt[6], full_popt[7] = f_m, f_c
                full_perr[6], full_perr[7] = 0.0, 0.0
            else:
                full_popt[6], full_popt[7] = popt[idx], popt[idx+1]
                full_perr[6], full_perr[7] = perr_partial[idx], perr_partial[idx+1]

            # ---------------------------------------------------------
            # 5. REMPLISSAGE FINAL
            # ---------------------------------------------------------
            dist1, dist2 = abs(full_popt[1] - l0), abs(full_popt[4] - l0)
            idx_m, idx_b = (0, 3) if dist1 <= dist2 else (3, 0)

            margin = max(3, int(0.15 * len(y_l)))
            y_continuum = np.concatenate((y_l[:margin], y_l[-margin:]))
            rms_cont = np.std(y_continuum)
            
            snr2 = full_popt[idx_m] / rms_cont if rms_cont > 0 else 0
            snr = full_popt[idx_m] / full_perr[idx_m] if full_perr[idx_m] > 0 else 0
            
            """
            line_flux = amp_fit * sigma_fit * np.sqrt(2 * np.pi)
            conv_factor = (C_MICRON_S / (mu_fit**2)) * 1e-17
            int_cgs = line_flux * conv_factor
            """
            
            C_MICRON_S = 2.99792458e14

            amp_main = full_popt[idx_m]
            mu_fit = full_popt[idx_m+1]
            sigma_main = full_popt[idx_m+2]
            conv_factor = (C_MICRON_S / (mu_fit**2)) * 1e-17

            line_flux = amp_main * sigma_main * np.sqrt(2 * np.pi)

            
            int_cgs = line_flux * conv_factor
            results.append({
                'Line': line_id, 'wavelength_th': l0, 'wavelength_obs': full_popt[idx_m+1],
                'amplitude': full_popt[idx_m], 'sigma': full_popt[idx_m+2],
                'amp_blend': full_popt[idx_b], 'mu_blend': full_popt[idx_b+1], 'sig_blend': full_popt[idx_b+2],
                'continuum_m': full_popt[6], 'continuum_c': full_popt[7], # c est maintenant bien le c mathématique
                'Velocity_km_s': 299792.458 * (full_popt[idx_m+1] - l0) / l0,
                'snr': snr,
                'snr2': snr2,
                'success': True,
                "Int_cgs":int_cgs
            })

        except Exception as e:
            results.append({'Line': line_id, 'success': False})

    return pd.DataFrame(results)

def analyze_double_lines_split_results(df_spectrum, df_lines, params_blend, R=2700):
    """
    Ajuste des doubles Gaussiennes en fixant le delta lambda théorique entre deux raies.
    Sépare les résultats sur deux lignes distinctes dans le DataFrame de sortie.
    Basé sur analyze_double_lines_flexible_v4 pour la robustesse du continuum.

    Format attendu pour params_blend:
    [(line_id_1, line_id_2, win_width, width_guess, fixed_m, fixed_c, custom_window), ...]
    """
    wave = df_spectrum["Wavelength"].values
    flux = df_spectrum["Int"].values
    results = []

    lines_dict = df_lines.set_index('Line').to_dict('index')

    for p in params_blend:
        line_1, line_2, win_width, width_guess, fixed_m_in, fixed_c_in, custom_win = p

        if line_1 not in lines_dict or line_2 not in lines_dict:
            err_msg = f"Raie {line_1} ou {line_2} absente de df_lines"
            results.append({'Line': line_1, 'blended_with': line_2, 'success': False, 'error': err_msg})
            results.append({'Line': line_2, 'blended_with': line_1, 'success': False, 'error': err_msg})
            continue

        lth_1 = lines_dict[line_1]['wavelength']
        lth_2 = lines_dict[line_2]['wavelength']
        
        # Le Delta Lambda figé
        fixed_delta_wave = lth_2 - lth_1

        # ---------------------------------------------------------
        # 1. GESTION DE LA FENÊTRE D'EXPLORATION
        # ---------------------------------------------------------
        if isinstance(custom_win, (list, tuple)) and len(custom_win) == 2:
            w_min, w_max = custom_win[0], custom_win[1]
            ww_eff = w_max - w_min
        else:
            w_min, w_max = lth_1 - win_width/2, lth_1 + win_width/2
            ww_eff = win_width

        mask = (wave >= w_min) & (wave <= w_max)
        x_l, y_l = wave[mask], flux[mask]

        if len(x_l) < 12:
            err_msg = 'Pas assez de points dans la fenêtre'
            results.append({'Line': line_1, 'blended_with': line_2, 'success': False, 'error': err_msg})
            results.append({'Line': line_2, 'blended_with': line_1, 'success': False, 'error': err_msg})
            continue

        sig_init = (lth_1 / R / 2.355) * width_guess

        # ---------------------------------------------------------
        # 2. IDENTIFICATION DES PARAMÈTRES FIGÉS (Continuum)
        # ---------------------------------------------------------
        is_fixed_cont = pd.notna(fixed_m_in) and pd.notna(fixed_c_in)

        if is_fixed_cont:
            f_m = float(fixed_m_in)
            # Conversion de l'intercept central en intercept mathématique à x=0 (comme dans ta v4)
            f_c = float(fixed_c_in) - (f_m * lth_1)
        else:
            f_m, f_c = np.nan, np.nan

        try:
            # Estimation Initiale du Continuum
            if not is_fixed_cont:
                m_g = (y_l[-1] - y_l[0]) / (x_l[-1] - x_l[0])
                c_g = np.mean(y_l) - (m_g * np.mean(x_l))
            else:
                m_g, c_g = f_m, f_c
                
            cont = m_g * x_l + c_g

            # Estimation Initiale des pics
            amp1 = np.max(y_l - cont)
            if amp1 < 0: amp1 = np.max(y_l) * 0.1

            g1_temp = amp1 * np.exp(-(x_l - lth_1)**2 / (2 * sig_init**2))
            residuals = y_l - (cont + g1_temp)

            mu2_guess = lth_1 + fixed_delta_wave
            idx_blend = np.argmin(np.abs(x_l - mu2_guess))
            amp2_guess = residuals[idx_blend]

            if amp2_guess < 0: amp2_guess = amp1 * 0.3

            # ---------------------------------------------------------
            # 3. CONSTRUCTION DYNAMIQUE DES VARIABLES POUR LE FIT
            # ---------------------------------------------------------
            p0 = [amp1, lth_1, sig_init, amp2_guess, sig_init]
            b_low = [0, x_l[0], 1e-4, 0, 1e-4]
            b_high = [np.inf, x_l[-1], ww_eff, np.inf, ww_eff]

            if not is_fixed_cont:
                p0.extend([m_g, c_g])
                b_low.extend([-np.inf, -np.inf])
                b_high.extend([np.inf, np.inf])

            # Le modèle s'adapte en direct selon ce qui est figé
            def model_dynamic(x, *args):
                a1_d, m1_d, s1_d, a2_d, s2_d = args[0:5]
                m2_d = m1_d + fixed_delta_wave # NOUVEAUTÉ : Delta figé
                
                idx = 5
                m_d = f_m if is_fixed_cont else args[idx]
                if not is_fixed_cont: idx += 1
                c_d = f_c if is_fixed_cont else args[idx]

                return a1_d*np.exp(-(x-m1_d)**2/(2*s1_d**2)) + a2_d*np.exp(-(x-m2_d)**2/(2*s2_d**2)) + m_d*x + c_d

            # FIT !
            popt, pcov = curve_fit(model_dynamic, x_l, y_l, p0=p0, bounds=(b_low, b_high), maxfev=10000)
            perr_partial = np.sqrt(np.diag(pcov))

            # ---------------------------------------------------------
            # 4. RECONSTRUCTION ET ASSIGNATION
            # ---------------------------------------------------------
            a1, m1, s1, a2, s2 = popt[0:5]
            ea1, em1, es1, ea2, es2 = perr_partial[0:5]
            m2 = m1 + fixed_delta_wave 
            
            if is_fixed_cont:
                m_fit, c_fit = f_m, f_c
            else:
                m_fit, c_fit = popt[5], popt[6]

            # ---------------------------------------------------------
            # 5. REMPLISSAGE FINAL SUR DEUX LIGNES
            # ---------------------------------------------------------
            margin = max(3, int(0.15 * len(y_l)))
            y_continuum = np.concatenate((y_l[:margin], y_l[-margin:]))
            rms_cont = np.std(y_continuum)
            C_MICRON_S = 2.99792458e14
            
            # --- Dictionnaire pour la Raie 1 ---
            snr_1 = a1 / ea1 if ea1 > 0 else 0
            snr2_1 = a1 / rms_cont if rms_cont > 0 else 0
            conv_1 = (C_MICRON_S / (m1**2)) * 1e-17
            line_flux_1 = a1 * s1 * np.sqrt(2 * np.pi)
            int_cgs_1 = line_flux_1 * conv_1
            
            row_1 = {
                'Line': line_1,
                'blended_with': line_2,
                'wavelength_th': lth_1,
                'wavelength_obs': m1,
                'amplitude': a1,
                'sigma': s1,
                'Int_cgs': int_cgs_1,
                'Velocity_km_s': 299792.458 * (m1 - lth_1) / lth_1,
                'snr': snr_1,
                'snr2': snr2_1,
                'continuum_m': m_fit,
                'continuum_c': c_fit, # c est bien le c mathématique à x=0
                'success': True
            }

            # --- Dictionnaire pour la Raie 2 ---
            snr_2 = a2 / ea2 if ea2 > 0 else 0
            snr2_2 = a2 / rms_cont if rms_cont > 0 else 0
            conv_2 = (C_MICRON_S / (m2**2)) * 1e-17
            line_flux_2 = a2 * s2 * np.sqrt(2 * np.pi)
            int_cgs_2 = line_flux_2 * conv_2

            row_2 = {
                'Line': line_2,
                'blended_with': line_1,
                'wavelength_th': lth_2,
                'wavelength_obs': m2,
                'amplitude': a2,
                'sigma': s2,
                'Int_cgs': int_cgs_2,
                'Velocity_km_s': 299792.458 * (m2 - lth_2) / lth_2,
                'snr': snr_2,
                'snr2': snr2_2,
                'continuum_m': m_fit,
                'continuum_c': c_fit, # c est bien le c mathématique à x=0
                'success': True
            }

            results.extend([row_1, row_2])

        except Exception as e:
            err_msg = str(e)
            results.append({'Line': line_1, 'blended_with': line_2, 'success': False, 'error': err_msg})
            results.append({'Line': line_2, 'blended_with': line_1, 'success': False, 'error': err_msg})

    return pd.DataFrame(results)

def estimate_local_noise(df_spectrum, col_wave="Wavelength", col_flux="Int", wave_min=None, wave_max=None):
    """
    Isole une section de bruit pur, soustrait le continuum linéaire local, 
    et calcule l'écart-type (RMS) des résidus.
    """
    # 1. Masquage et extraction de la section de bruit
    mask = (df_spectrum[col_wave] >= wave_min) & (df_spectrum[col_wave] <= wave_max)
    df_noise = df_spectrum.loc[mask].copy()
    
    if len(df_noise) < 3:
        raise ValueError("La section spécifiée contient trop peu de points pour estimer le bruit de manière fiable.")
        
    x = df_noise[col_wave].values
    y = df_noise[col_flux].values
    
    # 2. Ajustement d'un continuum linéaire local (détrending)
    # On utilise un polynôme de degré 1 (y = mx + c)
    coefficients = np.polyfit(x, y, 1)
    slope, intercept = coefficients[0], coefficients[1]
    
    continuum_local = slope * x + intercept
    
    # 3. Calcul des résidus (bruit pur)
    residuals = y - continuum_local
    
    # 4. Calcul statistique du bruit (RMS) avec ddof=1 pour l'estimateur sans biais
    noise_rms = np.std(residuals, ddof=1)
    mean_background = np.mean(y)
    
    # Ajout au dataframe pour contrôle visuel éventuel
    df_noise['Continuum_Fit'] = continuum_local
    df_noise['Residuals'] = residuals
    
    return noise_rms, mean_background, df_noise






import pandas as pd

def create_unresolved_blends(df_fits, df_lines, unresolved_pairs):
    """
    Associe deux raies non résolues sous un même fit (le meilleur des deux disponibles).
    Génère deux lignes partageant le profil gaussien observé, mais avec
    des vitesses radiales calculées sur leurs longueurs d'onde théoriques respectives.
    
    unresolved_pairs : liste de tuples [("Raie1", "Raie2"), ...]
    """
    results = []
    lines_dict = df_lines.set_index('Line').to_dict('index')
    C_KMS = 299792.458

    for pair in unresolved_pairs:
        line1, line2 = pair

        if line1 not in lines_dict or line2 not in lines_dict:
            continue

        # 1. Trouver les fits existants dans le tableau fourni
        fit1 = df_fits[(df_fits['Line'] == line1) & (df_fits['success'] == True)]
        fit2 = df_fits[(df_fits['Line'] == line2) & (df_fits['success'] == True)]

        if fit1.empty and fit2.empty:
            continue # Aucun fit réussi pour aucune des deux raies

        # 2. Choisir le meilleur fit (basé sur le SNR)
        if not fit1.empty and not fit2.empty:
            snr1 = fit1.iloc[0].get('snr', 0)
            snr2 = fit2.iloc[0].get('snr', 0)
            best_fit = fit1.iloc[0].copy() if snr1 >= snr2 else fit2.iloc[0].copy()
        elif not fit1.empty:
            best_fit = fit1.iloc[0].copy()
        else:
            best_fit = fit2.iloc[0].copy()

        # 3. Récupérer le pic observé partagé
        mu_obs = best_fit['wavelength_obs']
        lth1 = lines_dict[line1]['wavelength']
        lth2 = lines_dict[line2]['wavelength']

        # 4. Créer la ligne pour la Raie 1
        row1 = best_fit.copy()
        row1['Line'] = line1
        row1['blended_with'] = line2
        row1['fit_type'] = 'UNRESOLVED'
        row1['wavelength_th'] = lth1
        row1['Velocity_km_s'] = C_KMS * (mu_obs - lth1) / lth1

        # 5. Créer la ligne pour la Raie 2
        row2 = best_fit.copy()
        row2['Line'] = line2
        row2['blended_with'] = line1
        row2['fit_type'] = 'UNRESOLVED'
        row2['wavelength_th'] = lth2
        row2['Velocity_km_s'] = C_KMS * (mu_obs - lth2) / lth2

        results.extend([row1, row2])

    return pd.DataFrame(results)




























#######################################GRAPHIQUE ############
#######################################GRAPHIQUE ############
#######################################GRAPHIQUE ############
#######################################GRAPHIQUE ############

def plot_spectrum_basics (df_spectrum,
                          n_plots = 3,
                          wl_min = 4.57,wl_max = 4.964,
                          ylim_min= 80, ylim_max = 200,label=""):
    viz.Setting_Figure("Excitation")

    delta_wl = (wl_max - wl_min) / n_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(25, 6 * n_plots))
    for i, ax in enumerate(axes):
        current_min = wl_min + i * delta_wl
        current_max = wl_min + (i + 1) * delta_wl
        
        ax.step(df_spectrum["Wavelength"], df_spectrum["Int"], "-", c="black", alpha=0.9,where="mid",linewidth=2,label=label)
    

        ax.set_xlim([current_min, current_max])
        ax.set_ylim([ylim_min, ylim_max])
        ax.set_ylabel("Surface brightness [MJy/sr]", size=20)
        ax.grid(linestyle="--", alpha=0.5)
        ax.text(0.01, 0.1, f"Section {i+1}: {current_min:.3f} - {current_max:.3f} $\mu$m", 
                transform=ax.transAxes, fontsize=15, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    handles = []
    axes[0].legend(handles=handles, 
                   loc='upper right', 
                   title="Paramètres d'ajustement",
                   fontsize=15, 
                   framealpha=0.9,
                   facecolor='white')
    axes[-1].set_xlabel("Wavelength [microns]", size=20)

    fig.tight_layout()
    return fig,axes
    
def plot_spectral_fit(df_spectrum, df_results, df_selection):
    """
    Trace le spectre, les ajustements (Gaussienne + Continuum) et ajoute les annotations via viz.
    Arguments:
        df_spectrum : Données (Wavelength, Int)
        df_results  : Résultat du fit des gaussiennes
        df_selection: DataFrame des raies pour l'annotation
        viz_module  : Le module externe contenant la fonction 'put_annotation'
    """
    # --- Paramètres ---
    n_plots = 3
    wl_min = 4.57
    wl_max = 4.964

    # Largeur de tracé autour du pic
    plot_width = 0.01

    delta_wl = (wl_max - wl_min) / n_plots
    fig, axes = plt.subplots(n_plots, 1, figsize=(25, 6 * n_plots))

    if n_plots == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # 1. Bornes
        current_min = wl_min + i * delta_wl
        current_max = wl_min + (i + 1) * delta_wl

        # 2. Données brutes
        ax.plot(df_spectrum["Wavelength"], df_spectrum["Int"],
                "-", c="black", alpha=0.6, lw=1)

        # 3. Filtrage et Tracé des Fits
        mask_fits = (df_results['success'] == True) & \
                    (df_results['wavelength_obs'] >= current_min) & \
                    (df_results['wavelength_obs'] <= current_max)

        for _, row in df_results[mask_fits].iterrows():
            mu = row['wavelength_obs']
            # x_plot = np.linspace(mu - plot_width, mu + plot_width, 50)
            x_plot = np.linspace(
                row['wavelength_obs'] - 0.002, row['wavelength_obs'] + 0.002, 100)
            continuum = row['continuum_m'] * x_plot + row['continuum_c']
            total_fit = (row['amplitude'] * np.exp(-(x_plot - mu)
                         ** 2 / (2 * row['sigma']**2))) + continuum

            # Continuum (vert pointillé transparent)
            ax.plot(x_plot, continuum, color='green',
                    linestyle='--', alpha=0.5, linewidth=1.5)
            # Fit Global (vert plein transparent)
            ax.plot(x_plot, total_fit, color='darkred',
                    linestyle='-', alpha=0.7, linewidth=2)

        viz.put_annotation(ax, df_selection, color="purple",
                           BaseName=r"CO ", alpha_text=0.8)

        # 5. Mise en forme
        ax.set_xlim([current_min, current_max])
        ax.set_ylim([80, 300])
        mask_view = (df_spectrum["Wavelength"] >= current_min) & (
            df_spectrum["Wavelength"] <= current_max)

        ax.set_ylabel("Surface brightness [MJy/sr]", size=16)
        ax.set_xlabel("Wavelength [microns]", size=16)
        ax.grid(linestyle="--", alpha=0.3)

        ax.text(0.01, 0.9, f"Section {i+1}: {current_min:.3f} - {current_max:.3f} $\mu$m",
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    fig.tight_layout()
    plt.show()
    return fig, axes


def plt_spectral1Line(df_spectrum, df_results, df_selection, nligne=1):
    """
    Trace le spectre, les ajustements (Gaussienne + Continuum) et ajoute les annotations via viz.
    Arguments:
        df_spectrum : Données (Wavelength, Int)
        df_results  : Résultat du fit des gaussiennes
        df_selection: DataFrame des raies pour l'annotation
        viz_module  : Le module externe contenant la fonction 'put_annotation'
    """
    # --- Paramètres ---
    n_plots = 1
    wl_min = 4.57
    wl_max = 4.964

    delta_wl = (wl_max - wl_min) / 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(25, 6 * n_plots))

    if n_plots == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # 1. Bornes
        current_min = wl_min + nligne * delta_wl
        current_max = wl_min + (nligne + 1) * delta_wl

        # 2. Données brutes
        ax.plot(df_spectrum["Wavelength"], df_spectrum["Int"],
                "-", c="black", alpha=0.6, lw=1)

        # 3. Filtrage et Tracé des Fits
        mask_fits = (df_results['success'] == True) & \
                    (df_results['wavelength_obs'] >= current_min) & \
                    (df_results['wavelength_obs'] <= current_max)

        for _, row in df_results[mask_fits].iterrows():
            mu = row['wavelength_obs']
            # x_plot = np.linspace(mu - plot_width, mu + plot_width, 50)
            x_plot = np.linspace(
                row['wavelength_obs'] - 0.002, row['wavelength_obs'] + 0.002, 100)
            continuum = row['continuum_m'] * x_plot + row['continuum_c']
            total_fit = (row['amplitude'] * np.exp(-(x_plot - mu)
                         ** 2 / (2 * row['sigma']**2))) + continuum

            # Continuum (vert pointillé transparent)
            ax.plot(x_plot, continuum, color='green',
                    linestyle='--', alpha=0.5, linewidth=1.5)
            # Fit Global (vert plein transparent)
            ax.plot(x_plot, total_fit, color='darkred',
                    linestyle='-', alpha=0.7, linewidth=2)

        viz.put_annotation(ax, df_selection, color="purple",
                           BaseName=r"CO ", alpha_text=0.8)

        # 5. Mise en forme
        ax.set_xlim([current_min, current_max])
        ax.set_ylim([80, 300])
        mask_view = (df_spectrum["Wavelength"] >= current_min) & (
            df_spectrum["Wavelength"] <= current_max)

        ax.set_ylabel("Surface brightness [MJy/sr]", size=16)
        ax.set_xlabel("Wavelength [microns]", size=16)
        ax.grid(linestyle="--", alpha=0.3)

        ax.text(0.01, 0.9, f"Section {i+1}: {current_min:.3f} - {current_max:.3f} $\mu$m",
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    fig.tight_layout()
    plt.show()
    return fig, axes


def plot_all_levels(df_spectrum, processed_data):
    # Setup Figure
    try:
        viz.Setting_Figure("Excitation")
    except:
        pass  # Si viz n'est pas chargé

    n_plots = 3
    wl_min = 4.57
    wl_max = 4.964
    ylim_min, ylim_max = 80, 300
    delta_wl = (wl_max - wl_min) / n_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(25, 6 * n_plots))
    if n_plots == 1:
        axes = [axes]

    # --- Boucle sur les 3 panneaux (Wavelength Sections) ---
    for i, ax in enumerate(axes):
        current_min = wl_min + i * delta_wl
        current_max = wl_min + (i + 1) * delta_wl

        # 1. Tracé du spectre brut (Fond)
        ax.plot(df_spectrum["Wavelength"], df_spectrum["Int"],
                "-", c="black", alpha=1, lw=2, zorder=1)

        # 2. Boucle sur les NIVEAUX pour tracer Fits et Annotations
        for level in processed_data:
            df_res = level["df_res"]
            color = level["color"]
            df_theo = level["df_theo"]

            # A. Tracé des fits (Gaussiennes)
            mask_fits = (df_res['success'] == True) & \
                        (df_res['wavelength_obs'] >= current_min) & \
                        (df_res['wavelength_obs'] <= current_max)

            for _, row in df_res[mask_fits].iterrows():
                mu = row['wavelength_obs']
                # Zoom local pour le tracé du fit
                x_plot = np.linspace(mu - 0.003, mu + 0.003, 100)
                continuum = row['continuum_m'] * x_plot + row['continuum_c']
                total_fit = (row['amplitude'] * np.exp(-(x_plot - mu)
                             ** 2 / (2 * row['sigma']**2))) + continuum

                # Tracé du fit spécifique à ce niveau
                ax.plot(x_plot, continuum, color=color,
                        linestyle='--', alpha=0.4, linewidth=1)
                ax.plot(x_plot, total_fit, color=color, linestyle='-',
                        alpha=0.9, linewidth=2, zorder=5)

            # B. Annotations via VIZ (si disponible)
            try:
                # On filtre df_theo pour n'annoter que ce qui est dans la fenêtre actuelle
                # (viz gère souvent ça, mais c'est plus sûr)
                viz.put_annotation(
                    ax, df_theo, color=color, BaseName=f"{level['name']}: ", alpha=0.9, alpha_text=0.8)
            except NameError:
                pass  # viz pas défini

        # 3. Mise en forme du subplot
        ax.set_xlim([current_min, current_max])
        ax.set_ylim([ylim_min, ylim_max])
        ax.set_ylabel("Surface brightness [MJy/sr]", size=16)
        ax.set_xlabel("Wavelength [microns]", size=16)
        ax.grid(linestyle="--", alpha=0.3)

        # Label de section
        ax.text(0.01, 0.1, f"Section {i+1}: {current_min:.3f} - {current_max:.3f} $\mu$m",
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # --- 5. LÉGENDE PERSONNALISÉE (Sur le premier axe) ---
    handles = []
    for level in processed_data:
        name = level["name"]
        col = level["color"]
        p = level["params"]  # [Win, Guess, Edge]

        # Formatage LaTeX demandé
        label_text = (f"{name} : $Win_{{width}}$ {p[0]} $\mu$m  |  "
                      f"$Guess_{{width}}$ {p[1]} $\\times$ R")

        line = Line2D([0], [0], color=col, lw=2,
                      linestyle='-', label=label_text)
        handles.append(line)

    axes[0].legend(handles=handles,
                   loc='upper right',
                   title="Paramètres d'ajustement par Niveau",
                   fontsize=10,
                   framealpha=0.9,
                   facecolor='white',
                   edgecolor='gray')

    fig.tight_layout()
    plt.show()
    return (ax)


"""
def plot_final_spectral_summary(df_spectrum, df_fits, levels_config, df_line_co, df_else=None, wl_range=(4.57, 4.964), n_plots=3, ylim=(80, 350)):
    '''
    Génère le résumé graphique en utilisant levels_config pour les annotations colorées.
    '''
    wl_min, wl_max = wl_range
    ylim_min, ylim_max = ylim
    delta_wl = (wl_max - wl_min) / n_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(25, 8 * n_plots))
    if n_plots == 1: axes = [axes]

    for i, ax in enumerate(axes):
        current_min = wl_min + i * delta_wl
        current_max = wl_min + (i + 1) * delta_wl
        
        # 1. Tracé du spectre
        mask_view = (df_spectrum["Wavelength"] >= current_min) & (df_spectrum["Wavelength"] <= current_max)
        ax.plot(df_spectrum.loc[mask_view, "Wavelength"], df_spectrum.loc[mask_view, "Int"], 
                "-", c="black", alpha=0.6, lw=1.5, zorder=1)

        # 2. Filtrage des fits
        mask_fits = (df_fits['wavelength_obs'] >= current_min) & (df_fits['wavelength_obs'] <= current_max)
        subset_fits = df_fits[mask_fits]
        text_y_positions = [ylim_max * 0.6, ylim_max * 0.75, ylim_max * 0.9] 

        # 3. Boucle sur les fits pour tracés et annotations
        for idx, (_, row) in enumerate(subset_fits.iterrows()):
            mu, sigma, amp = row['wavelength_obs'], row['sigma'], row['amplitude']
            m, c, line_name = row['continuum_m'], row['continuum_c'], row['Line']
            
            x_plot = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            continuum = m * x_plot + c
            total_fit = (amp * np.exp(-(x_plot - mu)**2 / (2 * sigma**2))) + continuum
            
            ax.plot(x_plot, continuum, color='green', linestyle='--', alpha=0.6, lw=1)
            ax.plot(x_plot, total_fit, color='crimson', linestyle='-', alpha=0.9, lw=2, zorder=5)
            
            # --- ANNOTATIONS DYNAMIQUES VIA LEVELS_CONFIG ---
            for label, line_list, params, color in levels_config:
                # On crée le sous-df à la volée pour l'annotation
                df_temp = df_line_co[df_line_co['Line'].isin(line_list)]
                viz.put_annotation(ax, df_temp, color=color, BaseName=f"{label}: ", alpha=0.9, alpha_text=0)
            
            # Gestion des "Rejected" si df_else est fourni
            if df_else is not None:
                viz.put_annotation(ax, df_else, color="black", BaseName="Rejected: ", alpha=0.9, alpha_text=0)
            
            # Étiquette de la raie
            y_peak = amp + (m * mu + c)
            label_text = (f"{line_name}\n"
                          f"v: {row['Velocity_km_s']:.1f} km/s\n"
                          f"I: {row['Int_cgs']:.1e}\n"
                          f"d$\\lambda$: {row['Delta_Lambda']:.1e}\n"
                          f"$\\sigma$: {sigma:.1e}")
            
            ax.annotate(label_text, xy=(mu, y_peak), xytext=(mu, text_y_positions[idx % 3]),
                        arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.5),
                        fontsize=9, ha='center', va='bottom', zorder=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=0.8))

        # Mise en forme
        ax.set_xlim([current_min, current_max])
        ax.set_ylim([ylim_min, ylim_max])
        ax.set_ylabel("Brightness [MJy/sr]", size=16)
        axes[-1].set_xlabel("Wavelength [$\\mu$m]", size=16)
        ax.grid(linestyle="--", alpha=0.3)

    custom_lines = [Line2D([0], [0], color='black', lw=2, alpha=0.6),
                    Line2D([0], [0], color='crimson', lw=2),
                    Line2D([0], [0], color='green', lw=2, linestyle='--')]
    axes[0].legend(custom_lines, ['Spectre JWST', 'Fit Gaussien', 'Continuum'], 
                   loc='upper right', fontsize=12, framealpha=1)

    plt.tight_layout()
    plt.show()
    return fig, axes
"""
def plot_overrides_check(df_spectrum, df_fits, overrides_dict, 
                         window_width=0.015, n_cols=5):
    """
    Trace uniquement les raies présentes dans overrides_dict pour vérification.
    Style standard (sans alerte couleur).
    
    Args:
        window_width: Largeur de la fenêtre de zoom en microns (autour de la raie)
    """
    
    # 1. Filtrer le DataFrame pour ne garder que les raies "override"
    target_lines = list(overrides_dict.keys())
    subset_fits = df_fits[df_fits['Line'].isin(target_lines)].copy()
    
    if subset_fits.empty:
        print("Aucune des raies de l'override n'a été trouvée dans les résultats de fit.")
        return

    n_lines = len(subset_fits)
    # Création d'une grille dynamique
    n_rows = math.ceil(n_lines / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 5 * n_rows))
    axes = axes.flatten() # Aplatir pour itérer facilement
    
    for idx, (_, row) in enumerate(subset_fits.iterrows()):
        ax = axes[idx]
        
        # Récupération des paramètres
        line_name = row['Line']
        mu = row['wavelength_obs']
        sigma = row['sigma']
        amp = row['amplitude']
        m = row['continuum_m']
        c_const = row['continuum_c']
        velocity = row['Velocity_km_s']
        
        # Définition de la fenêtre de zoom locale
        current_min = mu - (window_width / 2)
        current_max = mu + (window_width / 2)
        
        # --- 1. Tracé du spectre (Données) ---
        mask_view = (df_spectrum["Wavelength"] >= current_min) & (df_spectrum["Wavelength"] <= current_max)
        df_view = df_spectrum.loc[mask_view]
        wave_arr = df_view["Wavelength"].values
        flux_arr = df_view["Int"].values
        
        # Trait noir pour le spectre
        ax.plot(wave_arr, flux_arr, "-", c="black", alpha=1, lw=1.5, zorder=1, label='Spectre')

        # --- 2. Recalcul du modèle ---
        x_plot = np.linspace(current_min, current_max, 200)
        continuum_plot = m * x_plot + c_const
        gauss_plot = amp * np.exp(-(x_plot - mu)**2 / (2 * sigma**2))
        total_fit_plot = gauss_plot + continuum_plot
        
        # --- 3. Calcul du SNR (pour info dans le texte uniquement) ---
        mask_noise = (
            ((wave_arr >= mu - 15*sigma) & (wave_arr <= mu - 3*sigma)) | 
            ((wave_arr >= mu + 3*sigma) & (wave_arr <= mu + 15*sigma))
        )
        
        snr = 0.0
        if mask_noise.sum() > 5:
            flux_noise = flux_arr[mask_noise]
            wave_noise = wave_arr[mask_noise]
            local_cont = m * wave_noise + c_const
            noise_residuals = flux_noise - local_cont
            noise_std = np.std(noise_residuals)
            snr = amp / noise_std if noise_std > 0 else 0.0

        # --- 4. Plot du Fit et Continuum ---
        # Continuum en pointillés bleus
        ax.plot(x_plot, continuum_plot, color='blue', linestyle='--', alpha=0.6, lw=1)
        # Fit global en vert solide
        ax.plot(x_plot, total_fit_plot, color='ForestGreen', linestyle='-', alpha=0.9, lw=2, zorder=5)

        # --- 5. Annotation ---
        label_text = (f"{line_name}\n"
                      f"v: {velocity:.1f} km/s\n"
                      f"I: {row['Int_cgs']:.1e}\n"
                      f"SNR: {snr:.1f}")
        
        y_peak = amp + (m * mu + c_const)
        y_annot = np.max(flux_arr) * 1.05 if len(flux_arr) > 0 else y_peak
        
        # Boîte blanche simple avec bordure noire
        ax.annotate(label_text, 
                    xy=(mu, y_peak), 
                    xytext=(mu, y_annot),
                    arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.5),
                    fontsize=10, ha='center', va='bottom', zorder=10, color="black",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9, lw=1))

        # Mise en forme graphique
        ax.set_title(f"{line_name}", fontsize=11, fontweight='bold', color='dimgray')
        ax.grid(linestyle="--", alpha=0.3)
        ax.set_xlim(current_min, current_max)
    
    # Masquer les axes vides s'il y en a
    for i in range(n_lines, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    
    
    

def plot_final_spectral_summary(df_spectrum, df_fits, levels_config, df_line_co, df_else=None,
                                wl_range=(4.576, 4.925), n_plots=2, ylim=(100, 200),
                                snr_threshold=3.0, v_min=-5.0, v_max=20.0):
    """
    Génère le résumé graphique avec calcul robuste du SNR et alertes couleurs ciblées.
    """
    wl_min, wl_max = wl_range
    ylim_min, ylim_max = ylim
    delta_wl = (wl_max - wl_min) / n_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(25, 8 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        current_min = wl_min + i * delta_wl
        current_max = wl_min + (i + 1) * delta_wl

        # 1. Tracé du spectre (Optimisation avec Numpy)
        mask_view = (df_spectrum["Wavelength"] >= current_min) & (
            df_spectrum["Wavelength"] <= current_max)
        df_view = df_spectrum.loc[mask_view]

        # Extraction des arrays pour rapidité
        wave_arr = df_view["Wavelength"].values
        flux_arr = df_view["Int"].values

        ax.plot(wave_arr, flux_arr, "-", c="black", alpha=1, lw=1.5, zorder=1)

        # 2. Filtrage des fits
        mask_fits = (df_fits['wavelength_obs'] >= current_min) & (
            df_fits['wavelength_obs'] <= current_max)
        subset_fits = df_fits[mask_fits]
        text_y_positions = [ylim_max * 0.6, ylim_max * 0.75, ylim_max * 0.9]

        # 3. Boucle sur les fits
        for idx, (_, row) in enumerate(subset_fits.iterrows()):
            mu, sigma, amp = row['wavelength_obs'], row['sigma'], row['amplitude']
            m, c, line_name = row['continuum_m'], row['continuum_c'], row['Line']
            velocity = row['Velocity_km_s']

            # --- CALCUL DU MODÈLE ET DU SNR ---
            x_plot = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            continuum_plot = m * x_plot + c
            gauss_plot = amp * np.exp(-(x_plot - mu)**2 / (2 * sigma**2))
            total_fit_plot = gauss_plot + continuum_plot

            # Calcul du SNR Robuste (Sidebands)
            # On cherche le bruit entre 3 sigma et 15 sigma (loin du coeur de la raie)
            mask_noise = (
                ((wave_arr >= mu - 15*sigma) & (wave_arr <= mu - 3*sigma)) |
                ((wave_arr >= mu + 3*sigma) & (wave_arr <= mu + 15*sigma))
            )

            if mask_noise.sum() > 5:
                flux_noise = flux_arr[mask_noise]
                wave_noise = wave_arr[mask_noise]

                # Soustraction du continuum local pour avoir le vrai bruit RMS
                local_cont = m * wave_noise + c
                noise_residuals = flux_noise - local_cont
                noise_std = np.std(noise_residuals)

                snr = amp / noise_std if noise_std > 0 else 0.0
            else:
                snr = 0.0

            # --- LOGIQUE D'ALERTE ---
            is_snr_bad = snr < snr_threshold
            is_vel_bad = (velocity < v_min) or (velocity > v_max)

            # Couleur de la courbe gaussienne
            if is_snr_bad or is_vel_bad:
                fit_color = 'red'
                box_edge_color = 'red'
                box_lw = 2
            else:
                fit_color = 'ForestGreen'  # Vert si tout va bien
                box_edge_color = 'crimson'
                box_lw = 1

            # --- PLOT DU FIT ---
            ax.plot(x_plot, continuum_plot, color='blue',
                    linestyle='--', alpha=0.6, lw=1)
            ax.plot(x_plot, total_fit_plot, color=fit_color,
                    linestyle='-', alpha=0.9, lw=2, zorder=5)

            # --- ANNOTATIONS LIGNES (Fonction externe supposée) ---
            # for label, line_list, params, color in levels_config:
            #     df_temp = df_line_co[df_line_co['Line'].isin(line_list)]
            #     viz.put_annotation(ax, df_temp, color=color, BaseName=f"{label}: ", alpha=0.9, alpha_text=0)

            # --- CRÉATION DU LABEL TEXTE (MULTICOLORE) ---
            # Malheureusement, matplotlib ne supporte pas le multicolor dans un seul string simple.
            # On va utiliser une astuce : on écrit tout en noir, et on signale l'erreur globalement,
            # OU on affiche le texte critique en rouge si on accepte de séparer les strings.
            # Ici, pour rester simple et compatible : si alerte, le texte entier passe en rouge foncé,
            # ou on utilise des indicateurs.

            # Approche demandée : "Affiche que la ligne qui dépasse les critere en rouge"
            # Comme on ne peut pas facilement colorier un seul mot dans un `annotate` standard sans utiliser
            # des rendus complexes, on va colorier le texte en fonction du PROBLÈME dominant.

            annot_color = "black"

            # Construction du texte
            txt_name = f"{line_name}"

            # Gestion fine des couleurs par ligne (Hack visuel : on ne peut pas mixer les couleurs dans un seul annotate simple)
            # On va donc mettre des étoiles ou des marqueurs, ou colorier tout le bloc si c'est mauvais.
            # Pour respecter "seulement la phrase critère en rouge", il faudrait plusieurs annotate.
            # SIMPLIFICATION EFFICACE : On colore les valeurs dans le string si possible ? Non.
            # On va colorier tout le texte en rouge si c'est mauvais.

            if is_vel_bad or is_snr_bad:
                annot_color = "darkred"

            label_text = (f"{line_name}\n"
                          f"v: {velocity:.1f} km/s\n"
                          f"I: {row['Int_cgs']:.1e}\n"
                          f"SNR: {snr:.1f}")

            # Tracé de l'annotation
            y_peak = amp + (m * mu + c)
            ax.annotate(label_text,
                        xy=(mu, y_peak),
                        xytext=(mu, text_y_positions[idx % 3]),
                        arrowprops=dict(facecolor='black',
                                        arrowstyle='->', alpha=0.5),
                        fontsize=9,
                        ha='center', va='bottom', zorder=10,
                        color=annot_color,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=box_edge_color, alpha=0.9, lw=box_lw))

        # Mise en forme
        ax.set_xlim([current_min, current_max])
        ax.set_ylim([ylim_min, ylim_max])
        ax.set_ylabel("Brightness [MJy/sr]", size=16)
        axes[-1].set_xlabel("Wavelength [$\\mu$m]", size=16)
        ax.grid(linestyle="--", alpha=0.3)

    custom_lines = [Line2D([0], [0], color='black', lw=2, alpha=1),
                    Line2D([0], [0], color='ForestGreen', lw=2),
                    Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='blue', lw=2, linestyle='--')]

    axes[0].legend(custom_lines, ['Spectre JWST', 'Fit (OK)', 'Fit (Alert)', 'Continuum'],
                   loc='upper right', fontsize=12, framealpha=1)

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_final_spectral_summaryV2(df_spectrum, df_simple, df_blend, df_forced,
                                wl_range=(4.576, 4.925), n_plots=3, ylim=(100, 200),
                                snr_threshold=3.0, v_min=-5.0, v_max=20.0):
    """
    Génère un résumé spectral complet en distinguant visuellement les types de fits
    (Simple, Forcé, Blend) et en appliquant des alertes de qualité.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    wl_min, wl_max = wl_range
    ylim_min, ylim_max = ylim
    delta_wl = (wl_max - wl_min) / n_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(22, 6 * n_plots))
    if n_plots == 1:
        axes = [axes]

    # --- Préparation des données de fit globales (priorité Blend > Forced > Simple) ---
    # On crée un dictionnaire pour savoir quoi tracer pour chaque raie présente
    lines_to_plot = df_simple[df_simple['success'] == True]['Line'].unique()

    for i, ax in enumerate(axes):
        c_min = wl_min + i * delta_wl
        c_max = wl_min + (i + 1) * delta_wl

        # 1. Tracé du spectre
        mask_view = (df_spectrum["Wavelength"] >= c_min) & (
            df_spectrum["Wavelength"] <= c_max)
        view = df_spectrum[mask_view]
        ax.plot(view["Wavelength"], view["Int"], "-",
                c="black", lw=1.2, zorder=1, label="Spectre")

        # 2. Filtrage des raies dans cette fenêtre
        # On itère sur les noms de raies pour appliquer la logique de priorité
        text_pos_idx = 0
        y_levels = np.linspace(ylim_max * 0.65, ylim_max * 0.9, 3)

        for line_name in lines_to_plot:
            # Récupération de la "meilleure" donnée disponible pour cette raie
            row_data = None
            fit_type = "SIMPLE"

            # Priorité 1 : Blend
            if df_blend is not None and line_name in df_blend['Line'].values:
                row_data = df_blend[df_blend['Line'] == line_name].iloc[0]
                fit_type = "BLEND"
            # Priorité 2 : Forced
            elif df_forced is not None and line_name in df_forced['Line'].values:
                row_data = df_forced[df_forced['Line'] == line_name].iloc[0]
                fit_type = "FORCED"
            # Priorité 3 : Simple
            else:
                row_data = df_simple[df_simple['Line'] == line_name].iloc[0]
                fit_type = "SIMPLE"

            mu = row_data['wavelength_obs']
            if not (c_min <= mu <= c_max):
                continue

            # --- 3. CALCUL DU MODÈLE ---
            # Fenêtre locale pour le fit
            x_fit = np.linspace(mu - 0.005, mu + 0.005, 100)
            m, c_param = row_data['continuum_m'], row_data['continuum_c']
            cont_plot = m * x_fit + c_param

            # Logique d'alerte
            snr = row_data.get('snr', 0)
            vel = row_data.get('Velocity_km_s', 0)
            is_bad = (snr < snr_threshold) or (vel < v_min) or (vel > v_max)

            # Couleurs de base selon le type
            type_colors = {"BLEND": "blue",
                           "FORCED": "magenta", "SIMPLE": "crimson"}
            base_color = type_colors[fit_type]
            if is_bad:
                base_color = "red"  # Devient rouge vif si alerte

            if fit_type == "BLEND":
                # Double Gaussienne
                g1 = row_data['amplitude'] * \
                    np.exp(-(x_fit - mu)**2 / (2 * row_data['sigma']**2))
                g2 = row_data['amp_blend'] * np.exp(-(
                    x_fit - row_data['mu_blend'])**2 / (2 * row_data['sig_blend']**2))
                ax.plot(x_fit, g1 + g2 + cont_plot,
                        color=base_color, lw=2.5, zorder=5)
                ax.plot(x_fit, cont_plot, ":", color="gray", alpha=0.5)
            else:
                # Simple Gaussienne
                gauss = row_data['amplitude'] * \
                    np.exp(-(x_fit - mu)**2 / (2 * row_data['sigma']**2))
                ax.plot(x_fit, gauss + cont_plot,
                        color=base_color, lw=2, zorder=5)
                ax.plot(x_fit, cont_plot, ":", color=base_color, alpha=0.4)

            # --- 4. ANNOTATIONS ---
            annot_color = "darkred" if is_bad else "black"
            box_edge = "red" if is_bad else "gray"

            label_txt = f"{line_name} ({fit_type})\nv:{vel:.1f} | SNR:{snr:.1f}"
            y_peak = row_data['amplitude'] + (m * mu + c_param)

            ax.annotate(label_txt, xy=(mu, y_peak),
                        xytext=(mu, y_levels[text_pos_idx % 3]),
                        arrowprops=dict(arrowstyle='->',
                                        color=annot_color, alpha=0.3),
                        fontsize=8, ha='center', color=annot_color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=box_edge, alpha=0.8))
            text_pos_idx += 1

        # Esthétique
        ax.set_xlim(c_min, c_max)
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_ylabel("Brightness [MJy/sr]")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Wavelength [$\mu$m]")

    # Légende personnalisée
    legend_elements = [
        Line2D([0], [0], color='black', lw=1.5, label='Données'),
        Line2D([0], [0], color='crimson', lw=2, label='Simple'),
        Line2D([0], [0], color='magenta', lw=2, label='Forcé'),
        Line2D([0], [0], color='blue', lw=2, label='Blend'),
        Line2D([0], [0], color='red', lw=2, label='ALERTE (SNR/V)'),
    ]
    axes[0].legend(handles=legend_elements,
                   loc='upper right', ncol=3, fontsize=10)

    plt.tight_layout()
    return fig, axes

def plot_final_spectral_summaryV3(df_spectrum, df_fits, levels_config, df_line_co, df_else=None,
                                wl_range=(4.576, 4.925), n_plots=2, ylim=(100, 200),
                                snr_threshold=3.0, v_min=-5.0, v_max=20.0,
                                color_alert=True, plot_style='plot'):
    """
    Génère le résumé graphique V3.
    - color_alert (bool) : Active les alertes rouges (SNR/Vitesse). Si False, boîtes standard grises.
    - plot_style (str) : 'plot' (lignes continues) ou 'step' (histogramme style 'mid').
      Si 'step', le modèle est évalué exactement sur les longueurs d'onde du spectre.
    - Gère automatiquement les blends (Total = pointillés gras gris, Principale = Vert).
    """
    wl_min, wl_max = wl_range
    ylim_min, ylim_max = ylim
    delta_wl = (wl_max - wl_min) / n_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(25, 6 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        current_min = wl_min + i * delta_wl
        current_max = wl_min + (i + 1) * delta_wl

        # 1. Tracé du spectre (Optimisation avec Numpy)
        mask_view = (df_spectrum["Wavelength"] >= current_min) & (
            df_spectrum["Wavelength"] <= current_max)
        df_view = df_spectrum.loc[mask_view]

        wave_arr = df_view["Wavelength"].values
        flux_arr = df_view["Int"].values

        # Style du spectre
        if plot_style == 'step':
            ax.step(wave_arr, flux_arr, where='mid', c="black", alpha=1, lw=2, zorder=1)
        else:
            ax.plot(wave_arr, flux_arr, "-", c="black", alpha=1, lw=2, zorder=1)

        # 2. Filtrage des fits
        mask_fits = (df_fits['wavelength_obs'] >= current_min) & (
            df_fits['wavelength_obs'] <= current_max)
        subset_fits = df_fits[mask_fits]
        text_y_positions = [ylim_max * 0.6, ylim_max * 0.75, ylim_max * 0.9]

        # 3. Boucle sur les fits
        for idx, (_, row) in enumerate(subset_fits.iterrows()):
            mu, sigma, amp = row['wavelength_obs'], row['sigma'], row['amplitude']
            m, c, line_name = row['continuum_m'], row['continuum_c'], row['Line']
            velocity = row.get('Velocity_km_s', 0)
            
            # Détection s'il s'agit d'un blend
            is_blend = 'amp_blend' in row and not pd.isna(row.get('amp_blend', np.nan))

            # --- CALCUL DU MODÈLE SELON LE STYLE ---
            if plot_style == 'step':
                # Échantillonnage EXACT à la résolution du spectre
                mask_local = (wave_arr >= mu - 8*sigma) & (wave_arr <= mu + 8*sigma)
                if is_blend: # On élargit la fenêtre si c'est un blend
                    mask_local = (wave_arr >= mu - 8*sigma) & (wave_arr <= row['mu_blend'] + 8*row['sig_blend'])
                x_plot = wave_arr[mask_local]
            else:
                # Haute résolution pour un tracé lisse
                if is_blend:
                    min_w = min(mu - 8*sigma, row['mu_blend'] - 8*row['sig_blend'])
                    max_w = max(mu + 8*sigma, row['mu_blend'] + 8*row['sig_blend'])
                    x_plot = np.linspace(min_w, max_w, 200)
                else:
                    x_plot = np.linspace(mu - 8*sigma, mu + 8*sigma, 100)

            if len(x_plot) == 0:
                continue

            continuum_plot = m * x_plot + c
            g1_plot = amp * np.exp(-(x_plot - mu)**2 / (2 * sigma**2))
            
            # La composante principale (g1) + continuum
            main_comp_plot = g1_plot + continuum_plot
            
            if is_blend:
                g2_plot = row['amp_blend'] * np.exp(-(x_plot - row['mu_blend'])**2 / (2 * row['sig_blend']**2))
                total_fit_plot = g1_plot + g2_plot + continuum_plot
            else:
                total_fit_plot = main_comp_plot # Identique si pas de blend

            # --- CALCUL DU SNR ROBUSTE (Identique à V1) ---
            mask_noise = (
                ((wave_arr >= mu - 15*sigma) & (wave_arr <= mu - 3*sigma)) |
                ((wave_arr >= mu + 3*sigma) & (wave_arr <= mu + 15*sigma))
            )
            if mask_noise.sum() > 5:
                noise_residuals = flux_arr[mask_noise] - (m * wave_arr[mask_noise] + c)
                noise_std = np.std(noise_residuals)
                snr = amp / noise_std if noise_std > 0 else 0.0
            else:
                snr = 0.0

            # --- LOGIQUE D'ALERTE ET COULEURS ---
            is_snr_bad = snr < snr_threshold
            is_vel_bad = (velocity < v_min) or (velocity > v_max)
            is_bad = is_snr_bad or is_vel_bad

            if color_alert:
                fit_color = 'red' if is_bad else 'ForestGreen'
                annot_color = "darkred" if is_bad else "black"
                box_edge_color = "red" if is_bad else "crimson"
                box_lw = 2 if is_bad else 1
            else:
                fit_color = 'C1'
                annot_color = "black"
                box_edge_color = "gray"
                box_lw = 1

            # --- PLOT DU FIT ---
            if plot_style == 'step':
                ax.step(x_plot, continuum_plot, where='mid', color='blue', linestyle='--', alpha=0.6, lw=1.5)
                if is_blend:
                    # Somme totale en gris gras pointillé
                    ax.step(x_plot, total_fit_plot, where='mid', color='C0', linestyle='-.', lw=2, zorder=4)
                # Composante principale
                ax.step(x_plot, main_comp_plot, where='mid', color=fit_color,linestyle="--", lw=2.5, zorder=5)
            else:
                ax.plot(x_plot, continuum_plot, color='blue', linestyle='--', alpha=0.6, lw=1)
                if is_blend:
                    # Somme totale en gris gras pointillé
                    ax.plot(x_plot, total_fit_plot, color='C0', linestyle='-.', lw=2, zorder=4)
                # Composante principale
                ax.plot(x_plot, main_comp_plot, color=fit_color, lw=2, zorder=5)

            # --- CRÉATION DU LABEL TEXTE ---
            label_text = (f"{line_name}\n"
                          f"v: {velocity:.1f} km/s\n"
                          f"I: {row.get('Int_cgs', 0):.1e}\n"
                          f"SNR: {snr:.1f}")

            y_peak = amp + (m * mu + c)
            ax.annotate(label_text,
                        xy=(mu, y_peak),
                        xytext=(mu, text_y_positions[idx % 3]),
                        arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.5),
                        fontsize=9,
                        ha='center', va='bottom', zorder=10,
                        color=annot_color,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=box_edge_color, alpha=0.9, lw=box_lw))

        # Mise en forme
        ax.set_xlim([current_min, current_max])
        ax.set_ylim([ylim_min, ylim_max])
        ax.set_ylabel("Brightness [MJy/sr]", size=16)
        ax.grid(linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Wavelength [$\\mu$m]", size=16)

    # Création de la légende dynamique
    custom_lines = [
        Line2D([0], [0], color='black', lw=2, label='JWST extracted from latest cube'),
        Line2D([0], [0], color='C1', lw=2, label='Main componant (double gaussian fit)'),
        Line2D([0], [0], color='C0', lw=2.5, linestyle=':', label='Total of double guassian fit'),
        Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Continuum')
    ]
    if color_alert:
        custom_lines.insert(2, Line2D([0], [0], color='red', lw=2, label='Fit (Alerte SNR/V)'))

    axes[0].legend(handles=custom_lines, loc='upper right', fontsize=12, framealpha=1)

    plt.tight_layout()
    plt.show()
    return fig, axes



def Analyse_Veocity(df_final, n_sigma=2.5):
    df_analyse = df_final.copy()
    v = df_analyse['Velocity_km_s']
    mean_v = np.mean(v)
    median_v = np.median(v)
    std_v = np.std(v)

    print(f"--- Statistiques Globales ---")
    print(f"Nombre de raies : {len(v)}")
    print(f"Moyenne (Mean)  : {mean_v:.2f} km/s")
    print(f"Médiane (Med)   : {median_v:.2f} km/s")
    print(f"Ecart-type (Std): {std_v:.2f} km/s")

    # On considère comme outlier tout ce qui dépasse Mean +/- (n_sigma * Std)
    seuil_bas = mean_v - (n_sigma * std_v)
    seuil_haut = mean_v + (n_sigma * std_v)
    print(f"\nCritère de rejet : > {n_sigma} sigma")
    print(f"Intervalle accepté : [{seuil_bas:.2f}, {seuil_haut:.2f}] km/s")

    mask_outlier = (v < seuil_bas) | (v > seuil_haut)
    df_outliers = df_analyse[mask_outlier].copy()
    df_valid = df_analyse[~mask_outlier].copy()
    print(f"Nombre d'outliers détectés : {len(df_outliers)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
        18, 6), gridspec_kw={'width_ratios': [2, 1]})
    ax1.scatter(df_valid['wavelength_th'], df_valid['Velocity_km_s'],
                c='royalblue', alpha=0.7, label='Valid Lines', s=50, edgecolors='k')

    ax1.scatter(df_outliers['wavelength_th'], df_outliers['Velocity_km_s'],
                c='crimson', label='Outliers', s=80, marker='X', edgecolors='white')

    ax1.axhline(mean_v, color='green', linestyle='-',
                linewidth=2, label=f'Mean: {mean_v:.1f}')
    ax1.axhline(median_v, color='orange', linestyle='--',
                linewidth=2, label=f'Median: {median_v:.1f}')
    ax1.fill_between([df_analyse['wavelength_th'].min(), df_analyse['wavelength_th'].max()],
                     seuil_bas, seuil_haut, color='green', alpha=0.1, label=f'Interval $\pm{n_sigma}\sigma$')

    for _, row in df_outliers.iterrows():
        ax1.text(row['wavelength_th'], row['Velocity_km_s'], f"  {row['Line']}",
                 fontsize=9, color='darkred', verticalalignment='center')

    ax1.set_xlabel("Wavelength Theoretical [$\mu$m]", fontsize=14)
    ax1.set_ylabel("Velocity $v_{LSR}$ [km/s]", fontsize=14)
    ax1.set_title("Distribution des Vitesses par Longueur d'onde", fontsize=16)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.hist(df_valid['Velocity_km_s'], bins=10,
             color='royalblue', alpha=0.6, label='Valid')
    ax2.hist(df_outliers['Velocity_km_s'], bins=5,
             color='crimson', alpha=0.6, label='Outliers')
    ax2.axvline(mean_v, color='green', linestyle='-', linewidth=2)
    ax2.axvline(median_v, color='orange', linestyle='--')
    ax2.axvline(seuil_bas, color='red', linestyle=':')
    ax2.axvline(seuil_haut, color='red', linestyle=':')
    ax2.set_xlabel("Velocity [km/s]", fontsize=14)
    ax2.set_ylabel("Count", fontsize=14)
    ax2.set_title("Histogramme des Vitesses", fontsize=16)
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_gallery_fits(df_spectrum, df_fits, n_cols=4, window_view=0.1, color_mode='coolwarm'):
    """
    Trace une galerie de plots individuels.
    color_mode : peut être un nom de colormap (str) ou une couleur fixe (str).
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    df_to_plot = df_fits[df_fits['success'] == True].copy()
    if df_to_plot.empty:
        return None, None

    n_plots = len(df_to_plot)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    axes = axes.flatten()

    # --- Gestion de la couleur ---
    is_cmap = False
    try:
        cmap = cm.get_cmap(color_mode)
        norm = mcolors.Normalize(vmin=0, vmax=30)
        is_cmap = True
    except:
        # Si color_mode n'est pas une cmap, on l'utilise comme couleur fixe
        fixed_color = color_mode

    for i, (idx, row) in enumerate(df_to_plot.iterrows()):
        ax = axes[i]
        mu_obs = row['wavelength_obs']
        mu_th = row['wavelength_th']

        mask = (df_spectrum["Wavelength"] >= mu_th - window_view/2) & \
               (df_spectrum["Wavelength"] <= mu_th + window_view/2)
        wave_local = df_spectrum.loc[mask, "Wavelength"]
        flux_local = df_spectrum.loc[mask, "Int"]

        ax.plot(wave_local, flux_local, c='black', lw=1, alpha=0.7)
        ax.axvline(mu_th, color='gray', linestyle='--', alpha=0.5, lw=1)

        # Choix de la couleur pour ce subplot
        line_c = cmap(norm(abs(row['Velocity_km_s']))
                      ) if is_cmap else fixed_color

        x_plot = np.linspace(wave_local.min(), wave_local.max(), 100)
        continuum = row['continuum_m'] * x_plot + row['continuum_c']
        gaussian = row['amplitude'] * \
            np.exp(-(x_plot - mu_obs)**2 / (2 * row['sigma']**2))

        ax.plot(x_plot, continuum, linestyle=':', color=line_c, lw=2)
        ax.plot(x_plot, gaussian + continuum,
                linestyle='-', color=line_c, lw=2.5)

        ax.set_title(f"{row['Line']}", fontsize=12, fontweight='bold')
        info_txt = f"v: {row['Velocity_km_s']:.1f}\nSNR: {row['snr']:.1f}"
        ax.text(0.05, 0.95, info_txt, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_gallery_double_fits(df_spectrum, df_fits, n_cols=4, window_view=0.1):
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    df_to_plot = df_fits[df_fits['success'] == True].copy()
    if df_to_plot.empty:
        return None, None

    n_plots = len(df_to_plot)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for i, (idx, row) in enumerate(df_to_plot.iterrows()):
        ax = axes[i]
        mu_th = row['wavelength_th']

        mask = (df_spectrum["Wavelength"] >= mu_th - window_view /
                2) & (df_spectrum["Wavelength"] <= mu_th + window_view/2)
        wave_local, flux_local = df_spectrum.loc[mask,
                                                 "Wavelength"], df_spectrum.loc[mask, "Int"]

        ax.plot(wave_local, flux_local, c='black',
                lw=2, alpha=0.9, label='Obs')
        ax.axvline(mu_th, color='gray', linestyle=':', alpha=0.6, lw=1)

        x_plot = np.linspace(wave_local.min(), wave_local.max(), 200)
        cont = row['continuum_m'] * x_plot + row['continuum_c']

        g1 = row['amplitude'] * \
            np.exp(-(x_plot - row['wavelength_obs'])
                   ** 2 / (2 * row['sigma']**2))
        g2 = row['amp_blend'] * \
            np.exp(-(x_plot - row['mu_blend'])**2 / (2 * row['sig_blend']**2))

        dist1 = abs(row['wavelength_obs'] - mu_th)
        dist2 = abs(row['mu_blend'] - mu_th)
        g_main, g_sec = (g1, g2) if dist1 <= dist2 else (g2, g1)

        ax.plot(x_plot, cont, linestyle=':', color='gray', alpha=0.5)
        ax.plot(x_plot, g_main + cont, linestyle='-',
                color='green', lw=3, alpha=0.8, label='Main')
        ax.plot(x_plot, g_sec + cont, linestyle=':',
                color='orange', lw=1.5, alpha=0.9, label='Blend')
        ax.plot(x_plot, g1 + g2 + cont, linestyle='-.',
                color='blue', lw=2.5, alpha=0.7, label='Total')

        ax.set_title(f"{row['Line']}", fontsize=10)
        ax.text(0.05, 0.95, f"v: {row['Velocity_km_s']:.1f}\nSNR: {row['snr']:.1f}",
                transform=ax.transAxes, fontsize=8, va='top', bbox=dict(boxstyle="round", fc="white", alpha=0.6))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig, axes



def plot_double_gaussian_gallery_delta(df_spectrum, df_results, plot_style='step', cols=3):
    """
    Trace une galerie des ajustements de doubles Gaussiennes.

    Paramètres:
    -----------
    df_spectrum : DataFrame
        Spectre original contenant "Wavelength" et "Int".
    df_results : DataFrame
        Résultats générés par la fonction `analyze_double_lines_split_results`.
    plot_style : str
        'step' pour un tracé en escalier (idéal pour les pixels) ou 'plot' pour ligne/points.
    cols : int
        Nombre de colonnes dans la galerie.
    """
    # Ne garder que les fits qui ont réussi
    df_success = df_results[df_results['success'] == True]

    # Regrouper les paires de raies (pour ne pas tracer A+B puis B+A)
    processed_pairs = set()
    pairs_to_plot = []
    
    for idx, row in df_success.iterrows():
        line1 = row['Line']
        line2 = row['blended_with']
        
        # Créer un identifiant unique pour la paire (indépendant de l'ordre)
        pair_key = frozenset([line1, line2])
        
        if pair_key not in processed_pairs:
            # Chercher la ligne correspondant au compagnon
            row2 = df_success[(df_success['Line'] == line2) & (df_success['blended_with'] == line1)]
            
            if not row2.empty:
                pairs_to_plot.append((row, row2.iloc[0]))
                processed_pairs.add(pair_key)

    n_plots = len(pairs_to_plot)
    if n_plots == 0:
        print("Aucun fit réussi à tracer.")
        return

    rows = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    wave = df_spectrum['Wavelength'].values
    flux = df_spectrum['Int'].values

    for i, (r1, r2) in enumerate(pairs_to_plot):
        ax = axes[i]
        v1 = r1['Velocity_km_s']
        v2 = r2['Velocity_km_s']
        m1,mtrue1, a1, s1 = r1['wavelength_obs'], r1['wavelength_th'],r1['amplitude'], r1['sigma']
        m2,mtrue2, a2, s2 = r2['wavelength_obs'], r2['wavelength_th'],r2['amplitude'], r2['sigma']
        
        cont_m, cont_c = r1['continuum_m'], r1['continuum_c']

        # Définition dynamique d'une belle fenêtre de tracé (6 sigmas de part et d'autre)
        w_min = min(m1, m2) - 6 * max(s1, s2)
        w_max = max(m1, m2) + 6 * max(s1, s2)

        mask = (wave >= w_min) & (wave <= w_max)
        x_val = wave[mask]
        y_val = flux[mask]

        if len(x_val) == 0:
            ax.set_title(f"Erreur de fenêtre: {r1['Line']} and {r2['Line']}")
            continue

        # Reconstruction des modèles
        y_cont = cont_m * x_val + cont_c
        g1 = a1 * np.exp(-(x_val - m1)**2 / (2 * s1**2))
        g2 = a2 * np.exp(-(x_val - m2)**2 / (2 * s2**2))
        y_fit_total = y_cont + g1 + g2

        # 1. Tracer les données brutes
        if plot_style == 'step':
            ax.step(x_val, y_val, where='mid', color='black', alpha=0.7, label='Spectrum')
            ax.step(x_val, y_cont + g1, color='dodgerblue', linestyle='--', alpha=0.8, label=r1['Line'])
            ax.step(x_val, y_cont + g2, color='darkorange', linestyle='--', alpha=0.8, label=r2['Line'])
            ax.step(x_val, y_fit_total, color='red', linewidth=1.5, label='Total fit')
        else:
            ax.plot(x_val, y_val, color='black', alpha=0.5, marker='.', linestyle='-', label='Spectrum')
            ax.plot(x_val, y_cont + g1, color='dodgerblue', linestyle='--', alpha=0.8, label=r1['Line'])
            ax.plot(x_val, y_cont + g2, color='darkorange', linestyle='--', alpha=0.8, label=r2['Line'])
            ax.plot(x_val, y_fit_total, color='red', linewidth=1.5, label='Total fit')

        ax.plot(x_val, y_cont, color='gray', linestyle=':', label='Continuum')

        delta_lambda = abs(m2 - m1)
        y_arrow = np.min([a1, a2]) * 0.5 + np.mean(y_cont) 
        ax.annotate('', xy=(m1, y_arrow), xytext=(m2, y_arrow),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
        ax.text((m1 + m2) / 2, y_arrow + (max(a1, a2)*0.05), 
                fr'$\Delta\lambda = {delta_lambda:.5f}$',
                color='purple', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        
        velocity_text = f"v({r1['Line']}) = {v1:.1f} km/s\nv({r2['Line']}) = {v2:.1f} km/s"
        ax.text(0.05, 0.95, velocity_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_title(f"{r1['Line']} and {r2['Line']}", fontsize=11, fontweight='bold')
        ax.set_xlabel(r'Wavelength $\lambda$ [$mu m$]')
        ax.set_ylabel('Specific Intensity')
        
        # Lignes verticales discrètes pour les centres de raies observés
        ax.axvline(m1, color='dodgerblue', alpha=0.3, linestyle='-')
        ax.axvline(m2, color='darkorange', alpha=0.3, linestyle='-')
    
        ax.axvspan(mtrue1, m1, color='dodgerblue', alpha=0.15, zorder=0)
        ax.axvspan(mtrue2, m2, color='darkorange', alpha=0.15, zorder=0)
        
        ax.axvline(mtrue1, color='darkblue', alpha=0.4, linestyle='--',label=r"Theoretical $\lambda$")
        ax.axvline(mtrue2, color='red', alpha=0.4, linestyle='--',label=r"Theoretical $\lambda$")
        
        ax.legend(fontsize=8, loc='best')


    # Masquer les subplots vides si la grille est plus grande que le nombre de paires
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_gallery_triple_fits(df_spectrum, df_fits, n_cols=4, window_view=0.1):
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    df_to_plot = df_fits[df_fits['success'] == True].copy()
    if df_to_plot.empty:
        return None, None

    n_plots = len(df_to_plot)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for i, (idx, row) in enumerate(df_to_plot.iterrows()):
        ax = axes[i]
        mu_th = row['wavelength_th']

        mask = (df_spectrum["Wavelength"] >= mu_th - window_view /
                2) & (df_spectrum["Wavelength"] <= mu_th + window_view/2)
        wave_local, flux_local = df_spectrum.loc[mask,
                                                 "Wavelength"], df_spectrum.loc[mask, "Int"]

        # Spectre observé (Noir épais)
        ax.plot(wave_local, flux_local, c='black',
                lw=2.5, alpha=0.9, label='Obs')
        ax.axvline(mu_th, color='gray', linestyle=':', alpha=0.6)

        x_plot = np.linspace(wave_local.min(), wave_local.max(), 200)
        cont = row['continuum_m'] * x_plot + row['continuum_c']

        # Reconstruction des composantes
        # 1. Principale (Green)
        g_main = row['amplitude'] * \
            np.exp(-(x_plot - row['wavelength_obs'])
                   ** 2 / (2 * row['sigma']**2))

        # 2. Secondaires (Orange pointillés)
        # S'assure de récupérer les colonnes des blends (adapter les noms si nécessaire)
        g_b1 = row['amp_2'] * \
            np.exp(-(x_plot - row['mu_2'])**2 / (2 * row['sig_2']**2))
        g_b2 = row['amp_3'] * \
            np.exp(-(x_plot - row['mu_3'])**2 / (2 * row['sig_3']**2))

        # Tracés
        ax.plot(x_plot, cont, linestyle=':', color='gray', alpha=0.5)
        ax.plot(x_plot, g_main + cont, linestyle='-',
                color='green', lw=2, alpha=0.9, label='Main')
        ax.plot(x_plot, g_b1 + cont, linestyle=':',
                color='orange', lw=2, alpha=0.8, label='Blend 1')
        ax.plot(x_plot, g_b2 + cont, linestyle=':',
                color='orange', lw=2, alpha=0.8, label='Blend 2')

        # Somme totale (Bleu tirets)
        total_flux = cont + g_main + g_b1 + g_b2
        ax.plot(x_plot, total_flux, linestyle='--',
                color='blue', lw=2.5, alpha=0.8, label='Total')

        ax.set_title(f"{row['Line']}", fontsize=10)
        ax.text(0.05, 0.95, f"v: {row['v_km_s']:.1f}\nSNR: {row['snr']:.1f}",
                transform=ax.transAxes, fontsize=8, va='top', bbox=dict(boxstyle="round", fc="white", alpha=0.6))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig, axes


def plot_gallery_sided(df_spectrum, df_fits, n_cols=4, window_view=0.1):
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    df_to_plot = df_fits[df_fits['success'] == True].copy()
    if df_to_plot.empty:
        return None, None

    n_plots = len(df_to_plot)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for i, (idx, row) in enumerate(df_to_plot.iterrows()):
        ax = axes[i]
        mu_th = row['wavelength_th']

        mask = (df_spectrum["Wavelength"] >= mu_th - window_view /
                2) & (df_spectrum["Wavelength"] <= mu_th + window_view/2)
        wave_local, flux_local = df_spectrum.loc[mask,
                                                 "Wavelength"], df_spectrum.loc[mask, "Int"]

        # Spectre
        ax.plot(wave_local, flux_local, c='black',
                lw=2, alpha=0.9, label='Obs')
        ax.axvline(mu_th, color='gray', linestyle=':', alpha=0.6, lw=1)

        # Modèles
        x_plot = np.linspace(wave_local.min(), wave_local.max(), 200)
        cont = row['continuum_m'] * x_plot + row['continuum_c']

        g_target = row['amplitude'] * \
            np.exp(-(x_plot - row['wavelength_obs'])
                   ** 2 / (2 * row['sigma']**2))
        g_blend = row['amp_blend'] * \
            np.exp(-(x_plot - row['mu_blend'])**2 / (2 * row['sig_blend']**2))

        # Tracé Composantes
        ax.plot(x_plot, cont, linestyle=':', color='gray', alpha=0.5)
        ax.plot(x_plot, g_target + cont, linestyle='-',
                color='green', lw=2, alpha=0.9, label='Target')

        side_label = "Blend (Left)" if row['side'] == 'blue' else "Blend (Right)"
        ax.plot(x_plot, g_blend + cont, linestyle='--',
                color='orange', lw=1.5, alpha=0.9, label=side_label)

        # Somme
        ax.plot(x_plot, g_target + g_blend + cont, linestyle='-',
                color='blue', lw=2.5, alpha=0.7, label='Total')

        ax.set_title(f"{row['Line']}", fontsize=10, fontweight='bold')
        ax.text(0.05, 0.95, f"v: {row['v_km_s']:.1f}\nSNR: {row['snr']:.1f}",
                transform=ax.transAxes, fontsize=8, va='top', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.legend(fontsize=7, loc='upper right')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig, axes


import numpy as np
import matplotlib.pyplot as plt

def Analyse_Velocity_Detailed(df_final, n_sigma=2.5, relative_to_median=True):
    """
    Version détaillée de l'analyse de vitesse.
    Permet d'afficher les vitesses relatives (V - Mediane) ou absolues (V).
    """
    df_analyse = df_final.copy()

    # 1. Calculs Statistiques
    v = df_analyse['Velocity_km_s']
    median_v = np.median(v)
    print(median_v)
    mean_v = np.mean(v)
    std_v = np.std(v)

    # 2. Configuration selon le mode (Relatif vs Absolu)
    if relative_to_median:
        df_analyse['Plot_V'] = v - median_v
        center_val = 0.0
        mean_offset = mean_v - median_v
        y_label = r"Relative Velocity ($V - V_{median}$) [km/s]"
    else:
        df_analyse['Plot_V'] = v
        center_val = median_v
        mean_offset = mean_v
        y_label = r"Absolute Velocity [km/s]"

    # Détection Outliers
    seuil_bas = center_val - (n_sigma * std_v)
    seuil_haut = center_val + (n_sigma * std_v)

    mask_outlier = (df_analyse['Plot_V'] < seuil_bas) | (df_analyse['Plot_V'] > seuil_haut)
    df_outliers = df_analyse[mask_outlier].copy()
    df_valid = df_analyse[~mask_outlier].copy()

    # --- CONFIGURATION DU PLOT ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={'width_ratios': [2, 1]})

    # === GRAPHIQUE 1 : SCATTER PLOT ===
    ax1.scatter(df_valid['wavelength_th'], df_valid['Plot_V'],
                c='dodgerblue', alpha=0.8, s=120, edgecolors='k', label='Valid Lines', zorder=3)
    ax1.scatter(df_outliers['wavelength_th'], df_outliers['Plot_V'],
                c='crimson', s=180, marker='D', edgecolors='white', label='Outliers', zorder=4)

    # Lignes de référence
    ax1.axhline(center_val, color='black', linestyle='-', linewidth=2,
                label=rf'Median ({median_v:.1f} km/s)')
    ax1.axhline(mean_offset, color='green', linestyle='--', alpha=0.6, label='Mean')

    # Zones Sigma
    print(center_val,n_sigma,std_v)
    ax1.axhline(center_val + n_sigma * std_v, color='red', linestyle=':', lw=2, label=rf'+{n_sigma} $\sigma$ Limit')
    ax1.axhline(center_val - n_sigma * std_v, color='red', linestyle=':', lw=2, label=rf'-{n_sigma} $\sigma$ Limit')
    ax1.fill_between([df_analyse['wavelength_th'].min(), df_analyse['wavelength_th'].max()],
                     center_val - std_v, center_val + std_v, color='gray', alpha=0.3, label=r'$\pm 1\sigma$ Range')

    # Étiquettage
    for i, (idx, row) in enumerate(df_analyse.iterrows()):
        offset_y = 0.8 if i % 2 == 0 else -1.2
        color_txt = 'darkred' if mask_outlier[idx] else 'black'
        weight = 'bold' if mask_outlier[idx] else 'normal'
        fs = 11 if mask_outlier[idx] else 9

        ax1.text(row['wavelength_th'], row['Plot_V'] + offset_y, f"{row['Line']}",
                 fontsize=fs, color=color_txt, fontweight=weight, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=0.5))

    ax1.set_xlabel(r"Theoretical Wavelength [$\mu$m]", fontsize=16)
    ax1.set_ylabel(y_label, fontsize=16)
    ax1.set_title("Detailed Velocity Analysis", fontsize=18, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right', frameon=True, shadow=True, fontsize=12)

    # === GRAPHIQUE 2 : HISTOGRAMME ===
    counts, bins, patches = ax2.hist(df_valid['Plot_V'], bins=12, color='dodgerblue', alpha=0.6, edgecolor='white', label='Valid')
    ax2.hist(df_outliers['Plot_V'], bins=bins, color='crimson', alpha=0.8, edgecolor='white', label='Outliers')

    ax2.set_xlabel(y_label, fontsize=16)
    ax2.set_ylabel("Count", fontsize=16)
    ax2.set_title("Distribution et Ailes", fontsize=18, fontweight='bold')

    ax2.axvline(center_val, color='k', linewidth=2)
    ax2.axvline(center_val + std_v, color='gray', linestyle='--')
    ax2.axvline(center_val - std_v, color='gray', linestyle='--')

    # Identification des ailes
    left_wing = df_analyse[df_analyse['Plot_V'] < center_val - std_v]
    if not left_wing.empty:
        txt_left = "\n".join(left_wing['Line'].values)
        ax2.text(0.02, 0.98, rf"Blue Shifted (below -1$\sigma$):"+"\n"+txt_left,
                 transform=ax2.transAxes, va='top', ha='left', fontsize=10, color='navy',
                 bbox=dict(boxstyle="round", fc="#e6f2ff", ec="navy", alpha=0.9))

    right_wing = df_analyse[df_analyse['Plot_V'] > center_val + std_v]
    if not right_wing.empty:
        txt_right = "\n".join(right_wing['Line'].values)
        ax2.text(0.98, 0.98, rf"Red Shifted (above +1$\sigma$):"+"\n"+txt_right,
                 transform=ax2.transAxes, va='top', ha='right', fontsize=10, color='darkred',
                 bbox=dict(boxstyle="round", fc="#ffe6e6", ec="darkred", alpha=0.9))

    stats_text = (f"GLOBAL STATS:\nN = {len(v)}\nMedian = {median_v:.2f} km/s\nSigma = {std_v:.2f} km/s")
    ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.8))

    plt.tight_layout()
    plt.show()
    
    return fig, (ax1, ax2)




def plot_h_and_co_distributions(df_co, show_fine_co=False):
    fig, ax = plt.subplots(figsize=(12, 8))

    # 1. Reconstruction des données EXACTES pour H I et H2 (issues du graphique)
    # Note : Leurs bins originaux ont une largeur de 5, on les garde pour rester fidèle à l'image
    bins_exact = np.arange(-5, 40, 5)
    hi_counts = [1, 4, 16, 14, 6, 3, 1, 0]
    h2_counts = [0, 0, 0, 2, 8, 2, 0, 1]
    
    hi_data = np.repeat(bins_exact[:-1] + 2.5, hi_counts)
    h2_data = np.repeat(bins_exact[:-1] + 2.5, h2_counts)

    # 2. Extraction des données pour CO
    v_co = df_co['Velocity_km_s'].dropna().values

    # 3. Tracé des histogrammes de base
    ax.hist(hi_data, bins=bins_exact, histtype='step', edgecolor='#0044cc', linewidth=2.5, alpha=0.8, label='H I')
    ax.hist(h2_data, bins=bins_exact, histtype='step', edgecolor='#ff7f0e', linewidth=2.5, alpha=0.8, label='H$_2$')
    
    # Création des intervalles pour CO avec une largeur fixe de 10
    min_co, max_co = np.floor(v_co.min()), np.ceil(v_co.max())
    bins_co_base = np.arange(min_co, max_co + 5, 5) 
    
    ax.hist(v_co, bins=bins_co_base, histtype='step', edgecolor='forestgreen', linewidth=2.5, label='CO (largeur 10)')

    # Ajout optionnel de la distribution fine de CO
    if show_fine_co:
        # Largeur de 2 km/s pour la vue détaillée (ajustable selon tes besoins)
        bins_co_fine = np.arange(min_co, max_co + 2, 2)
        ax.hist(v_co, bins=bins_co_fine, histtype='step', edgecolor='limegreen', linewidth=1.5, linestyle='--', alpha=0.9, label='CO (fin)')

    # 4. Calcul des statistiques et lignes médianes
    datasets = [
        ('H I', hi_data, '#0044cc'), 
        ('H$_2$', h2_data, '#ff7f0e'), 
        ('CO', v_co, 'forestgreen')
    ]
    
    stats_text_lines = []
    for name, data, color in datasets:
        if len(data) > 0:
            mean_v = np.mean(data)
            med_v = np.median(data)
            std_v = np.std(data)
            
            # Ligne verticale pour la médiane
            ax.axvline(med_v, color=color, linestyle='--', linewidth=1.5, alpha=0.5)
            
            # Ajout du texte pour cet élément
            stats_text_lines.append(f"{name} (N={len(data)}):")
            stats_text_lines.append(f"  Médiane = {med_v:.2f} km/s")
            stats_text_lines.append(f"  Moyenne = {mean_v:.2f} km/s")
            stats_text_lines.append(f"  $\sigma$ = {std_v:.2f} km/s")
            stats_text_lines.append("") # Ligne vide d'espacement

    # 5. Esthétique et Labels
    ax.set_xlabel(r"Velocity [km s$^{-1}$]", fontsize=25)
    ax.set_ylabel("Number of lines", fontsize=25)
    
    # Ticks orientés vers l'intérieur
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=25)
    
    # 6. Ajout de la légende et de l'encart statistique
    ax.legend(loc='upper right', fontsize=20, framealpha=1)
    
    stats_text = "\n".join(stats_text_lines).strip()
    ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, va='top', ha='left', 
            fontsize=12, family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="gray", alpha=0.9))
            
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Permet de ne pas couper l'encart texte lors de l'enregistrement ou de l'affichage
    fig.tight_layout()
    plt.subplots_adjust(right=0.75) 
    
    plt.show()

    return fig, ax


def plot_h2_distrib(fig, ax):
    # Intervalles (bins) et valeurs exactes extraites de l'image
    bins = np.arange(-5, 40, 5)
    hi_counts = [1, 4, 16, 14, 6, 3, 1, 0]
    h2_counts = [0, 0, 0, 2, 8, 2, 0, 1]
    
    # Génération de données factices correspondant aux comptes pour utiliser ax.hist
    hi_data = np.repeat(bins[:-1] + 2.5, hi_counts)
    h2_data = np.repeat(bins[:-1] + 2.5, h2_counts)
    total_data = np.concatenate([hi_data, h2_data])

    # Tracé des histogrammes
    # Total en gris clair (zorder bas pour rester en arrière-plan)
    ax.hist(total_data, bins=bins, color='#d3d3d3', label='Total',alpha=0.8, zorder=1)
    # H I et H2 en mode "step" (contours seuls)
    ax.hist(hi_data, bins=bins, histtype='step', edgecolor='#0044cc', linewidth=1.5,alpha=0.8, label='H I', zorder=2)
    ax.hist(h2_data, bins=bins, histtype='step', edgecolor='#ff7f0e', linewidth=1.5,alpha=0.8, label='H$_2$', zorder=3)

    # Étiquettes et limites
    ax.set_xlabel(r'$c(\lambda_{\rm obs} - \lambda_{\rm rest})/\lambda_{\rm rest}$ (km s$^{-1}$)', fontsize=12)
    ax.set_ylabel('Number of lines', fontsize=12)
    ax.set_xlim(-10, 40)
    ax.set_ylim(0, 17)

    # Configuration des graduations (ticks)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    # Réorganisation de la légende pour correspondre à l'image
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 0] # Ordre désiré : H I, H2, Total
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper right', framealpha=1)
    
    
def plot_velocity_histogram(df_final, n_sigma=2.5, relative_to_median=True):
    """
    Isole uniquement l'histogramme des vitesses (relatives ou absolues).
    """
    df_analyse = df_final.copy()

    v = df_analyse['Velocity_km_s']
    median_v = np.median(v)
    std_v = np.std(v)

    if relative_to_median:
        df_analyse['Plot_V'] = v - median_v
        center_val = 0.0
        y_label = r"Relative Velocity ($V - V_{median}$) [km/s]"
    else:
        df_analyse['Plot_V'] = v
        center_val = median_v
        y_label = r"Absolute Velocity [km/s]"

    seuil_bas = center_val - (n_sigma * std_v)
    seuil_haut = center_val + (n_sigma * std_v)

    mask_outlier = (df_analyse['Plot_V'] < seuil_bas) | (df_analyse['Plot_V'] > seuil_haut)
    df_outliers = df_analyse[mask_outlier].copy()
    df_valid = df_analyse[~mask_outlier].copy()

    fig, ax = plt.subplots(figsize=(8, 8))

    counts, bins, patches = ax.hist(df_valid['Plot_V'], bins=12, color='dodgerblue', alpha=0.6, edgecolor='white', label='Valid')
    ax.hist(df_outliers['Plot_V'], bins=bins, color='crimson', alpha=0.8, edgecolor='white', label='Outliers')

    ax.set_xlabel(y_label, fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.set_title("Distribution et Ailes de Vitesse", fontsize=18, fontweight='bold')

    ax.axvline(center_val, color='k', linewidth=2, label='Median')
    ax.axvline(center_val + std_v, color='gray', linestyle='--',)# label=r'$\pm 1\sigma$')
    ax.axvline(center_val - std_v, color='gray', linestyle='--')

    left_wing = df_analyse[df_analyse['Plot_V'] < center_val - std_v]
    if not left_wing.empty:
        txt_left = "\n".join(left_wing['Line'].values)
        ax.text(0.02, 0.98, rf"Blue Shifted (< -1$\sigma$):"+"\n"+txt_left,
                transform=ax.transAxes, va='top', ha='left', fontsize=10, color='navy',
                bbox=dict(boxstyle="round", fc="#e6f2ff", ec="navy", alpha=0.9))

    right_wing = df_analyse[df_analyse['Plot_V'] > center_val + std_v]
    if not right_wing.empty:
        txt_right = "\n".join(right_wing['Line'].values)
        ax.text(0.98, 0.98, rf"Red Shifted (> +1$\sigma$):"+"\n"+txt_right,
                transform=ax.transAxes, va='top', ha='right', fontsize=10, color='darkred',
                bbox=dict(boxstyle="round", fc="#ffe6e6", ec="darkred", alpha=0.9))

    stats_text = (f"N = {len(v)}\nMedian = {median_v:.2f} km/s\nSigma = {std_v:.2f} km/s")
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="square", fc="white", ec="black", alpha=0.8))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_smart_gallery(df_spectrum, df_simple, df_blend, df_forced, n_cols=4, window_view=1.0):
    """
    Trace une galerie intelligente mélangeant raies simples, forcées et blends.

    Priorité d'affichage :
    1. Blend (si présent dans df_blend) -> Affichage double gaussienne.
    2. Forcé (si présent dans df_forced) -> Affichage simple gaussienne (Magenta).
    3. Simple (par défaut dans df_simple) -> Affichage simple gaussienne (Rouge).
    """

    # On filtre les raies réussies du DF principal pour définir la grille
    df_to_plot = df_simple[df_simple['success'] == True].copy()

    if df_to_plot.empty:
        print("Aucune raie réussie dans df_simple.")
        return None, None

    n_plots = len(df_to_plot)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]  # Gestion du cas unique
    axes = np.array(axes).flatten()

    # --- BOUCLE PRINCIPALE ---
    for i, (idx, row_simple) in enumerate(df_to_plot.iterrows()):
        ax = axes[i]
        line_name = row_simple['Line']
        mu_th = row_simple['wavelength_th']

        # --- 1. DÉCISION DU TYPE DE PLOT ---

        # A. Vérifier si c'est un BLEND
        is_blend = False
        row_data = None

        # On regarde si la raie existe dans df_blend
        if df_blend is not None and 'Line' in df_blend.columns:
            match_blend = df_blend[df_blend['Line'] == line_name]
            if not match_blend.empty:
                # On prend la ligne du blend pour les données
                row_data = match_blend.iloc[0]
                is_blend = True

        # B. Vérifier si c'est un FORCÉ (si ce n'est pas déjà un blend)
        is_forced = False
        if not is_blend and df_forced is not None and 'Line' in df_forced.columns:
            match_forced = df_forced[df_forced['Line'] == line_name]
            if not match_forced.empty:
                row_data = match_forced.iloc[0]
                is_forced = True

        # C. Sinon c'est un SIMPLE
        if row_data is None:
            row_data = row_simple

        # --- 2. PRÉPARATION DU SPECTRE (Commun à tous) ---
        mask = (df_spectrum["Wavelength"] >= mu_th - window_view/2) & \
               (df_spectrum["Wavelength"] <= mu_th + window_view/2)

        wave_local = df_spectrum.loc[mask, "Wavelength"]
        flux_local = df_spectrum.loc[mask, "Int"]

        # Tracer les données brutes
        ax.plot(wave_local, flux_local, c='black',
                lw=1.5, alpha=0.8, label='Obs')
        ax.axvline(mu_th, color='gray', linestyle='--', alpha=0.5, lw=1)

        # Grille x pour le modèle
        if len(wave_local) > 0:
            x_plot = np.linspace(wave_local.min(), wave_local.max(), 200)
        else:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
            continue

        # --- 3. TRACÉ SPÉCIFIQUE SELON LE CAS ---

        # === CAS BLEND (Logique de plot_gallery_double_fits) ===
        if is_blend:
            # Récupération params
            amp1 = row_data['amplitude']
            mu1 = row_data['wavelength_obs']
            sig1 = row_data['sigma']

            amp2 = row_data['amp_blend']
            mu2 = row_data['mu_blend']
            sig2 = row_data['sig_blend']

            m = row_data['continuum_m']
            c = row_data['continuum_c']

            cont = m * x_plot + c
            g1 = amp1 * np.exp(-(x_plot - mu1)**2 / (2 * sig1**2))
            g2 = amp2 * np.exp(-(x_plot - mu2)**2 / (2 * sig2**2))

            # Détermination Main vs Blend (celui le plus proche de mu_th)
            dist1 = abs(mu1 - mu_th)
            dist2 = abs(mu2 - mu_th)
            g_main, g_sec = (g1, g2) if dist1 <= dist2 else (g2, g1)

            # Plots
            ax.plot(x_plot, cont, linestyle=':', color='gray', alpha=0.5)
            ax.plot(x_plot, g_main + cont, linestyle='-',
                    color='green', lw=2, alpha=0.8, label='Main')
            ax.plot(x_plot, g_sec + cont, linestyle=':',
                    color='orange', lw=1.5, alpha=0.9, label='Blend')
            ax.plot(x_plot, g1 + g2 + cont, linestyle='-.',
                    color='blue', lw=2.5, alpha=0.7, label='Total')

            title_color = 'blue'
            fit_type = "BLEND"

        # === CAS SIMPLE OU FORCÉ (Logique de plot_gallery_fits) ===
        else:
            # Paramètres
            amp = row_data['amplitude']
            mu = row_data['wavelength_obs']
            sig = row_data['sigma']
            m = row_data['continuum_m']
            c = row_data['continuum_c']

            cont = m * x_plot + c
            gaussian = amp * np.exp(-(x_plot - mu)**2 / (2 * sig**2))

            if is_forced:
                # Style FORCÉ
                line_c = 'magenta'
                fit_type = "FORCED"
                ax.plot(x_plot, cont, linestyle=':', color=line_c, lw=2)
                ax.plot(x_plot, gaussian + cont, linestyle='-',
                        color=line_c, lw=2.5, label='Forced')
                title_color = 'magenta'
            else:
                # Style STANDARD
                line_c = 'crimson'
                fit_type = "SIMPLE"
                ax.plot(x_plot, cont, linestyle=':', color=line_c, lw=2)
                ax.plot(x_plot, gaussian + cont, linestyle='-',
                        color=line_c, lw=2.5, label='Simple')
                title_color = 'black'

        # --- 4. ESTHÉTIQUE & INFOS ---
        ax.set_title(f"{line_name} ({fit_type})", fontsize=11,
                     fontweight='bold', color=title_color)

        # Info box
        snr_val = row_data['snr'] if 'snr' in row_data else 0
        v_val = row_data['Velocity_km_s'] if 'Velocity_km_s' in row_data else 0

        info_txt = f"v: {v_val:.1f}\nSNR: {snr_val:.1f}"
        ax.text(0.05, 0.95, info_txt, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # Nettoyage des axes vides
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_spectral_summary_v2(df_spectrum, df_merged,
                             wl_range=(4.576, 4.925), n_plots=4, ylim=(100, 200),
                             snr_threshold=3.0, v_range=(-10, 25)):
    """
    Trace le résumé spectral à partir du 'Grand DF' fusionné.
    - Affiche les composantes gaussiennes en pointillés.
    - Met en valeur la gaussienne principale.
    - Calcule le SNR sur les résidus du continuum.
    """
    wl_min, wl_max = wl_range
    ylim_min, ylim_max = ylim
    delta_wl = (wl_max - wl_min) / n_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(20, 7 * n_plots))
    if n_plots == 1:
        axes = [axes]

    # Pour chaque panneau (bande de fréquence)
    for i, ax in enumerate(axes):
        c_min = wl_min + i * delta_wl
        c_max = wl_min + (i + 1) * delta_wl

        # --- A. TRACÉ DU SPECTRE OBSERVÉ ---
        mask_view = (df_spectrum["Wavelength"] >= c_min) & (
            df_spectrum["Wavelength"] <= c_max)
        if mask_view.sum() == 0:
            continue

        wave_view = df_spectrum.loc[mask_view, "Wavelength"].values
        flux_view = df_spectrum.loc[mask_view, "Int"].values

        ax.plot(wave_view, flux_view, color="black", lw=1,
                alpha=0.9, zorder=1, label='Spectre')

        # --- B. RÉCUPÉRATION DES LIGNES DANS CETTE ZONE ---
        # On filtre le df_merged pour ne garder que ce qui est dans la fenêtre
        subset = df_merged[
            (df_merged['wavelength_obs'] >= c_min) &
            (df_merged['wavelength_obs'] <= c_max)
        ]

        # Gestion des hauteurs d'étiquettes pour éviter le chevauchement
        y_text_levels = [ylim_max * 0.70, ylim_max * 0.85]

        # --- C. BOUCLE SUR CHAQUE LIGNE ---
        for idx, (_, row) in enumerate(subset.iterrows()):
            fit_type = row['fit_type']
            line_name = row['Line']

            mu = row['wavelength_obs']
            sigma = row['sigma']
            amp = row['amplitude']
            m, c = row['continuum_m'], row['continuum_c']
            vel = row.get('Velocity_km_s', 0)

            # Grille locale pour le modèle (plus fine pour être joli)
            x_mod = np.linspace(mu - 0.006, mu + 0.006, 150)
            cont_local = m * x_mod + c

            # --- D. CONSTRUCTION DES MODÈLES ---
            g_main = amp * np.exp(-(x_mod - mu)**2 / (2 * sigma**2))
            g_blend = np.zeros_like(x_mod)  # Par défaut 0

            if fit_type == 'BLEND':
                # Récupération de la 2ème gaussienne
                amp2 = row['amp_blend']
                mu2 = row['mu_blend']
                sig2 = row['sig_blend']
                g_blend = amp2 * np.exp(-(x_mod - mu2)**2 / (2 * sig2**2))

                base_color = 'blue'

                # Tracé des composantes (Pointillés)
                # 1. La "Main" (Intéressante) : Tiretée, colorée
                ax.plot(x_mod, cont_local + g_main, ls='--',
                        color=base_color, lw=1.5, alpha=0.8)
                # 2. L'intruse (Blend) : Pointillés fins, gris ou cyan pâle
                ax.plot(x_mod, cont_local + g_blend, ls=':',
                        color='teal', lw=1.2, alpha=0.6)

            elif fit_type == 'FORCED':
                base_color = 'magenta'
                # Juste la main en tirets
                ax.plot(x_mod, cont_local + g_main, ls='--',
                        color=base_color, lw=1.5, alpha=0.8)

            else:  # SIMPLE
                base_color = 'crimson'
                ax.plot(x_mod, cont_local + g_main, ls='--',
                        color=base_color, lw=1.5, alpha=0.8)

            # Tracé de la SOMME (Ligne pleine solide)
            total_model = cont_local + g_main + g_blend
            ax.plot(x_mod, total_model, ls='-', color=base_color,
                    lw=2.5, alpha=0.9, zorder=5)

            # Tracé du continuum local
            ax.plot(x_mod, cont_local, ls=':', color='gray', alpha=0.5)

            # --- E. CALCUL DU SNR "OBSERVÉ" (Méthode robuste) ---
            # On prend une fenêtre un peu plus large autour de la raie
            mask_snr = (wave_view > mu - 0.02) & (wave_view < mu + 0.02)
            w_local = wave_view[mask_snr]
            f_local = flux_view[mask_snr]

            if len(w_local) > 10:
                # Modèle continuum sur cette zone
                c_model = m * w_local + c
                residuals = f_local - c_model

                # Astuce "2% les plus bas" :
                # On regarde les résidus qui sont proches de 0 ou négatifs (là où il n'y a pas de raie)
                # Pour éviter les outliers aberrants, on prend le quantile 2% à 50% (la moitié basse des points)
                # Cela nous donne une estimation du "tapis" de bruit.
                noise_candidates = residuals[residuals <
                                             np.percentile(residuals, 50)]

                # Calcul RMS sur ces points bas (le "socle" du bruit)
                if len(noise_candidates) > 0:
                    noise_rms = np.std(noise_candidates)
                else:
                    noise_rms = 1.0  # Fallback

                snr_obs = amp / noise_rms if noise_rms > 0 else 0
            else:
                snr_obs = 0

            # --- F. ANNOTATIONS & ALERTES ---
            is_bad = (snr_obs < snr_threshold) or (
                vel < v_range[0]) or (vel > v_range[1])

            box_color = "red" if is_bad else "black"
            bg_color = "mistyrose" if is_bad else "white"

            label_text = (f"{line_name}\n"
                          f"v: {vel:.1f} km/s\n"
                          f"SNR: {snr_obs:.1f}")

            y_pos = y_text_levels[idx % 2]  # Alterne haut/bas

            ax.annotate(label_text,
                        xy=(mu, total_model.max()),
                        xytext=(mu, ylim_max*0.9),
                        arrowprops=dict(arrowstyle='->',
                                        color=box_color, alpha=0.5),
                        fontsize=9, color=box_color, fontweight='bold', ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc=bg_color, ec=box_color, alpha=0.8))

        # Esthétique finale du panneau
        ax.set_ylim(ylim_min, ylim_max)
        ax.set_xlim(c_min, c_max)
        ax.set_ylabel("Flux (MJy/sr)")
        ax.grid(True, alpha=0.2, linestyle='--')

    axes[-1].set_xlabel("Wavelength ($\mu$m)")

    # Légende manuelle pour expliquer les tirets
    lines = [Line2D([0], [0], color='black', lw=2, label='Fit Global'),
             Line2D([0], [0], color='black', ls='--',
                    lw=1.5, label='Composante Principale'),
             Line2D([0], [0], color='teal', ls=':', lw=1.5, label='Comp. Secondaire (Blend)')]
    axes[0].legend(handles=lines, loc='upper right', fontsize=10)

    plt.tight_layout()
    return fig, axes


def plot_smart_gallery_from_merged(df_spectrum, df_merged, n_cols=6, window_view=0.008, clean_mode=False,style_plot="plot"):
    """
    Trace une galerie de vignettes (une par raie) à partir du DataFrame fusionné.

    Args:
        df_spectrum: Le spectre complet.
        df_merged: Le grand DataFrame issu de merge_spectral_dataframes.
        clean_mode (bool): Si True, affiche tout en bleu uniforme sans texte (pour publication).
    """
    if df_merged.empty:
        print("Aucune donnée à tracer.")
        return

    n_plots = len(df_merged)
    n_rows = math.ceil(n_plots / n_cols)

    # Taille ajustée selon le mode (plus compact si clean mode car pas de texte)
    h_ratio = 3 if clean_mode else 4
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, h_ratio * n_rows))

    if n_plots == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    # Couleur uniforme pour le clean mode
    UNIFORM_COLOR = 'tab:blue'

    for i, (idx, row) in enumerate(df_merged.iterrows()):
        ax = axes[i]

        # Récupération des infos
        line_name = row['Line']
        fit_type = row['fit_type']
        mu = row['wavelength_obs']
        sigma = row['sigma']
        amp = row['amplitude']
        m, c = row['continuum_m'], row['continuum_c']

        # --- 1. PRÉPARATION DES DONNÉES ---
        # Fenêtre locale autour de la raie
        mask = (df_spectrum["Wavelength"] >= mu - window_view/2) & \
               (df_spectrum["Wavelength"] <= mu + window_view/2)

        wave_local = df_spectrum.loc[mask, "Wavelength"]
        flux_local = df_spectrum.loc[mask, "Int"]

        if len(wave_local) == 0:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
            continue
        if style_plot == "plot" : 
        # Plot des points observés
            ax.plot(wave_local, flux_local, c='black',
                    lw=2, alpha=0.9, label='Obs')
        elif style_plot == "step" : 
        # Plot des points observés
            ax.step(wave_local, flux_local, c='black',
                    lw=2, alpha=0.9, label='Obs')
        # Grille pour le modèle
        x_plot = np.linspace(wave_local.min(), wave_local.max(), 200)
        cont = m * x_plot + c

        # --- 2. CALCUL DES MODÈLES ---
        g_main = amp * np.exp(-(x_plot - mu)**2 / (2 * sigma**2))
        g_blend = np.zeros_like(x_plot)

        # Gestion spécifique des Blends
        if fit_type == 'BLEND':
            g_blend = row['amp_blend'] * \
                np.exp(-(x_plot - row['mu_blend']) **
                       2 / (2 * row['sig_blend']**2))

        total_model = cont + g_main + g_blend

        # --- 3. GESTION DES COULEURS ET STYLE ---
        if clean_mode:
            main_color = UNIFORM_COLOR
            blend_color = UNIFORM_COLOR
            fit_linestyle = '-'
            title_text = ""
            ax.text(0.05, 0.9, line_name, transform=ax.transAxes,
                    fontsize=10, fontweight='bold', color='black')
        else:
            if fit_type == 'BLEND':
                main_color = 'green'  # La raie qu'on veut
                blend_color = 'orange'  # L'intrus
                fit_linestyle = '-'
                title_color = 'blue'
            elif fit_type == 'FORCED':
                main_color = 'magenta'
                blend_color = None
                title_color = 'magenta'
            else:  # SIMPLE
                main_color = 'crimson'
                blend_color = None
                title_color = 'black'

            ax.set_title(f"{line_name} ({fit_type})", fontsize=10,
                         color=title_color, fontweight='bold')

        # --- 4. TRACÉ DES COURBES ---
        ax.plot(x_plot, cont, linestyle=':', color='gray', alpha=0.5)

        ax.plot(x_plot, g_main + cont, linestyle='--',
                color=main_color, lw=1.5, alpha=0.8)
        if fit_type == 'BLEND':
            ax.plot(x_plot, g_blend + cont, linestyle=':',
                    color=blend_color, lw=1.5, alpha=0.8)
        ax.plot(x_plot, total_model, linestyle='-', color=main_color,
                lw=2, alpha=0.6 if clean_mode else 0.4)

        if not clean_mode:
            snr = row.get('snr', 0)
            vel = row.get('Velocity_km_s', 0)
            info_txt = f"v: {vel:.1f}\nSNR: {snr:.1f}"
            ax.text(0.95, 0.99, info_txt, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Nettoyage des axes vides
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return fig, axes




def plot_smart_gallery_from_mergedV2(df_spectrum, df_merged, n_cols=6, window_view=0.008, clean_mode=False, style_plot="plot"):
    """
    Trace une galerie de vignettes (une par raie ou par paire) à partir du DataFrame fusionné.
    Gère les fits SIMPLES, FORCÉS, BLENDS et regroupe les DOUBLES.
    Affiche l'écart lambda théorique vs observé avec un code couleur (Rouge=Redshift, Bleu=Blueshift).
    """
    if df_merged.empty:
        print("Aucune donnée à tracer.")
        return

    # 1. Regroupement des raies DOUBLE pour ne pas créer deux subplots
    processed_doubles = set()
    items_to_plot = [] # Liste d'éléments à tracer (soit 1 ligne, soit un tuple de 2 lignes)

    for idx, row in df_merged.iterrows():
        if row['fit_type'] == 'DOUBLE':
            line1 = row['Line']
            line2 = row.get('blended_with', None)
            pair_key = frozenset([line1, line2])
            
            if pair_key not in processed_doubles:
                try:
                    row2 = df_merged[(df_merged['Line'] == line2) & (df_merged['fit_type'] == 'DOUBLE')].iloc[0]
                    items_to_plot.append(('DOUBLE', row, row2))
                except:
                    # Sécurité si le compagnon manque pour une raison X ou Y
                    items_to_plot.append(('SINGLE', row))
                processed_doubles.add(pair_key)
        else:
            items_to_plot.append(('SINGLE', row))

    n_plots = len(items_to_plot)
    if n_plots == 0:
        return

    n_rows = math.ceil(n_plots / n_cols)
    h_ratio = 3 if clean_mode else 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, h_ratio * n_rows))

    if n_plots == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    UNIFORM_COLOR = 'tab:blue'
    N_SIGMA = 5 # Pour la fenêtre locale si window_view n'est pas suffisant

    for i, item in enumerate(items_to_plot):
        ax = axes[i]
        item_type = item[0]

        # --- GESTION DES DONNÉES (SINGLE vs DOUBLE) ---
        if item_type == 'SINGLE':
            row = item[1]
            line_name = row['Line']
            fit_type = row['fit_type']
            mu, sigma, amp = row['wavelength_obs'], row['sigma'], row['amplitude']
            lth = row['wavelength_th']
            m, c = row['continuum_m'], row['continuum_c']
            
            # Fenêtre
            w_min, w_max = mu - window_view/2, mu + window_view/2
            if fit_type == 'BLEND':
                w_min = min(w_min, row['mu_blend'] - 3*row['sig_blend'])
                w_max = max(w_max, row['mu_blend'] + 3*row['sig_blend'])
                
            title_name = f"{line_name} ({fit_type})"
            
        else: # DOUBLE
            r1, r2 = item[1], item[2]
            line_name = f"{r1['Line']} and {r2['Line']}"
            fit_type = 'DOUBLE'
            
            # On prend les infos de la première pour la fenêtre globale
            mu1, sig1 = r1['wavelength_obs'], r1['sigma']
            mu2, sig2 = r2['wavelength_obs'], r2['sigma']
            lth1, lth2 = r1['wavelength_th'], r2['wavelength_th']
            
            w_min = min(mu1 - 3*sig1, mu2 - 3*sig2, mu1 - window_view/2)
            w_max = max(mu1 + 3*sig1, mu2 + 3*sig2, mu1 + window_view/2)
            
            title_name = f"{r1['Line']} and {r2['Line']} (DOUBLE)"

        # --- EXTRACTION SPECTRE LOCAL ---
        mask = (df_spectrum["Wavelength"] >= w_min) & (df_spectrum["Wavelength"] <= w_max)
        wave_local = df_spectrum.loc[mask, "Wavelength"].values
        flux_local = df_spectrum.loc[mask, "Int"].values

        if len(wave_local) == 0:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
            continue

        if style_plot == "plot": 
            ax.plot(wave_local, flux_local, c='black', lw=2, alpha=0.9, label='Obs', marker='.')
        elif style_plot == "step": 
            ax.step(wave_local, flux_local, c='black', lw=2, alpha=0.9, label='Obs', where='mid')

        # Grille pour le modèle
        x_plot = np.linspace(wave_local.min(), wave_local.max(), 200)

        # --- CALCUL DES MODÈLES ---
        if item_type == 'SINGLE':
            cont = m * x_plot + c
            g_main = amp * np.exp(-(x_plot - mu)**2 / (2 * sigma**2))
            g_sec = np.zeros_like(x_plot)
            if fit_type == 'BLEND':
                g_sec = row['amp_blend'] * np.exp(-(x_plot - row['mu_blend'])**2 / (2 * row['sig_blend']**2))
            total_model = cont + g_main + g_sec
        else: # DOUBLE
            # Pour DOUBLE, rappel : c est l'ordonnée à l'origine absolue x=0
            cont = r1['continuum_m'] * x_plot + r1['continuum_c']
            g_main = r1['amplitude'] * np.exp(-(x_plot - r1['wavelength_obs'])**2 / (2 * r1['sigma']**2))
            g_sec = r2['amplitude'] * np.exp(-(x_plot - r2['wavelength_obs'])**2 / (2 * r2['sigma']**2))
            total_model = cont + g_main + g_sec

        # --- GESTION DES COULEURS ---
        if clean_mode:
            main_color, blend_color, total_color = UNIFORM_COLOR, UNIFORM_COLOR, UNIFORM_COLOR
            title_color = 'black'
            ax.text(0.05, 0.9, line_name, transform=ax.transAxes, fontsize=6, fontweight='bold', color='black')
        else:
            if fit_type == 'BLEND':
                main_color, blend_color, total_color = 'green', 'orange', 'green'
                title_color = 'blue'
            elif fit_type == 'DOUBLE':
                main_color, blend_color, total_color = 'purple', 'orchid', 'purple'
                title_color = 'purple'
            elif fit_type == 'FORCED':
                main_color, blend_color, total_color = 'magenta', None, 'magenta'
                title_color = 'magenta'
            else: # SIMPLE
                main_color, blend_color, total_color = 'crimson', None, 'crimson'
                title_color = 'black'

        if not clean_mode:
            ax.set_title(title_name, fontsize=9, color=title_color, fontweight='bold')

        # --- TRACÉ DES COURBES ---
        ax.plot(x_plot, cont, linestyle=':', color='gray', alpha=0.5)
        ax.plot(x_plot, g_main + cont, linestyle='--', color=main_color, lw=1.5, alpha=0.8)
        
        if fit_type in ['BLEND', 'DOUBLE']:
            ax.plot(x_plot, g_sec + cont, linestyle='--', color=blend_color, lw=1.5, alpha=0.8)
            
        ax.plot(x_plot, total_model, linestyle='-', color=total_color, lw=2, alpha=0.6 if clean_mode else 0.4)

        # --- SHIFT DOPPLER (Lignes et Couleurs) ---
        def plot_shift(l_th, l_obs, color_line):
            shift_color = 'cornflowerblue' if l_obs > l_th else 'salmon'
            ax.axvspan(l_th, l_obs, color=shift_color, alpha=0.2, zorder=0)
            ax.axvline(l_th, color='gray', ls='--', alpha=0.7, lw=1)
            ax.axvline(l_obs, color=color_line, ls='-', alpha=0.7, lw=1)

        if item_type == 'SINGLE':
            plot_shift(lth, mu, main_color)
        else:
            plot_shift(lth1, mu1, main_color)
            plot_shift(lth2, mu2, blend_color)

        # --- BOÎTES D'INFORMATIONS ---
        if not clean_mode:
            if item_type == 'SINGLE':
                snr = row.get('snr', 0)
                vel = row.get('Velocity_km_s', 0)
                info_txt = f"v: {vel:.1f}\nSNR: {snr:.1f}"
                ax.text(0.95, 0.95, info_txt, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle="round", fc="white", ec=main_color, alpha=0.8))
            else:
                v1, v2 = r1.get('Velocity_km_s', 0), r2.get('Velocity_km_s', 0)
                snr1, snr2 = r1.get('snr', 0), r2.get('snr', 0)
                info_txt = f"v={v1:.1f} \n{r1['Line']}: SNR={snr1:.1f} \n{r2['Line']}: SNR={snr2:.1f}"
                ax.text(0.5, 0.95, info_txt, transform=ax.transAxes, fontsize=8,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle="round", fc="white", ec=total_color, alpha=0.8))

    # Nettoyage des axes vides si la grille est plus grande que le nombre d'items
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return fig, axes







def plot_spectral_summary_clean(df_spectrum, df_merged,type_plot ="step",
                                wl_range=(4.576, 4.925), n_plots=3, ylim=(100, 200),
                                snrtype='snr2',snr_threshold=3.0, v_range=(-10, 25),
                                clean_mode=False,alpha=0.9):
    """
    Trace le résumé spectral avec :
    1. Fits coupés proprement (sigma-clipping).
    2. Continuum discret.
    3. Gestion intelligente des chevauchements d'étiquettes.
    """
    wl_min, wl_max = wl_range
    delta_wl = (wl_max - wl_min) / n_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(20, 7 * n_plots))
    if n_plots == 1:
        axes = [axes]

    # Couleur pour le mode clean
    CLEAN_COLOR = "#D55E00"

    # Paramètre de coupure : combien de sigma avant de couper ?
    N_SIGMA_CUT = 4.5

    for i, ax in enumerate(axes):
        ax.set_ylim(ylim)

        c_min = wl_min + i * delta_wl
        c_max = wl_min + (i + 1) * delta_wl
        plot_width = c_max - c_min

        # 1. Tracé spectre (Fond noir)
        mask_view = (df_spectrum["Wavelength"] >= c_min) & (
            df_spectrum["Wavelength"] <= c_max)
        if mask_view.sum() == 0:
            continue

        wave_view = df_spectrum.loc[mask_view, "Wavelength"].values
        flux_view = df_spectrum.loc[mask_view, "Int"].values
        if type_plot == "plot" : 
            ax.plot(wave_view, flux_view, color="black", lw=2, alpha=0.9, zorder=1)
        elif type_plot == "step" : 
            ax.step(wave_view, flux_view, color="black", lw=2, alpha=0.9, zorder=1)
        # 2. Filtrage et Tri
        subset = df_merged[(df_merged['wavelength_obs'] >= c_min) & (
            df_merged['wavelength_obs'] <= c_max)]
        subset = subset.sort_values(by='wavelength_obs')

        # --- LOGIQUE ANTI-CHEVAUCHEMENT ---
        last_x_pos = -999
        level_toggle = 0

        # Hauteurs des étiquettes
        y_range = ylim[1] - ylim[0]
        y_levels = [ylim[0] + 0.96 * y_range,
                    ylim[0] + 0.82 * y_range]  # Haut, Bas

        for idx, (_, row) in enumerate(subset.iterrows()):
            fit_type = row['fit_type']
            mu, sigma, amp = row['wavelength_obs'], row['sigma'], row['amplitude']
            m, c = row['continuum_m'], row['continuum_c']

            # --- CALCUL DYNAMIQUE DE LA FENÊTRE DE TRACÉ ---
            if fit_type == 'BLEND':
                # On prend le min et max des deux gaussiennes pour couvrir tout le blend
                mu2, sig2 = row['mu_blend'], row['sig_blend']
                start_x = min(mu - N_SIGMA_CUT*sigma, mu2 - N_SIGMA_CUT*sig2)
                end_x = max(mu + N_SIGMA_CUT*sigma, mu2 + N_SIGMA_CUT*sig2)
            else:
                start_x = mu - N_SIGMA_CUT * sigma
                end_x = mu + N_SIGMA_CUT * sigma

            # Grille modèle restreinte à la zone utile
            x_mod = np.linspace(start_x, end_x, 100)
            cont_local = m * x_mod + c

            # Reconstruction gaussiennes
            g_main = amp * np.exp(-(x_mod - mu)**2 / (2 * sigma**2))
            g_blend = np.zeros_like(x_mod)

            if fit_type == 'BLEND':
                g_blend = row['amp_blend'] * \
                    np.exp(-(x_mod - row['mu_blend']) **
                           2 / (2 * row['sig_blend']**2))

            total_model = cont_local + g_main + g_blend

            # --- GESTION COULEUR ---
            if clean_mode:
                base_color = "darkgreen"
                sec_color = "darkorange"
            else:
                if fit_type == 'BLEND':
                    base_color, sec_color = 'blue', 'teal'
                elif fit_type == 'FORCED':
                    base_color, sec_color = 'magenta', 'magenta'
                else:
                    base_color, sec_color = 'crimson', 'crimson'

            # --- TRACÉ DES COURBES ---
            # A. Continuum (Discret, coupé aussi)
            ax.plot(x_mod, cont_local, ls=':', color='gray',
                    lw=1.2, alpha=0.5, zorder=2)

            # B. Main Component
            ax.plot(x_mod, cont_local + g_main, ls='--',
                    color=base_color, lw=1.5, alpha=0.5, zorder=3)

            # C. Blend Component
            if fit_type == 'BLEND':
                ax.plot(x_mod, cont_local + g_blend, ls=':',
                        color=sec_color, lw=1.2, alpha=0.3, zorder=3)

            # D. Total
            ax.plot(x_mod, total_model, ls='-', color=base_color,
                    lw=2.5, alpha=alpha, zorder=5)

            # --- ANNOTATIONS ---
            if not clean_mode:
                snr_val = row.get(snrtype, 0)
                vel_val = row.get('Velocity_km_s', 0)

                is_bad = (snr_val < snr_threshold) or (
                    vel_val < v_range[0]) or (vel_val > v_range[1])
                box_color = "red" if is_bad else "black"

                # Gestion chevauchement étiquettes
                dist_to_prev = abs(mu - last_x_pos)
                threshold_overlap = plot_width * 0.05

                if dist_to_prev < threshold_overlap:
                    level_toggle = (level_toggle + 1) % 2
                else:
                    level_toggle = 0

                y_text_pos = y_levels[level_toggle]
                last_x_pos = mu

                label_txt = f"{row['Line']}\nv:{vel_val:.1f}|SNR:{snr_val:.1f}"

                # La flèche pointe vers le sommet de la courbe (total_model.max())
                ax.annotate(label_txt,
                            xy=(mu, total_model.max()),
                            xytext=(mu, y_text_pos),
                            arrowprops=dict(arrowstyle='->',
                                            color=box_color, alpha=0.5),
                            fontsize=9, color=box_color, ha='center', va='top',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=box_color, alpha=0.8, zorder=20))

        ax.set_xlim(c_min, c_max)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Wavelength ($\mu$m)")
    plt.tight_layout()
    return fig, axes







def plot_spectral_summary_cleanV2(df_spectrum, df_merged, type_plot="step",
                                wl_range=(4.576, 4.925), n_plots=3, ylim=(100, 200),
                                snrtype='snr2', snr_threshold=3.0, v_range=(-10, 25),
                                clean_mode=False, alpha=0.9):
    """
    Trace le résumé spectral avec :
    1. Fits coupés proprement.
    2. Continuum corrigé (y = mx+c global).
    3. Tracé des modèles DOUBLES sans doublons.
    4. Code couleur Doppler : Rouge (Shift positif) / Bleu (Shift négatif).
    5. Étiquettes verticales.
    """
    wl_min, wl_max = wl_range
    delta_wl = (wl_max - wl_min) / n_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(20, 7 * n_plots))
    if n_plots == 1:
        axes = [axes]

    N_SIGMA_CUT = 4.5

    for i, ax in enumerate(axes):
        ax.set_ylim(ylim)

        c_min = wl_min + i * delta_wl
        c_max = wl_min + (i + 1) * delta_wl
        plot_width = c_max - c_min

        # 1. Tracé spectre
        mask_view = (df_spectrum["Wavelength"] >= c_min) & (df_spectrum["Wavelength"] <= c_max)
        if mask_view.sum() == 0:
            continue

        wave_view = df_spectrum.loc[mask_view, "Wavelength"].values
        flux_view = df_spectrum.loc[mask_view, "Int"].values
        
        if type_plot == "plot": 
            ax.plot(wave_view, flux_view, color="black", lw=2, alpha=0.9, zorder=1)
        elif type_plot == "step": 
            ax.step(wave_view, flux_view, color="black", lw=2, alpha=0.9, zorder=1)
            
        # 2. Filtrage et Tri
        subset = df_merged[(df_merged['wavelength_obs'] >= c_min) & (df_merged['wavelength_obs'] <= c_max)]
        subset = subset.sort_values(by='wavelength_obs')

        # Variables de suivi
        last_x_pos = -999
        level_toggle = 0
        y_range = ylim[1] - ylim[0]
        y_levels = [ylim[0] + 0.85 * y_range, ylim[0] + 0.65 * y_range]  # Ajusté pour le texte vertical
        
        # Pour éviter de tracer 2 fois le modèle complet des DOUBLE
        plotted_pairs = set()

        for idx, (_, row) in enumerate(subset.iterrows()):
            fit_type = row['fit_type']
            mu, sigma, amp = row['wavelength_obs'], row['sigma'], row['amplitude']
            lth = row['wavelength_th']
            m, c = row['continuum_m'], row['continuum_c']

            # --- CALCUL DE LA FENÊTRE ---
            if fit_type == 'BLEND':
                mu2, sig2 = row['mu_blend'], row['sig_blend']
                start_x = min(mu - N_SIGMA_CUT*sigma, mu2 - N_SIGMA_CUT*sig2)
                end_x = max(mu + N_SIGMA_CUT*sigma, mu2 + N_SIGMA_CUT*sig2)
            elif fit_type == 'DOUBLE':
                try:
                    comp = df_merged[(df_merged['Line'] == row['blended_with']) & (df_merged['fit_type'] == 'DOUBLE')].iloc[0]
                    mu2, sig2 = comp['wavelength_obs'], comp['sigma']
                    start_x = min(mu - N_SIGMA_CUT*sigma, mu2 - N_SIGMA_CUT*sig2)
                    end_x = max(mu + N_SIGMA_CUT*sigma, mu2 + N_SIGMA_CUT*sig2)
                except:
                    start_x, end_x = mu - N_SIGMA_CUT * sigma, mu + N_SIGMA_CUT * sigma
            else:
                start_x, end_x = mu - N_SIGMA_CUT * sigma, mu + N_SIGMA_CUT * sigma

            x_mod = np.linspace(start_x, end_x, 150)

            # --- RECONSTRUCTION DU MODÈLE CORRIGÉ ---
            # Le continuum est strictement mx+c pour tout le monde maintenant
            cont_local = m * x_mod + c

            g_main = amp * np.exp(-(x_mod - mu)**2 / (2 * sigma**2))
            g_blend = np.zeros_like(x_mod)
            g_compagnon = np.zeros_like(x_mod)

            if fit_type == 'BLEND':
                g_blend = row['amp_blend'] * np.exp(-(x_mod - row['mu_blend'])**2 / (2 * row['sig_blend']**2))
            elif fit_type == 'DOUBLE':
                try:
                    comp = df_merged[(df_merged['Line'] == row['blended_with']) & (df_merged['fit_type'] == 'DOUBLE')].iloc[0]
                    g_compagnon = comp['amplitude'] * np.exp(-(x_mod - comp['wavelength_obs'])**2 / (2 * comp['sigma']**2))
                except:
                    pass

            total_model = cont_local + g_main + g_blend + g_compagnon

            # --- GESTION COULEUR GLOBALE ---
            if clean_mode:
                base_color, sec_color = "darkgreen", "darkorange"
            else:
                if fit_type == 'DOUBLE': base_color, sec_color = 'purple', 'orchid'
                elif fit_type == 'BLEND': base_color, sec_color = 'blue', 'teal'
                elif fit_type == 'FORCED': base_color, sec_color = 'magenta', 'magenta'
                else: base_color, sec_color = 'crimson', 'crimson'

            # --- TRACÉ DES COURBES ---
            ax.plot(x_mod, cont_local, ls=':', color='gray', lw=1.2, alpha=0.5, zorder=2)
            
            # Trace la composante individuelle (pour que chaque raie ait son pic en pointillé)
            ax.plot(x_mod, cont_local + g_main, ls='--', color=sec_color, lw=1.5, alpha=0.6, zorder=3)
            if fit_type == 'BLEND':
                ax.plot(x_mod, cont_local + g_blend, ls=':', color=sec_color, lw=1.2, alpha=0.6, zorder=3)

            # Traçage du modèle TOTAL (avec sécurité anti-doublon pour les DOUBLE)
            if fit_type == 'DOUBLE':
                pair_key = frozenset([row['Line'], row.get('blended_with', None)])
                if pair_key not in plotted_pairs:
                    ax.plot(x_mod, total_model, ls='-', color=base_color, lw=2.5, alpha=alpha, zorder=5)
                    plotted_pairs.add(pair_key)
            else:
                ax.plot(x_mod, total_model, ls='-', color=base_color, lw=2.5, alpha=alpha, zorder=5)

            # --- MISE EN ÉVIDENCE DU SHIFT DOPPLER (ROUGE / BLEU) ---
            delta_lambda = mu - lth
            shift_color = 'salmon' if delta_lambda > 0 else 'cornflowerblue' # Rouge si s'éloigne, Bleu si se rapproche
            
            ax.axvspan(lth, mu, color=shift_color, alpha=0.15, zorder=0)
            
            # Ligne Théorique (Pointillés gris)
            ax.axvline(lth, color='gray', ls='--', alpha=0.7, lw=1.2, zorder=1)
            # Ligne Observée (Continue couleur raie)
            ax.axvline(mu, color=base_color, ls='-', alpha=0.7, lw=1.2, zorder=1)

            # --- ANNOTATIONS VERTICALES ---
            if not clean_mode:
                snr_val = row.get(snrtype, 0)
                vel_val = row.get('Velocity_km_s', 0)

                is_bad = (snr_val < snr_threshold) or (vel_val < v_range[0]) or (vel_val > v_range[1])
                box_color = "red" if is_bad else "black"

                # Logique anti-chevauchement (ajustée pour des textes verticaux plus fins)
                dist_to_prev = abs(mu - last_x_pos)
                if dist_to_prev < (plot_width * 0.02):
                    level_toggle = (level_toggle + 1) % 2
                else:
                    level_toggle = 0

                y_text_pos = y_levels[level_toggle]
                last_x_pos = mu

                # Le texte est maintenant formaté sur une seule ligne (puisqu'il sera tourné)
                label_txt = f"{row['Line']} | v: {vel_val:.1f} | SNR: {snr_val:.1f}"
                peak_y = cont_local[np.argmin(np.abs(x_mod - mu))] + amp

                ax.annotate(label_txt,
                            xy=(mu, peak_y),
                            xytext=(mu, y_text_pos),
                            arrowprops=dict(arrowstyle='-', color=box_color, alpha=0.3),
                            fontsize=9, color=box_color, ha='center', va='bottom', rotation=90,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=box_color, alpha=0.8, zorder=20))

        ax.set_xlim(c_min, c_max)
        ax.grid(True, alpha=0.2)

        # Ajout d'une petite légende explicative dans le premier panel uniquement
        if i == 0:
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='gray', lw=1.2, ls='--'),
                            Line2D([0], [0], color='black', lw=1.2, ls='-')]
            ax.legend(custom_lines, [r'Théorique ($\lambda_{th}$)', r'Observé ($\lambda_{obs}$)'], loc='upper left')

    axes[-1].set_xlabel(r"Wavelength $\lambda$ [$\mu m$]")
    plt.tight_layout()
    return fig, axes



def plot_spectral_summary_cleanV3(df_spectrum, df_merged, type_plot="step",
                                wl_range=(4.576, 4.925), n_plots=3, ylim=(100, 200),
                                snrtype='snr2', snr_threshold=3.0, v_range=(-10, 25),
                                clean_mode=False, alpha=0.9):
    """
    Trace le résumé spectral avec :
    1. Fits coupés proprement.
    2. Continuum corrigé (y = mx+c global).
    3. Tracé des modèles DOUBLES sans doublons.
    4. Code couleur Doppler : Rouge (Shift positif) / Bleu (Shift négatif).
    5. Étiquettes verticales.
    """
    wl_min, wl_max = wl_range
    delta_wl = (wl_max - wl_min) / n_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(20, 7 * n_plots))
    if n_plots == 1:
        axes = [axes]

    N_SIGMA_CUT = 4.5

    for i, ax in enumerate(axes):
        ax.set_ylim(ylim)

        c_min = wl_min + i * delta_wl
        c_max = wl_min + (i + 1) * delta_wl
        plot_width = c_max - c_min

        # 1. Tracé spectre
        mask_view = (df_spectrum["Wavelength"] >= c_min) & (df_spectrum["Wavelength"] <= c_max)
        if mask_view.sum() == 0:
            continue

        wave_view = df_spectrum.loc[mask_view, "Wavelength"].values
        flux_view = df_spectrum.loc[mask_view, "Int"].values
        
        if type_plot == "plot": 
            ax.plot(wave_view, flux_view, color="black", lw=2, alpha=0.9, zorder=1)
        elif type_plot == "step": 
            ax.step(wave_view, flux_view, color="black", lw=2, alpha=0.9, zorder=1)
            
        # 2. Filtrage et Tri
        subset = df_merged[(df_merged['wavelength_obs'] >= c_min) & (df_merged['wavelength_obs'] <= c_max)]
        subset = subset.sort_values(by='wavelength_obs')

        # Variables de suivi
        last_x_pos = -999
        level_toggle = 0
        y_range = ylim[1] - ylim[0]
        y_levels = [ylim[0] + 0.85 * y_range, ylim[0] + 0.65 * y_range]  # Ajusté pour le texte vertical
        
        # Pour éviter de tracer 2 fois le modèle complet des DOUBLE
        plotted_pairs = set()

        for idx, (_, row) in enumerate(subset.iterrows()):
            fit_type = row['fit_type']
            mu, sigma, amp = row['wavelength_obs'], row['sigma'], row['amplitude']
            lth = row['wavelength_th']
            m, c = row['continuum_m'], row['continuum_c']

            # --- CALCUL DE LA FENÊTRE ---
            if fit_type == 'BLEND':
                mu2, sig2 = row['mu_blend'], row['sig_blend']
                start_x = min(mu - N_SIGMA_CUT*sigma, mu2 - N_SIGMA_CUT*sig2)
                end_x = max(mu + N_SIGMA_CUT*sigma, mu2 + N_SIGMA_CUT*sig2)
            elif fit_type == 'DOUBLE':
                try:
                    comp = df_merged[(df_merged['Line'] == row['blended_with']) & (df_merged['fit_type'] == 'DOUBLE')].iloc[0]
                    mu2, sig2 = comp['wavelength_obs'], comp['sigma']
                    start_x = min(mu - N_SIGMA_CUT*sigma, mu2 - N_SIGMA_CUT*sig2)
                    end_x = max(mu + N_SIGMA_CUT*sigma, mu2 + N_SIGMA_CUT*sig2)
                except:
                    start_x, end_x = mu - N_SIGMA_CUT * sigma, mu + N_SIGMA_CUT * sigma
            else:
                start_x, end_x = mu - N_SIGMA_CUT * sigma, mu + N_SIGMA_CUT * sigma

            x_mod = np.linspace(start_x, end_x, 150)

            # --- RECONSTRUCTION DU MODÈLE CORRIGÉ ---
            # Le continuum est strictement mx+c pour tout le monde maintenant
            cont_local = m * x_mod + c

            g_main = amp * np.exp(-(x_mod - mu)**2 / (2 * sigma**2))
            g_blend = np.zeros_like(x_mod)
            g_compagnon = np.zeros_like(x_mod)

            if fit_type == 'BLEND':
                g_blend = row['amp_blend'] * np.exp(-(x_mod - row['mu_blend'])**2 / (2 * row['sig_blend']**2))
            elif fit_type == 'DOUBLE':
                try:
                    comp = df_merged[(df_merged['Line'] == row['blended_with']) & (df_merged['fit_type'] == 'DOUBLE')].iloc[0]
                    g_compagnon = comp['amplitude'] * np.exp(-(x_mod - comp['wavelength_obs'])**2 / (2 * comp['sigma']**2))
                except:
                    pass

            total_model = cont_local + g_main + g_blend + g_compagnon

            # --- GESTION COULEUR GLOBALE ---
            if clean_mode:
                base_color, sec_color = "darkgreen", "darkorange"
            else:
                if fit_type == 'DOUBLE': base_color, sec_color = 'purple', 'orchid'
                elif fit_type == 'BLEND': base_color, sec_color = 'blue', 'teal'
                elif fit_type == 'FORCED': base_color, sec_color = 'magenta', 'magenta'
                else: base_color, sec_color = 'crimson', 'crimson'

            # --- TRACÉ DES COURBES ---
            ax.plot(x_mod, cont_local, ls=':', color='gray', lw=1.2, alpha=0.5, zorder=2)
            
            # Trace la composante individuelle (pour que chaque raie ait son pic en pointillé)
            ax.plot(x_mod, cont_local + g_main, ls='--', color=sec_color, lw=1.5, alpha=0.6, zorder=3)
            if fit_type == 'BLEND':
                ax.plot(x_mod, cont_local + g_blend, ls=':', color=sec_color, lw=1.2, alpha=0.6, zorder=3)

            # Traçage du modèle TOTAL (avec sécurité anti-doublon pour les DOUBLE)
            if fit_type == 'DOUBLE':
                pair_key = frozenset([row['Line'], row.get('blended_with', None)])
                if pair_key not in plotted_pairs:
                    ax.plot(x_mod, total_model, ls='-', color=base_color, lw=2.5, alpha=alpha, zorder=5)
                    plotted_pairs.add(pair_key)
            else:
                ax.plot(x_mod, total_model, ls='-', color=base_color, lw=2.5, alpha=alpha, zorder=5)

            # --- MISE EN ÉVIDENCE DU SHIFT DOPPLER (ROUGE / BLEU) ---
            delta_lambda = mu - lth
            shift_color = 'salmon' if delta_lambda > 0 else 'cornflowerblue' # Rouge si s'éloigne, Bleu si se rapproche
            
            ax.axvspan(lth, mu, color=shift_color, alpha=0.15, zorder=0)
            
            # Ligne Théorique (Pointillés gris)
            ax.axvline(lth, color='gray', ls='--', alpha=0.7, lw=1.2, zorder=1)
            # Ligne Observée (Continue couleur raie)
            ax.axvline(mu, color=base_color, ls='-', alpha=0.7, lw=1.2, zorder=1)

            # --- ANNOTATIONS VERTICALES ---
            if not clean_mode:
                snr_val = row.get(snrtype, 0)
                vel_val = row.get('Velocity_km_s', 0)

                is_bad = (snr_val < snr_threshold) or (vel_val < v_range[0]) or (vel_val > v_range[1])
                box_color = "red" if is_bad else "black"

                # Logique anti-chevauchement (ajustée pour des textes verticaux plus fins)
                dist_to_prev = abs(mu - last_x_pos)
                if dist_to_prev < (plot_width * 0.02):
                    level_toggle = (level_toggle + 1) % 2
                else:
                    level_toggle = 0

                y_text_pos = y_levels[level_toggle]
                last_x_pos = mu

                # Le texte est maintenant formaté sur une seule ligne (puisqu'il sera tourné)
                label_txt = f"{row['Line']} | v: {vel_val:.1f} | SNR: {snr_val:.1f}"
                peak_y = cont_local[np.argmin(np.abs(x_mod - mu))] + amp

                ax.annotate(label_txt,
                            xy=(mu, peak_y),
                            xytext=(mu, y_text_pos),
                            arrowprops=dict(arrowstyle='-', color=box_color, alpha=0.3),
                            fontsize=9, color=box_color, ha='center', va='bottom', rotation=90,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=box_color, alpha=0.8, zorder=20))

        ax.set_xlim(c_min, c_max)
        ax.grid(True, alpha=0.2)

        # Ajout d'une petite légende explicative dans le premier panel uniquement
        if i == 0:
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='gray', lw=1.2, ls='--'),
                            Line2D([0], [0], color='black', lw=1.2, ls='-')]
            ax.legend(custom_lines, [r'Théorique ($\lambda_{th}$)', r'Observé ($\lambda_{obs}$)'], loc='upper left')

    axes[-1].set_xlabel(r"Wavelength $\lambda$ [$\mu m$]")
    plt.tight_layout()
    return fig, axes


def plot_velocity_stacking_visible(df_spectrum, df_fits, Vchoice=10.0, window_kms=400):
    """
    Trace toutes les raies superposées en vitesse relative.
    Affiche les données réelles (pas le fit) avec le continuum soustrait.
    """
    c_light = 299792.458 # km/s
    
    # Création de la figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Listes pour gérer l'ordre d'affichage (Z-order)
    high_velocity_lines = []
    
    # 1. Boucle sur chaque raie détectée
    for idx, row in df_fits.iterrows():
        mu_obs = row['wavelength_obs']
        v_obs = row['Velocity_km_s']
        line_name = row['Line']
        
        # --- A. Retrouver la longueur d'onde de repos (Lambda Théorique) ---
        # C'est notre "0 km/s" de référence pour CETTE raie
        lam_rest = mu_obs / (1 + (v_obs / c_light))
        
        # --- B. Découper le spectre autour de la raie ---
        # On prend une fenêtre large (window_kms) pour voir les ailes
        delta_lam_window = lam_rest * (window_kms / c_light)
        
        mask = (df_spectrum["Wavelength"] >= lam_rest - delta_lam_window) & \
               (df_spectrum["Wavelength"] <= lam_rest + delta_lam_window)
        
        if mask.sum() < 3: continue # Pas assez de points
            
        slice_wave = df_spectrum.loc[mask, "Wavelength"].values
        slice_flux = df_spectrum.loc[mask, "Int"].values
        
        # --- C. Soustraire le continuum local (modèle linéaire m*x + c) ---
        # Cela permet de comparer les hauteurs de raies (MJy/sr) partant de 0
        local_continuum = row['continuum_m'] * slice_wave + row['continuum_c']
        flux_norm = slice_flux - local_continuum
        
        # --- D. Convertir l'axe X en Vitesse (km/s) ---
        # v = c * (lambda - lambda_rest) / lambda_rest
        velocity_axis = ((slice_wave - lam_rest) / lam_rest) * c_light
        
        # --- E. Plotting conditionnel ---
        is_fast = abs(v_obs) > Vchoice
        
        if is_fast:
            # On stocke pour tracer EN DERNIER (au-dessus) et annoter
            # On cherche le pic réel dans les données pour pointer la flèche
            peak_idx = np.argmax(flux_norm)
            peak_v = velocity_axis[peak_idx]
            peak_flux = flux_norm[peak_idx]
            
            high_velocity_lines.append({
                'x': velocity_axis, 
                'y': flux_norm, 
                'name': line_name, 
                'v_mesure': v_obs,
                'peak_v': peak_v,       # Coordonnée X du sommet réel
                'peak_flux': peak_flux  # Coordonnée Y du sommet réel
            })
        else:
            # Raie "Standard" : On trace tout de suite en gris/bleu
            # alpha=0.35 permet de voir la superposition si 40 raies s'empilent
            ax.plot(velocity_axis, flux_norm, color='SlateGray', alpha=0.35, lw=1.5, zorder=1)

    # 2. Tracé des raies rapides (en ROUGE, par-dessus)
    # On décale les textes verticalement pour éviter qu'ils ne se marchent dessus
    text_offset_cycle = [20, 40, 60, 80] # Décalage en pixels (points)
    
    for i, item in enumerate(high_velocity_lines):
        # La courbe
        ax.plot(item['x'], item['y'], color='Crimson', alpha=0.9, lw=2.5, zorder=5, label='High V' if i==0 else "")
        
        # L'annotation (Flèche)
        offset = text_offset_cycle[i % len(text_offset_cycle)]
        
        # Texte informatif
        label_txt = f"{item['name']}\n{item['v_mesure']:.1f} km/s"
        
        ax.annotate(label_txt,
                    xy=(item['peak_v'], item['peak_flux']), # Pointe le sommet
                    xytext=(10, offset),                    # Position du texte (relatif)
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color='DarkRed', lw=1.5),
                    color='DarkRed', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="Crimson", alpha=0.8))

    # 3. Mise en forme finale
    ax.set_xlabel(r"Vitesse relative  $\lambda_{theorique}$ [km/s]", fontsize=14)
    ax.set_ylabel("Flux (Continuum soustrait) [MJy/sr]", fontsize=14)
    ax.set_title(f"Profils de vitesse superposés ({len(df_fits)} raies)", fontsize=16)
    
    # Lignes guides
    ax.axvline(0, color='black', linestyle='--', alpha=0.8, lw=1, label='Centre théorique')
    ax.axvline(Vchoice, color='red', linestyle=':', alpha=0.5)
    ax.axvline(-Vchoice, color='red', linestyle=':', alpha=0.5, label=f'Seuil +/- {Vchoice} km/s')
    
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()













import matplotlib.cm as cm

def plot_spectral_stacking_cmap(df_spectrum, df_fits, window_kms=100):
    """
    Trace toutes les raies superposées, centrées sur 0 km/s.
    Chaque raie a une couleur unique tirée d'une colormap.
    """
    c_light = 299792.458 # km/s
    
    # Préparation de la figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Création de la palette de couleurs (autant de couleurs que de raies)
    num_lines = len(df_fits)
    colors = cm.nipy_spectral(np.linspace(0, 1, num_lines))
    
    # Boucle sur les raies
    for idx, row in df_fits.iterrows():
        line_name = row['Line']
        mu_obs = row['wavelength_obs']
        v_obs = row['Velocity_km_s']
        
        # 1. Calcul du Lambda au repos (pour centrer à 0)
        lam_rest = mu_obs / (1 + (v_obs / c_light))
        
        # 2. Fenêtre d'extraction
        delta_lam = lam_rest * (window_kms / c_light)
        mask = (df_spectrum["Wavelength"] >= lam_rest - delta_lam) & \
               (df_spectrum["Wavelength"] <= lam_rest + delta_lam)
        
        if mask.sum() < 5: continue
            
        slice_wave = df_spectrum.loc[mask, "Wavelength"].values
        slice_flux = df_spectrum.loc[mask, "Int"].values
        
        # 3. Soustraction du continuum
        continuum = row['continuum_m'] * slice_wave + row['continuum_c']
        flux_norm = slice_flux - continuum
        
        # 4. Axe X en vitesse
        velocity_axis = ((slice_wave - lam_rest) / lam_rest) * c_light
        
        # 5. Tracé
        ax.plot(velocity_axis, flux_norm, 
                color=colors[idx], 
                alpha=0.8, 
                lw=1.5, 
                label=f"{line_name}")

    # Mise en forme
    ax.set_xlabel("Vitesse relative [km/s]", fontsize=14)
    ax.set_ylabel("Flux (Continuum soustrait) [MJy/sr]", fontsize=14)
    ax.set_title("Superposition des profils de raies", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linestyle=':', alpha=0.5)

    # Légende à l'extérieur (à droite)
    # ncol=2 permet de ranger les noms sur 2 colonnes si la liste est longue
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., ncol=1, fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_spectral_stacking_stats_vertical(df_spectrum, df_fits, window_kms=100):
    c_light = 299792.458 

    velocities = df_fits['Velocity_km_s'].values
    v_mean = np.nanmean(velocities)
    v_med = np.nanmedian(velocities)
    v_q1 = np.nanpercentile(velocities, 25)
    v_q3 = np.nanpercentile(velocities, 75)

    fig, ax = plt.subplots(figsize=(14, 8))
    
    num_lines = len(df_fits)
    colors = cm.nipy_spectral(np.linspace(0, 1, num_lines))
    
    for idx, (_, row) in enumerate(df_fits.iterrows()):
        line_name = row['Line']
        mu_obs = row['wavelength_obs']
        v_obs = row['Velocity_km_s']
        
        lam_rest = mu_obs / (1 + (v_obs / c_light))
        
        delta_lam = lam_rest * (window_kms / c_light) 
        mask = (df_spectrum["Wavelength"] >= lam_rest - delta_lam) & \
               (df_spectrum["Wavelength"] <= lam_rest + delta_lam)
        
        if mask.sum() < 5: continue
            
        slice_wave = df_spectrum.loc[mask, "Wavelength"].values
        slice_flux = df_spectrum.loc[mask, "Int"].values
        
        continuum = row['continuum_m'] * slice_wave + row['continuum_c']
        flux_norm = slice_flux - continuum
        
        velocity_axis = ((slice_wave - lam_rest) / lam_rest) * c_light
        
        # --- AJOUT: Détection et plot du pic max ---
        if len(flux_norm) > 0:
            idx_max = np.argmax(flux_norm)
            v_peak = velocity_axis[idx_max]
            f_peak = flux_norm[idx_max]
            
            # s=40 définit la taille du point, zorder=3 assure qu'il est au-dessus de la ligne
            ax.scatter(v_peak, f_peak, color=colors[idx], s=40, zorder=3)
        # -------------------------------------------

        ax.plot(velocity_axis, flux_norm, 
                color=colors[idx], 
                alpha=0.6, 
                lw=1.5, 
                label=f"{line_name}")

    ax.axvline(v_mean, color='black', linestyle='-', lw=2.5)
    ax.axvline(v_med, color='blue', linestyle='--', lw=2.5)
    ax.axvline(v_q1, color='green', linestyle=':', lw=2)
    ax.axvline(v_q3, color='green', linestyle=':', lw=2)
    ax.axvspan(v_q1, v_q3, color='green', alpha=0.1)

    stats_text = (f"Average : {v_mean:.1f} km/s\n"
                  f"Median : {v_med:.1f} km/s\n"
                  f"Q1 (25pc) : {v_q1:.1f} km/s\n"
                  f"Q3 (75pc) : {v_q3:.1f} km/s")

    ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))

    ax.set_xlabel("Velocity [km/s]", fontsize=14)
    ax.set_ylabel("Flux (Continuum substracted) [MJy/sr]", fontsize=14)
    #ax.set_title("Superposition des profils et Statistiques de Vitesse", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(-window_kms, window_kms)

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., ncol=2, fontsize=10)
    
    plt.tight_layout()
    return fig,ax


def plot_gaussian_stacking_stats_vertical(df_fits, window_kms=100):
    c_light = 299792.458 

    velocities = df_fits['Velocity_km_s'].values
    v_mean = np.nanmean(velocities)
    v_med = np.nanmedian(velocities)
    v_q1 = np.nanpercentile(velocities, 25)
    v_q3 = np.nanpercentile(velocities, 75)

    fig, ax = plt.subplots(figsize=(14, 8))
    
    num_lines = len(df_fits)
    colors = cm.nipy_spectral(np.linspace(0, 1, num_lines))
    
    # Création d'un axe de vitesse synthétique régulier pour une belle courbe
    velocity_axis = np.linspace(-window_kms, window_kms, 500)
    
    for idx, (_, row) in enumerate(df_fits.iterrows()):
        line_name = row['Line']
        mu_obs = row['wavelength_obs']
        v_obs = row['Velocity_km_s']
        amp = row['amplitude']
        sigma = row['sigma']
        
        # Calcul de la longueur d'onde de repos
        lam_rest = mu_obs / (1 + (v_obs / c_light))
        
        # Conversion de l'axe de vitesse synthétique en longueurs d'onde
        slice_wave = lam_rest * (1 + velocity_axis / c_light)
        
        # Calcul du profil gaussien (déjà soustrait du continuum par définition)
        flux_norm = amp * np.exp(-(slice_wave - mu_obs)**2 / (2 * sigma**2))
        
        # --- Détection et plot du pic max ---
        if len(flux_norm) > 0:
            idx_max = np.argmax(flux_norm)
            v_peak = velocity_axis[idx_max]
            f_peak = flux_norm[idx_max]
            
            # Point sur le pic de la gaussienne
            ax.scatter(v_peak, f_peak, color=colors[idx], s=40, zorder=3)
        # -------------------------------------------

        # Tracé de la gaussienne propre
        ax.plot(velocity_axis, flux_norm, 
                color=colors[idx], 
                alpha=0.6, 
                lw=1.5, 
                label=f"{line_name}")

    ax.axvline(v_mean, color='black', linestyle='-', lw=2.5)
    ax.axvline(v_med, color='blue', linestyle='--', lw=2.5)
    ax.axvline(v_q1, color='green', linestyle=':', lw=2)
    ax.axvline(v_q3, color='green', linestyle=':', lw=2)
    ax.axvspan(v_q1, v_q3, color='green', alpha=0.1)

    stats_text = (f"Average : {v_mean:.1f} km/s\n"
                  f"Median : {v_med:.1f} km/s\n"
                  f"Q1 (25pc) : {v_q1:.1f} km/s\n"
                  f"Q3 (75pc) : {v_q3:.1f} km/s")

    ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))

    ax.set_xlabel("Velocity [km/s]", fontsize=14)
    ax.set_ylabel("Flux Gaussian Model [MJy/sr]", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(-window_kms, window_kms)

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., ncol=2, fontsize=10)
    
    plt.tight_layout()
    return fig, ax



import plotly.graph_objects as go
def plot_spectral_stacking_plotly(df_spectrum, df_fits, window_kms=100):
    c_light = 299792.458 

    # --- 1. Calcul des statistiques ---
    velocities = df_fits['Velocity_km_s'].values
    v_mean = np.nanmean(velocities)
    v_med = np.nanmedian(velocities)
    v_q1 = np.nanpercentile(velocities, 25)
    v_q3 = np.nanpercentile(velocities, 75)
    
    # Texte pour la boîte d'annotation
    stats_text = (f"<b>Statistiques de Vitesse</b><br>"
                  f"Moyenne : {v_mean:.1f} km/s<br>"
                  f"Médiane : {v_med:.1f} km/s<br>"
                  f"Q1 (25%) : {v_q1:.1f} km/s<br>"
                  f"Q3 (75%) : {v_q3:.1f} km/s")

    fig = go.Figure()

    # --- 2. Zone colorée (Q1-Q3) et Lignes verticales ---
    
    # Zone Q1-Q3
    fig.add_vrect(
        x0=v_q1, x1=v_q3,
        fillcolor="green", opacity=0.1,
        layer="below", line_width=0,
    )

    # Lignes verticales (Moyenne, Médiane, Q1, Q3)
    fig.add_vline(x=v_mean, line_width=3, line_dash="solid", line_color="black")
    fig.add_vline(x=v_med, line_width=3, line_dash="dash", line_color="blue")
    fig.add_vline(x=v_q1, line_width=2, line_dash="dot", line_color="green")
    fig.add_vline(x=v_q3, line_width=2, line_dash="dot", line_color="green")

    # --- 3. Boucle sur les raies (Tracés) ---
    num_lines = len(df_fits)
    # Palette de couleurs
    colors = pc.sample_colorscale('Turbo', [i/(num_lines-1) for i in range(num_lines)])

    for idx, (_, row) in enumerate(df_fits.iterrows()):
        line_name = row['Line']
        mu_obs = row['wavelength_obs']
        v_obs = row['Velocity_km_s']
        
        # Calculs physiques
        lam_rest = mu_obs / (1 + (v_obs / c_light))
        delta_lam = lam_rest * (window_kms / c_light) 
        
        # Masque
        mask = (df_spectrum["Wavelength"] >= lam_rest - delta_lam) & \
               (df_spectrum["Wavelength"] <= lam_rest + delta_lam)
        
        if mask.sum() < 5: continue
            
        slice_wave = df_spectrum.loc[mask, "Wavelength"].values
        slice_flux = df_spectrum.loc[mask, "Int"].values
        
        # Continuum et Normalisation
        continuum = row['continuum_m'] * slice_wave + row['continuum_c']
        flux_norm = slice_flux - continuum
        velocity_axis = ((slice_wave - lam_rest) / lam_rest) * c_light
        
        # --- AJOUT: Détection du pic max ---
        if len(flux_norm) > 0:
            idx_max = np.argmax(flux_norm)
            v_peak = velocity_axis[idx_max]
            f_peak = flux_norm[idx_max]
            
            # --- AJOUT: Tracé du Point Max (Marker) ---
            # On le fait avant ou après la ligne, peu importe, mais on cache la légende
            # pour ne pas avoir de doublons.
            fig.add_trace(go.Scatter(
                x=[v_peak],
                y=[f_peak],
                mode='markers',
                marker=dict(color=colors[idx], size=10, symbol='circle'), # Point assez gros
                showlegend=False, # Pas de doublon dans la légende
                hoverinfo='skip'  # On laisse le hover de la ligne gérer l'info ou on met le même
            ))

        # Ajout de la courbe
        fig.add_trace(go.Scatter(
            x=velocity_axis,
            y=flux_norm,
            mode='lines',
            name=line_name,
            line=dict(color=colors[idx], width=1.5),
            opacity=0.7,
            # Le hovertemplate assure que l'étiquette s'affiche partout
            hovertemplate=f"<b>{line_name}</b><br>Vit: %{{x:.1f}} km/s<br>Flux: %{{y:.2e}}<extra></extra>"
        ))

    # --- 4. Annotation (Boîte de stats) ---
    fig.add_annotation(
        x=0.02, y=0.98, 
        xref="paper", yref="paper",
        text=stats_text,
        showarrow=False,
        align="left",
        font=dict(size=12, color="black"),
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black",
        borderwidth=1,
        borderpad=5
    )

    # --- 5. Mise en forme (Layout) ---
    fig.update_layout(
        title="<b>Superposition des profils et Statistiques de Vitesse</b>",
        xaxis_title="Vitesse [km/s]",
        yaxis_title="Flux (Continuum soustrait) [MJy/sr]",
        template="simple_white",
        
        # 'closest' est le mieux pour voir l'étiquette spécifique de la courbe survolée
        hovermode="closest", 
        
        xaxis=dict(range=[-window_kms, window_kms], showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),
        height=700,
        width=1100,
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bordercolor="Black",
            borderwidth=1,
        )
    )

    fig.show()

def plot_spectral_stacking_interactive_highlight(df_spectrum, df_fits, window_kms=1200):
    """
    Trace les raies en semi-transparent.
    Au survol de la souris, la raie entière devient opaque (alpha=1).
    NÉCESSITE UN ENVIRONNEMENT JUPYTER (Notebook/Lab).
    """
    c_light = 299792.458
    
    # --- Paramètres visuels ---
    OPACITY_GHOST = 0.15  # Transparence par défaut
    OPACITY_FULL = 1.0    # Opacité au survol
    WIDTH_GHOST = 2       # Épaisseur par défaut
    WIDTH_FULL = 4        # Épaisseur au survol (pour mieux ressortir)

    # On utilise FigureWidget au lieu de Figure pour l'interaction Python
    fig = go.FigureWidget()

    # Palette de couleurs vives
    colors = [f"hsl({h},85%,55%)" for h in np.linspace(0, 360, len(df_fits) + 1)]

    # --- 1. Création des traces (Boucle standard) ---
    print("Génération des courbes...")
    for idx, row in df_fits.iterrows():
        line_name = row['Line']
        mu_obs = row['wavelength_obs']
        v_obs = row['Velocity_km_s']
        
        # Calculs standard
        lam_rest = mu_obs / (1 + (v_obs / c_light))
        delta_lam = lam_rest * (window_kms / c_light)
        mask = (df_spectrum["Wavelength"] >= lam_rest - delta_lam) & \
               (df_spectrum["Wavelength"] <= lam_rest + delta_lam)
        
        if mask.sum() < 3: continue
            
        slice_wave = df_spectrum.loc[mask, "Wavelength"].values
        slice_flux = df_spectrum.loc[mask, "Int"].values
        continuum = row['continuum_m'] * slice_wave + row['continuum_c']
        flux_norm = slice_flux - continuum
        velocity_axis = ((slice_wave - lam_rest) / lam_rest) * c_light
        
        # Ajout de la trace en mode "GHOST" par défaut
        fig.add_trace(go.Scatter(
            x=velocity_axis,
            y=flux_norm,
            mode='lines',
            name=line_name,
            opacity=OPACITY_GHOST,    # <--- Transparent au début
            line=dict(color=colors[idx], width=WIDTH_GHOST),
            # Info-bulle simple
            hovertemplate=f"<b>{line_name}</b><br>V: %{{x:.1f}}<br>I: %{{y:.2e}}<extra></extra>"
        ))

    # --- 2. Définition des fonctions de Callback (La magie Python) ---
    
    # Fonction appelée quand la souris ENTRÉE sur une courbe
    def hover_fn(trace, points, selector):
        # 'trace' est l'objet survolé. On récupère son index dans la figure.
        # L'index est stocké dans une propriété interne _index par FigureWidget
        hovered_idx = trace._index
        
        # batch_update bloque le rafraîchissement pour faire toutes les modifs d'un coup
        with fig.batch_update():
            # On parcourt TOUTES les courbes de la figure
            for i, t in enumerate(fig.data):
                if i == hovered_idx:
                    # C'est la courbe survolée : ON ALLUME
                    t.opacity = OPACITY_FULL
                    t.line.width = WIDTH_FULL
                else:
                    # Ce n'est pas elle : ON ÉTEINT (Ghost mode)
                    t.opacity = OPACITY_GHOST
                    t.line.width = WIDTH_GHOST

    # Fonction appelée quand la souris SORT du graphique
    # Optionnel : remet tout en "ghost" si on quitte la zone
    def unhover_fn(trace, points, selector):
         with fig.batch_update():
            for t in fig.data:
                t.opacity = OPACITY_GHOST
                t.line.width = WIDTH_GHOST

    # --- 3. Connexion des Callbacks aux Traces ---
    print("Activation de l'interactivité...")
    for trace in fig.data:
        trace.on_hover(hover_fn)
        # Décommentez la ligne suivante si vous voulez que tout redevienne transparent quand la souris sort
        # trace.on_unhover(unhover_fn) 

    # --- 4. Mise en forme finale ---
    fig.update_layout(
        title="Survolez une raie pour la mettre en surbrillance",
        xaxis_title="Vitesse relative [km/s]",
        yaxis_title="Flux (Continuum soustrait)",
        template="plotly_dark",
        hovermode="closest",
        showlegend=False, # On cache la légende pour ne pas gêner l'interaction visuelle
        width=1100, height=700
    )
    fig.add_vline(x=0, line_color="white", line_dash="dash", opacity=0.5)

    # IMPORTANT AVEC FIGUREWIDGET :
    # On ne fait pas fig.show(), on retourne l'objet fig lui-même.
    # Jupyter se charge de l'afficher interactivement en bas de cellule.
    return fig
    


def plot_local_noise(df_noise, noise_rms, col_wave="Wavelength", col_flux="Int"):
    """
    Trace la région de bruit, le continuum affine ajusté, et une illustration 
    visuelle (bandes) du niveau de bruit RMS.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = df_noise[col_wave].values
    y = df_noise[col_flux].values
    continuum = df_noise['Continuum_Fit'].values
    
    # 1. Tracé des données observées
    ax.step(x, y, where='mid', color='black', label='Données (Bruit pur)', linewidth=1.5, alpha=0.8)
    
    # 2. Tracé du continuum linéaire
    ax.plot(x, continuum, color='blue', linestyle='--', linewidth=2, label='Continuum fitté $C(\lambda)$')
    
    # 3. Illustration visuelle du bruit (Bandes 1 sigma et 3 sigma)
    ax.fill_between(x, continuum - noise_rms, continuum + noise_rms, 
                    color='blue', alpha=0.3, label=r'$\pm 1 \sigma_{bruit}$')
    ax.fill_between(x, continuum - 3*noise_rms, continuum + 3*noise_rms, 
                    color='blue', alpha=0.1, label=r'$\pm 3 \sigma_{bruit}$')
    
    # 4. Esthétique et Annotations
    ax.set_xlabel("Wavelength [$\mu$m]", fontsize=14)
    ax.set_ylabel("Surface Brightness [MJy/sr]", fontsize=14)
    ax.set_title("Illustration de l'estimation du Bruit Local", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=12)
    
    stats_text = f"RMS ($\sigma_{{bruit}}$) = {noise_rms:.3f}\nN points = {len(x)}"
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plot_snr_distribution(df_clean, col_wave='wavelength_th', col_snr='SNR_noise', col_line='Line', snr_threshold=3.0):
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Séparation visuelle : Bon SNR vs Mauvais SNR
    mask_good = df_clean[col_snr] >= snr_threshold
    df_good = df_clean[mask_good]
    df_bad = df_clean[~mask_good]
    
    # Tracé des points
    ax.scatter(df_good[col_wave], df_good[col_snr], color='forestgreen', s=60, zorder=3,)# label='Détection robuste')
    ax.scatter(df_bad[col_wave], df_bad[col_snr], color='crimson', s=60, marker='X', zorder=3,)# label='Détection marginale / Bruit')
    
    # Ligne de seuil et zone de rejet
    ax.axhline(snr_threshold, color='red', linestyle='--', linewidth=2, label=f' SNR threshold = {snr_threshold}')
    ax.axhspan(0, snr_threshold, color='red', alpha=0.1, zorder=1)
    
    # Ajout des étiquettes (Noms des raies)
    for _, row in df_clean.iterrows():
        ax.annotate(row[col_line], 
                    (row[col_wave], row[col_snr]),
                    textcoords="offset points", 
                    xytext=(1, 9), # Décalage léger vers le haut
                    ha='center', 
                    fontsize=10,
                    color='black' if row[col_snr] >= snr_threshold else 'darkred')
        
    # Esthétique
    ax.set_xlabel(r"Wavelength[$\mu$m]", fontsize=25)
    ax.set_ylabel("Signal-to-Noise Ratio (SNR)", fontsize=25)
    
    # On force l'axe Y à commencer à 0 pour bien voir la zone de bruit
    ax.set_ylim(bottom=0, top=max(df_clean[col_snr].max() * 1.15, 10)) 
    
    ax.grid(True, linestyle=':', alpha=0.6, zorder=0)
    ax.legend(loc='upper right', fontsize=20)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax











def plot_smart_gallery_from_mergedV3(df_spectrum, df_merged, n_cols=6, window_view=0.005, clean_mode=False, style_plot="plot"):
    """
    Trace une galerie de vignettes (une par raie ou par paire) à partir du DataFrame fusionné.
    Gère les fits SIMPLES, FORCÉS, BLENDS, DOUBLES et UNRESOLVED.
    """
    if df_merged.empty:
        print("Aucune donnée à tracer.")
        return

    # 1. Regroupement des raies DOUBLE et UNRESOLVED pour ne pas créer deux subplots
    processed_pairs = set()
    items_to_plot = [] 

    for idx, row in df_merged.iterrows():
        if row['fit_type'] in ['DOUBLE', 'UNRESOLVED']:
            line1 = row['Line']
            line2 = row.get('blended_with', None)
            pair_key = frozenset([line1, line2])
            
            if pair_key not in processed_pairs:
                try:
                    row2 = df_merged[(df_merged['Line'] == line2) & (df_merged['fit_type'] == row['fit_type'])].iloc[0]
                    items_to_plot.append((row['fit_type'], row, row2))
                except:
                    items_to_plot.append(('SINGLE', row))
                processed_pairs.add(pair_key)
        else:
            items_to_plot.append(('SINGLE', row))

    n_plots = len(items_to_plot)
    if n_plots == 0:
        return

    n_rows = math.ceil(n_plots / n_cols)
    h_ratio = 3 if clean_mode else 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, h_ratio * n_rows))

    # Réduire globalement la taille de la police des axes (X et Y)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)

    if n_plots == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    UNIFORM_COLOR = 'tab:blue'

    for i, item in enumerate(items_to_plot):
        ax = axes[i]
        item_type = item[0]

        # --- GESTION DES FENÊTRES (ZOOM FORCÉ) ---
        if item_type == 'SINGLE':
            row = item[1]
            mu, sigma = row['wavelength_obs'], row['sigma']
            lth = row['wavelength_th']
            w_min = mu - max(window_view/2, 4*sigma)
            w_max = mu + max(window_view/2, 4*sigma)
            title_name = f"{row['Line']} ({row['fit_type']})"
            
        else: # DOUBLE ou UNRESOLVED
            r1, r2 = item[1], item[2]
            mu1, mu2 = r1['wavelength_obs'], r2['wavelength_obs']
            sig1, sig2 = r1['sigma'], r2['sigma']
            lth1, lth2 = r1['wavelength_th'], r2['wavelength_th']
            
            w_min = min(mu1, mu2) - max(window_view/4, 3*max(sig1, sig2))
            w_max = max(mu1, mu2) + max(window_view/4, 3*max(sig1, sig2))
            title_name = f"{r1['Line']} and {r2['Line']} ({item_type})"

        # --- EXTRACTION SPECTRE LOCAL ---
        mask = (df_spectrum["Wavelength"] >= w_min) & (df_spectrum["Wavelength"] <= w_max)
        wave_local = df_spectrum.loc[mask, "Wavelength"].values
        flux_local = df_spectrum.loc[mask, "Int"].values

        if len(wave_local) == 0:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
            continue

        if style_plot == "plot": 
            ax.plot(wave_local, flux_local, c='black', lw=1.5, alpha=0.8, marker='o', markersize=2)
        elif style_plot == "step": 
            ax.step(wave_local, flux_local, c='black', lw=1.5, alpha=0.8, where='mid')

        # Grille pour le modèle
        x_plot = np.linspace(wave_local.min(), wave_local.max(), 200)

        # --- CALCUL DES MODÈLES ---
        if item_type == 'SINGLE':
            cont = row['continuum_m'] * x_plot + row['continuum_c']
            g_main = row['amplitude'] * np.exp(-(x_plot - mu)**2 / (2 * sigma**2))
            g_sec = np.zeros_like(x_plot)
            if row['fit_type'] == 'BLEND':
                g_sec = row['amp_blend'] * np.exp(-(x_plot - row['mu_blend'])**2 / (2 * row['sig_blend']**2))
            total_model = cont + g_main + g_sec
            
        elif item_type == 'DOUBLE':
            cont = r1['continuum_m'] * x_plot + r1['continuum_c']
            g_main = r1['amplitude'] * np.exp(-(x_plot - mu1)**2 / (2 * sig1**2))
            g_sec = r2['amplitude'] * np.exp(-(x_plot - mu2)**2 / (2 * sig2**2))
            total_model = cont + g_main + g_sec
            
        elif item_type == 'UNRESOLVED':
            # Même modèle pour les deux, on trace juste g_main
            cont = r1['continuum_m'] * x_plot + r1['continuum_c']
            g_main = r1['amplitude'] * np.exp(-(x_plot - mu1)**2 / (2 * sig1**2))
            g_sec = np.zeros_like(x_plot) 
            total_model = cont + g_main

        # --- GESTION DES COULEURS ---
        if clean_mode:
            main_color, blend_color, total_color = UNIFORM_COLOR, UNIFORM_COLOR, UNIFORM_COLOR
            title_color = 'black'
        else:
            if item_type == 'SINGLE' and row['fit_type'] == 'BLEND':
                main_color, blend_color, total_color = 'green', 'orange', 'green'
                title_color = 'blue'
            elif item_type == 'DOUBLE':
                main_color, blend_color, total_color = 'purple', 'orchid', 'purple'
                title_color = 'purple'
            elif item_type == 'UNRESOLVED':
                main_color, blend_color, total_color = 'cyan', 'cyan', 'cyan'
                title_color = 'teal'
            elif item_type == 'SINGLE' and row['fit_type'] == 'FORCED':
                main_color, blend_color, total_color = 'magenta', None, 'magenta'
                title_color = 'magenta'
            else: # SIMPLE
                main_color, blend_color, total_color = 'crimson', None, 'crimson'
                title_color = 'black'

        if not clean_mode:
            ax.set_title(title_name, fontsize=8, color=title_color, fontweight='bold')

        # --- TRACÉ DES COURBES ---
        ax.plot(x_plot, cont, linestyle=':', color='gray', alpha=0.5)
        ax.plot(x_plot, g_main + cont, linestyle='--', color=main_color, lw=1.5, alpha=0.8)
        
        if item_type in ['SINGLE', 'DOUBLE'] and np.any(g_sec > 0):
            ax.plot(x_plot, g_sec + cont, linestyle='--', color=blend_color, lw=1.5, alpha=0.8)
            
        ax.plot(x_plot, total_model, linestyle='-', color=total_color, lw=2, alpha=0.6 if clean_mode else 0.4)

        # --- SHIFT DOPPLER (Lignes et Couleurs) ---
        def plot_shift(l_th, l_obs, color_line):
            shift_color = 'cornflowerblue' if l_obs > l_th else 'salmon'
            ax.axvspan(l_th, l_obs, color=shift_color, alpha=0.15, zorder=0)
            ax.axvline(l_th, color='gray', ls='--', alpha=0.7, lw=1)
            ax.axvline(l_obs, color=color_line, ls='-', alpha=0.7, lw=1)

        if item_type == 'SINGLE':
            plot_shift(lth, mu, main_color)
        else:
            # DOUBLE ou UNRESOLVED : on dessine les deux shifts
            plot_shift(lth1, mu1, main_color)
            plot_shift(lth2, mu2, blend_color)

        # --- BOÎTES D'INFORMATIONS ---
        if not clean_mode:
            if item_type == 'SINGLE':
                snr = row.get('snr', 0)
                vel = row.get('Velocity_km_s', 0)
                info_txt = f"v: {vel:.1f}\nSNR: {snr:.1f}"
                ax.text(0.95, 0.95, info_txt, transform=ax.transAxes, fontsize=8,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle="round", fc="white", ec=main_color, alpha=0.8))
            else: # DOUBLE ou UNRESOLVED
                v1, v2 = r1.get('Velocity_km_s', 0), r2.get('Velocity_km_s', 0)
                snr1, snr2 = r1.get('snr', 0), r2.get('snr', 0)
                info_txt = f"v={v1:.1f} \n{r1['Line']}: SNR={snr1:.1f} \n{r2['Line']}: SNR={snr2:.1f}"
                ax.text(0.5, 0.95, info_txt, transform=ax.transAxes, fontsize=8,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle="round", fc="white", ec=total_color, alpha=0.8))

        # Esthétique finale (clean mode)
        ax.set_xlim(w_min, w_max)
        if clean_mode:
            ax.set_yticklabels([])
            ax.set_xticklabels([])

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return fig, axes






def plot_unresolved_gallery(df_spectrum, df_unresolved, n_cols=4, window_view=0.006, style_plot="step"):
    if df_unresolved is None or df_unresolved.empty: return

    processed_pairs = set()
    pairs_to_plot = []

    for idx, row in df_unresolved.iterrows():
        if row['success']:
            line1, line2 = row['Line'], row.get('blended_with', None)
            pair_key = frozenset([line1, line2])
            if pair_key not in processed_pairs:
                try:
                    row2 = df_unresolved[(df_unresolved['Line'] == line2)].iloc[0]
                    pairs_to_plot.append((row, row2))
                except: pass
                processed_pairs.add(pair_key)

    n_plots = len(pairs_to_plot)
    if n_plots == 0: return

    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    
    if n_plots == 1: axes = [axes]
    axes = np.array(axes).flatten()

    for i, (r1, r2) in enumerate(pairs_to_plot):
        ax = axes[i]
        mu_obs, sigma, amp = r1['wavelength_obs'], r1['sigma'], r1['amplitude']
        m, c = r1['continuum_m'], r1['continuum_c']
        lth1, lth2 = r1['wavelength_th'], r2['wavelength_th']
        
        # Fenêtre strictement bornée
        w_min = mu_obs - max(window_view/2, 4*sigma)
        w_max = mu_obs + max(window_view/2, 4*sigma)
        
        mask = (df_spectrum["Wavelength"] >= w_min) & (df_spectrum["Wavelength"] <= w_max)
        x_val, y_val = df_spectrum.loc[mask, "Wavelength"].values, df_spectrum.loc[mask, "Int"].values
        
        if len(x_val) == 0: continue
            
        if style_plot == "plot": ax.plot(x_val, y_val, c='black', lw=1.5, alpha=0.8, marker='.')
        else: ax.step(x_val, y_val, c='black', lw=1.5, alpha=0.8, where='mid')
            
        x_plot = np.linspace(w_min, w_max, 200)
        cont = m * x_plot + c
        g_main = amp * np.exp(-(x_plot - mu_obs)**2 / (2 * sigma**2))
        
        ax.plot(x_plot, cont, ls=':', color='gray', alpha=0.5)
        ax.plot(x_plot, cont + g_main, ls='-', color='cyan', lw=2, alpha=0.8)
        
        ax.axvline(mu_obs, color='black', ls='-', lw=1.5, alpha=0.5)
        
        c1, c2 = 'dodgerblue', 'darkorange'
        
        # Coloration des écarts bornée strictement
        ax.axvspan(lth1, mu_obs, color='salmon' if mu_obs > lth1 else 'cornflowerblue', alpha=0.15)
        ax.axvspan(lth2, mu_obs, color='salmon' if mu_obs > lth2 else 'cornflowerblue', alpha=0.15)
        
        ax.axvline(lth1, color=c1, ls='--', lw=1.5, alpha=0.8)
        ax.axvline(lth2, color=c2, ls='--', lw=1.5, alpha=0.8)
        
        text1 = f"{r1['Line']}\n$\Delta\lambda$: {mu_obs-lth1:.4f}\nv: {r1['Velocity_km_s']:.1f}"
        text2 = f"{r2['Line']}\n$\Delta\lambda$: {mu_obs-lth2:.4f}\nv: {r2['Velocity_km_s']:.1f}"
        
        # Les flèches pointent proprement au milieu du graphe, plus de risque d'explosion Y
        y_pointe = np.mean(y_val)
        
        ax.annotate(text1, xy=(lth1, y_pointe), xytext=(0.02, 0.95), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color=c1, alpha=0.7), color=c1, 
                    fontsize=8, ha='left', va='top', bbox=dict(boxstyle="round", fc="white", ec=c1, alpha=0.8))
                    
        ax.annotate(text2, xy=(lth2, y_pointe), xytext=(0.98, 0.95), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color=c2, alpha=0.7), color=c2, 
                    fontsize=8, ha='right', va='top', bbox=dict(boxstyle="round", fc="white", ec=c2, alpha=0.8))

        ax.set_title(f"{r1['Line']} / {r2['Line']}", fontsize=9, fontweight='bold')
        ax.set_xlim(w_min, w_max)
        
    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
    return fig, axes





























###########################################################
# Merge data




def merge_fit_results(df_main, df_corrections):
    """
    Fusionne les résultats globaux avec les corrections manuelles.
    La priorité est donnée à 'df_corrections'.
    """
    if df_corrections is None or df_corrections.empty:
        print("Aucune correction à appliquer. Retour du DF original.")
        return df_main.copy()

    # 1. Copie pour ne pas toucher l'original
    df_final = df_main.copy()

    # 2. On supprime du DF principal les lignes qui existent dans les corrections
    # On se base sur la colonne 'Line' (l'identifiant unique)
    lignes_a_corriger = df_corrections['Line'].unique()

    # Combien on en remplace ?
    nb_overlap = df_final[df_final['Line'].isin(lignes_a_corriger)].shape[0]
    print(
        f"Fusion : Remplacement de {nb_overlap} raies par leur version manuelle/double.")

    # Filtrage : On garde tout ce qui n'est PAS dans la liste des corrections
    df_final = df_final[~df_final['Line'].isin(lignes_a_corriger)]

    # 3. Concaténation (Ajout des nouvelles versions)
    df_merged = pd.concat([df_final, df_corrections], ignore_index=True)

    # 4. Tri final par longueur d'onde théorique pour que ce soit propre
    if 'wavelength_th' in df_merged.columns:
        df_merged = df_merged.sort_values('wavelength_th')

    df_merged = df_merged.reset_index(drop=True)

    return df_merged


def merge_spectral_dataframes(df_simple, df_blend, df_forced):
    """
    Fusionne les résultats en donnant la priorité : Blend > Forcé > Simple.
    Ajoute une colonne 'fit_type'.
    """
    # 1. Préparation des copies avec étiquettes
    # On s'assure que les colonnes spécifiques aux blends existent partout (remplies de NaN si absentes)
    cols_blend = ['amp_blend', 'mu_blend', 'sig_blend']

    # Traitement Blend (Priorité 1)
    if df_blend is not None and not df_blend.empty:
        d1 = df_blend.copy()
        d1['fit_type'] = 'BLEND'
    else:
        d1 = pd.DataFrame()

    # Traitement Forcé (Priorité 2)
    if df_forced is not None and not df_forced.empty:
        d2 = df_forced.copy()
        d2['fit_type'] = 'FORCED'
        for c in cols_blend:
            d2[c] = np.nan  # Pas de composante blend
    else:
        d2 = pd.DataFrame()

    # Traitement Simple (Priorité 3)
    if df_simple is not None and not df_simple.empty:
        d3 = df_simple[df_simple['success'] == True].copy()
        d3['fit_type'] = 'SIMPLE'
        for c in cols_blend:
            d3[c] = np.nan
    else:
        d3 = pd.DataFrame()

    # 2. Concaténation dans l'ordre de priorité
    # L'ordre est crucial ici : d1 (Blend) en premier, etc.
    df_tot = pd.concat([d1, d2, d3], ignore_index=True)

    # 3. Suppression des doublons sur 'Line'
    # keep='first' garde l'entrée venant du DF le plus prioritaire
    df_merged = df_tot.drop_duplicates(
        subset=['Line'], keep='first').reset_index(drop=True)

    return df_merged




def merge_spectral_dataframesV2(df_simple, df_blend, df_forced,df_unresolved, df_double=None):
    """
    Fusionne les résultats en donnant la priorité : Double > Blend > Forcé > Simple.
    Ajoute une colonne 'fit_type'.
    """
    
    # 1. Traitement Double (Priorité 1 - NOUVEAU & MAX PRIORITY)
    if df_double is not None and not df_double.empty:
        # On ne garde que les fits réussis
        d0 = df_double[df_double['success'] == True].copy()
        d0['fit_type'] = 'DOUBLE'
    else:
        d0 = pd.DataFrame()

    if df_unresolved is not None and not df_unresolved.empty:
        d_unres = df_unresolved[df_unresolved['success'] == True].copy()
    else:
        d_unres = pd.DataFrame()
        
    # 2. Traitement Blend (Priorité 2)
    if df_blend is not None and not df_blend.empty:
        d1 = df_blend.copy()
        if 'success' in d1.columns:
            d1 = d1[d1['success'] == True]
        d1['fit_type'] = 'BLEND'
    else:
        d1 = pd.DataFrame()

    # 3. Traitement Forcé (Priorité 3)
    if df_forced is not None and not df_forced.empty:
        d2 = df_forced.copy()
        if 'success' in d2.columns:
            d2 = d2[d2['success'] == True]
        d2['fit_type'] = 'FORCED'
    else:
        d2 = pd.DataFrame()

    # 4. Traitement Simple (Priorité 4)
    if df_simple is not None and not df_simple.empty:
        d3 = df_simple[df_simple['success'] == True].copy()
        d3['fit_type'] = 'SIMPLE'
    else:
        d3 = pd.DataFrame()

    # 2. Concaténation dans l'ordre de priorité STRICT (d0 puis d1 puis d2 puis d3)
    # pd.concat gère tout seul les colonnes qui n'existent pas partout (ex: 'blended_with') en mettant des NaN.
    dfs_to_concat = [df for df in [d0, d_unres, d1, d2, d3] if not df.empty]

    
    if not dfs_to_concat:
        return pd.DataFrame() # Retourne un DF vide si tout est vide
        
    df_tot = pd.concat(dfs_to_concat, ignore_index=True)

    # 3. Suppression des doublons sur 'Line'
    # keep='first' garde l'entrée venant du DF le plus prioritaire dans l'ordre de concaténation
    df_merged = df_tot.drop_duplicates(subset=['Line'], keep='first').reset_index(drop=True)

    # Sécurité : on s'assure que les colonnes spécifiques aux blends/doubles existent partout
    # même si le DF qui les générait était vide.
    cols_speciales = ['amp_blend', 'mu_blend', 'sig_blend', 'blended_with']
    for c in cols_speciales:
        if c not in df_merged.columns:
            df_merged[c] = np.nan

    return df_merged


def mark_unresolved_pairs(df_merged, unresolved_pairs):
    """
    A partir d'un df_merged deja fitte (par fit_manual_lines2 ou equivalent),
    modifie les colonnes 'fit_type' et 'blended_with' pour les raies listees
    dans unresolved_pairs.

    Les raies doivent deja exister dans df_merged (elles ont ete fittees
    individuellement). Cette fonction se contente de les re-tagger.

    Parametres
    ----------
    df_merged        : DataFrame resultat du fit (doit contenir 'Line', 'fit_type')
    unresolved_pairs : liste de tuples (line1, line2)

    Retour
    ------
    Copie de df_merged avec 'fit_type' = 'UNRESOLVED' et 'blended_with' renseignes
    pour chaque raie concernee.
    """
    df = df_merged.copy()

    if 'blended_with' not in df.columns:
        df['blended_with'] = pd.NA

    missing = []
    for line1, line2 in unresolved_pairs:
        for ln, partner in [(line1, line2), (line2, line1)]:
            mask = df['Line'] == ln
            if mask.sum() == 0:
                missing.append(ln)
                continue
            df.loc[mask, 'fit_type']     = 'UNRESOLVED'
            df.loc[mask, 'blended_with'] = partner

    if missing:
        print(f"[WARN] mark_unresolved_pairs : raies absentes de df_merged : {missing}")

    n_tagged = sum(
        1 for ln in [l for pair in unresolved_pairs for l in pair]
        if ln in df['Line'].values
    )
    print(f"[OK] {n_tagged} raies taguees UNRESOLVED ({len(unresolved_pairs)} paires).")
    return df

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
# =============================================================================
# 2. PLOT SPECTRAL SUMMARY V4
# =============================================================================

def plot_spectral_summary_cleanV4(df_spectrum, df_merged, type_plot="step",
                                   wl_range=(4.576, 4.925), n_plots=3,
                                   ylim=(100, 200),
                                   snrtype='snr2', snr_threshold=3.0,
                                   v_range=(-10, 25),
                                   clean_mode=False, alpha=0.9):
    """
    Trace le resume spectral avec gestion de 5 types de fits :
      SIMPLE   -- gaussienne libre             (rouge / crimson)
      FORCED   -- continuum contraint          (magenta)
      BLEND    -- double gaussienne asymetrique (bleu / teal)
      DOUBLE   -- deux gaussiennes independantes (violet / orchid)
      UNRESOLVED -- deux raies quasi-degenerees, fittees individuellement
                    mais visuellement liees par une accolade + shading commun
                    (orange / darkorange)

    Logique UNRESOLVED :
      - Les deux raies partagent une zone de shading commune (entre lth1 et lth2).
      - Une accolade horizontale relie les deux pics observes.
      - Les deux etiquettes sont forcees a des niveaux Y differents pour eviter
        tout recouvrement, independamment de la logique globale d'anti-overlap.
      - Le modele gaussien de chaque raie est trace individuellement.

    Parametres
    ----------
    df_spectrum     : DataFrame avec 'Wavelength' et 'Int'
    df_merged       : DataFrame resultat du fit (apres mark_unresolved_pairs)
    type_plot       : 'step' ou 'plot'
    wl_range        : (wl_min, wl_max) en um
    n_plots         : nombre de sous-panneaux verticaux
    ylim            : (y_min, y_max) commun a tous les panneaux
    snrtype         : colonne SNR a utiliser ('snr' ou 'snr2')
    snr_threshold   : seuil SNR pour coloration rouge/noir de l'etiquette
    v_range         : (v_min, v_max) en km/s pour la meme logique
    clean_mode      : si True, schema monochrome simplifie (vert/orange)
    alpha           : transparence des modeles de fit

    Retour
    ------
    (fig, axes)
    """
    wl_min, wl_max = wl_range
    delta_wl = (wl_max - wl_min) / n_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(20, 7 * n_plots))
    if n_plots == 1:
        axes = [axes]

    N_SIGMA_CUT = 4.5   # demi-largeur de la fenetre de tracé du modele (en sigma)

    # Couleurs par type (mode normal)
    COLOR_MAP = {
        'SIMPLE':     ('crimson',    'crimson'),
        'FORCED':     ('magenta',    'magenta'),
        'BLEND':      ('royalblue',  'teal'),
        'DOUBLE':     ('purple',     'orchid'),
        'UNRESOLVED': ('darkorange', 'orange'),
    }
    COLOR_CLEAN = ('darkgreen', 'darkorange')

    for i, ax in enumerate(axes):
        ax.set_ylim(ylim)
        c_min = wl_min + i * delta_wl
        c_max = wl_min + (i + 1) * delta_wl
        plot_width = c_max - c_min
        y_range = ylim[1] - ylim[0]

        # ── Spectre ───────────────────────────────────────────────────────────
        mask_view = ((df_spectrum["Wavelength"] >= c_min) &
                     (df_spectrum["Wavelength"] <= c_max))
        if mask_view.sum() == 0:
            continue
        wave_view = df_spectrum.loc[mask_view, "Wavelength"].values
        flux_view = df_spectrum.loc[mask_view, "Int"].values

        if type_plot == "plot":
            ax.plot(wave_view, flux_view, color="black", lw=2, alpha=0.9, zorder=1)
        else:
            ax.step(wave_view, flux_view, color="black", lw=2, alpha=0.9, zorder=1,
                    where='mid')

        # ── Subset des raies visibles dans ce panneau ─────────────────────────
        subset = df_merged[
            (df_merged['wavelength_obs'] >= c_min) &
            (df_merged['wavelength_obs'] <= c_max)
        ].sort_values('wavelength_obs')

        # ── Pre-traitement UNRESOLVED : construire les paires visibles ─────────
        # On identifie les paires dont les DEUX membres sont dans df_merged
        # (pas necessairement dans ce panneau -- la paire peut etre a cheval)
        plotted_unresolved = set()   # frozenset de (line1, line2) deja traitees
        plotted_double     = set()   # idem pour DOUBLE

        # Niveaux Y pour les etiquettes (anti-overlap general)
        y_levels = [
            ylim[0] + 0.87 * y_range,
            ylim[0] + 0.70 * y_range,
        ]
        # Niveau supplementaire reserve aux UNRESOLVED (toujours distinct)
        y_level_unres = [
            ylim[0] + 0.87 * y_range,
            ylim[0] + 0.68 * y_range,
        ]

        last_x_pos  = -999
        level_toggle = 0

        for _, row in subset.iterrows():
            fit_type = str(row.get('fit_type', 'SIMPLE'))
            mu    = row['wavelength_obs']
            sigma = row['sigma']
            amp   = row['amplitude']
            lth   = row['wavelength_th']
            m, c  = row['continuum_m'], row['continuum_c']

            base_color, sec_color = (COLOR_CLEAN if clean_mode
                                     else COLOR_MAP.get(fit_type, ('crimson', 'crimson')))

            # =================================================================
            # CAS UNRESOLVED
            # =================================================================
            if fit_type == 'UNRESOLVED':
                partner_name = row.get('blended_with', None)
                pair_key = frozenset([row['Line'], partner_name])

                # Recuperer le partenaire dans df_merged
                partner_rows = df_merged[df_merged['Line'] == partner_name]
                has_partner  = (not partner_rows.empty) and (partner_name is not None)

                if has_partner:
                    pr = partner_rows.iloc[0]
                    mu_p    = pr['wavelength_obs']
                    sigma_p = pr['sigma']
                    amp_p   = pr['amplitude']
                    lth_p   = pr['wavelength_th']
                    mp, cp  = pr['continuum_m'], pr['continuum_c']
                else:
                    mu_p    = mu
                    sigma_p = sigma
                    amp_p   = 0.0
                    lth_p   = lth

                # -- Shading commun entre les deux positions theoriques --------
                lth_lo = min(lth, lth_p)
                lth_hi = max(lth, lth_p)
                # On etend legerement le shading autour du blend
                shad_lo = min(lth_lo, mu, mu_p) - 1.5 * max(sigma, sigma_p)
                shad_hi = max(lth_hi, mu, mu_p) + 1.5 * max(sigma, sigma_p)
                ax.axvspan(shad_lo, shad_hi, color='orange', alpha=0.08, zorder=0)

                # -- Modele gaussien de CETTE raie (fenetre individuelle) ------
                x1 = np.linspace(mu - N_SIGMA_CUT * sigma,
                                  mu + N_SIGMA_CUT * sigma, 150)
                cont1 = m * x1 + c
                g1    = amp * np.exp(-(x1 - mu)**2 / (2 * sigma**2))
                ax.plot(x1, cont1 + g1, ls='-', color=base_color,
                        lw=2.2, alpha=alpha, zorder=5)
                ax.plot(x1, cont1, ls=':', color='gray', lw=1.0, alpha=0.4, zorder=2)

                # -- Lignes theoriques ----------------------------------------
                ax.axvline(lth,   color=base_color, ls='--', lw=1.2, alpha=0.6, zorder=1)
                ax.axvline(mu,    color=base_color, ls='-',  lw=1.0, alpha=0.5, zorder=1)

                # -- Accolade horizontale entre les deux mu observes ----------
                # (tracee une seule fois par paire)
                if pair_key not in plotted_unresolved and has_partner:
                    plotted_unresolved.add(pair_key)

                    mu_lo = min(mu, mu_p)
                    mu_hi = max(mu, mu_p)
                    # Hauteur de l'accolade : juste au-dessus du pic le plus haut
                    peak1 = (m * mu + c) + amp
                    peak2 = (mp * mu_p + cp) + amp_p
                    y_brace = max(peak1, peak2) + 0.015 * y_range

                    # Accolade = trait horizontal + deux petits traits verticaux
                    brace_h = 0.008 * y_range   # hauteur des "griffes"
                    ax.annotate("",
                                xy=(mu_hi, y_brace),
                                xytext=(mu_lo, y_brace),
                                arrowprops=dict(
                                    arrowstyle="-",
                                    color=base_color, lw=2.0,
                                    connectionstyle="bar,fraction=0.0"
                                ), zorder=8)
                    # Petits tirets verticaux aux extremites
                    for x_brace in [mu_lo, mu_hi]:
                        ax.plot([x_brace, x_brace],
                                [y_brace - brace_h, y_brace + brace_h],
                                color=base_color, lw=2.0, alpha=0.85, zorder=8)
                    # Label central de l'accolade
                    ax.text((mu_lo + mu_hi) / 2.0, y_brace + brace_h * 1.5,
                            "blend", ha='center', va='bottom',
                            fontsize=7, color=base_color, style='italic',
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec=base_color, alpha=0.75), zorder=9)

                # -- Etiquettes avec niveaux alternes forces ------------------
                if not clean_mode:
                    snr_val = row.get(snrtype, 0)
                    vel_val = row.get('Velocity_km_s', 0)
                    is_bad  = ((snr_val < snr_threshold) or
                               (vel_val < v_range[0]) or (vel_val > v_range[1]))
                    box_col = "red" if is_bad else base_color

                    # Alternance strict : les deux raies du blend vont a des
                    # niveaux differents en fonction de leur position (gauche/droite)
                    # La raie de plus courte longueur d'onde va au niveau haut.
                    if has_partner:
                        y_lbl = y_level_unres[0] if mu <= mu_p else y_level_unres[1]
                    else:
                        y_lbl = y_level_unres[0]

                    label_txt = (f"{row['Line']}\n"
                                 f"v:{vel_val:.1f} SNR:{snr_val:.1f}\n"
                                 f"blend:{partner_name}")
                    peak_y = (m * mu + c) + amp

                    ax.annotate(label_txt,
                                xy=(mu, peak_y),
                                xytext=(mu, y_lbl),
                                arrowprops=dict(arrowstyle='-',
                                                color=box_col, alpha=0.4, lw=0.8),
                                fontsize=7.5, color=box_col,
                                ha='center', va='bottom', rotation=90,
                                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                          ec=box_col, alpha=0.85, zorder=20))
                continue   # FIN du cas UNRESOLVED -- on passe a la raie suivante

            # =================================================================
            # CAS STANDARD (SIMPLE, FORCED, BLEND, DOUBLE)
            # =================================================================

            # -- Fenetre de tracé du modele -----------------------------------
            if fit_type == 'BLEND':
                mu2, sig2 = row['mu_blend'], row['sig_blend']
                start_x = min(mu - N_SIGMA_CUT * sigma, mu2 - N_SIGMA_CUT * sig2)
                end_x   = max(mu + N_SIGMA_CUT * sigma, mu2 + N_SIGMA_CUT * sig2)
            elif fit_type == 'DOUBLE':
                try:
                    comp    = df_merged[(df_merged['Line'] == row['blended_with']) &
                                        (df_merged['fit_type'] == 'DOUBLE')].iloc[0]
                    mu2, sig2 = comp['wavelength_obs'], comp['sigma']
                    start_x = min(mu - N_SIGMA_CUT * sigma, mu2 - N_SIGMA_CUT * sig2)
                    end_x   = max(mu + N_SIGMA_CUT * sigma, mu2 + N_SIGMA_CUT * sig2)
                except Exception:
                    start_x, end_x = mu - N_SIGMA_CUT * sigma, mu + N_SIGMA_CUT * sigma
            else:
                start_x, end_x = mu - N_SIGMA_CUT * sigma, mu + N_SIGMA_CUT * sigma

            x_mod    = np.linspace(start_x, end_x, 150)
            cont_loc = m * x_mod + c
            g_main   = amp * np.exp(-(x_mod - mu)**2 / (2 * sigma**2))
            g_blend  = np.zeros_like(x_mod)
            g_comp   = np.zeros_like(x_mod)

            if fit_type == 'BLEND':
                g_blend = (row['amp_blend'] *
                           np.exp(-(x_mod - row['mu_blend'])**2 /
                                  (2 * row['sig_blend']**2)))
            elif fit_type == 'DOUBLE':
                try:
                    comp   = df_merged[(df_merged['Line'] == row['blended_with']) &
                                       (df_merged['fit_type'] == 'DOUBLE')].iloc[0]
                    g_comp = (comp['amplitude'] *
                              np.exp(-(x_mod - comp['wavelength_obs'])**2 /
                                     (2 * comp['sigma']**2)))
                except Exception:
                    pass

            total_model = cont_loc + g_main + g_blend + g_comp

            # -- Tracé --------------------------------------------------------
            ax.plot(x_mod, cont_loc,    ls=':', color='gray',     lw=1.0, alpha=0.4, zorder=2)
            ax.plot(x_mod, cont_loc + g_main, ls='--', color=sec_color,
                    lw=1.5, alpha=0.6, zorder=3)
            if fit_type == 'BLEND':
                ax.plot(x_mod, cont_loc + g_blend, ls=':', color=sec_color,
                        lw=1.2, alpha=0.6, zorder=3)

            if fit_type == 'DOUBLE':
                pair_key = frozenset([row['Line'], row.get('blended_with', None)])
                if pair_key not in plotted_double:
                    ax.plot(x_mod, total_model, ls='-', color=base_color,
                            lw=2.5, alpha=alpha, zorder=5)
                    plotted_double.add(pair_key)
            else:
                ax.plot(x_mod, total_model, ls='-', color=base_color,
                        lw=2.5, alpha=alpha, zorder=5)

            # -- Shift Doppler ------------------------------------------------
            delta_lambda = mu - lth
            shift_color  = 'salmon' if delta_lambda > 0 else 'cornflowerblue'
            ax.axvspan(lth, mu, color=shift_color, alpha=0.15, zorder=0)
            ax.axvline(lth, color='gray',       ls='--', alpha=0.6, lw=1.0, zorder=1)
            ax.axvline(mu,  color=base_color,   ls='-',  alpha=0.6, lw=1.0, zorder=1)

            # -- Etiquettes (mode non-clean) ----------------------------------
            if not clean_mode:
                snr_val = row.get(snrtype, 0)
                vel_val = row.get('Velocity_km_s', 0)
                is_bad  = ((snr_val < snr_threshold) or
                           (vel_val < v_range[0]) or (vel_val > v_range[1]))
                box_col = "red" if is_bad else "black"

                dist_to_prev = abs(mu - last_x_pos)
                if dist_to_prev < (plot_width * 0.02):
                    level_toggle = (level_toggle + 1) % 2
                else:
                    level_toggle = 0

                y_text_pos  = y_levels[level_toggle]
                last_x_pos  = mu

                label_txt  = f"{row['Line']} | v:{vel_val:.1f} | SNR:{snr_val:.1f}"
                peak_y     = cont_loc[np.argmin(np.abs(x_mod - mu))] + amp

                ax.annotate(label_txt,
                            xy=(mu, peak_y),
                            xytext=(mu, y_text_pos),
                            arrowprops=dict(arrowstyle='-',
                                            color=box_col, alpha=0.3, lw=0.8),
                            fontsize=9, color=box_col,
                            ha='center', va='bottom', rotation=90,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      ec=box_col, alpha=0.8, zorder=20))

        ax.set_xlim(c_min, c_max)
        ax.grid(True, alpha=0.2)

        # -- Legende (premier panneau uniquement) -----------------------------
        if i == 0:
            legend_elements = [
                Line2D([0], [0], color='gray',       ls='--', lw=1.5,
                       label=r'$\lambda_{th}$'),
                Line2D([0], [0], color='black',      ls='-',  lw=1.5,
                       label=r'$\lambda_{obs}$'),
                Line2D([0], [0], color='crimson',    ls='-',  lw=2,
                       label='SIMPLE'),
                Line2D([0], [0], color='magenta',    ls='-',  lw=2,
                       label='FORCED'),
                Line2D([0], [0], color='royalblue',  ls='-',  lw=2,
                       label='BLEND'),
                Line2D([0], [0], color='purple',     ls='-',  lw=2,
                       label='DOUBLE'),
                mpatches.Patch(facecolor='red', alpha=0.35,
                               label='UNRESOLVED blend'),
            ]
            ax.legend(handles=legend_elements, loc='upper left',
                      fontsize=8, ncol=4, framealpha=0.9)

    axes[-1].set_xlabel(r"Wavelength $\lambda$ [$\mu$m]", fontsize=12)
    plt.tight_layout()
    return fig, axes







def build_df_merged_for_plot(df_final_fits, df_continuum):
    """
    Fusionne df_final_fits (SIMPLE) et df_continuum (FORCED) en un seul
    DataFrame pret pour plot_spectral_summary_cleanV4.

    Regles
    ------
    - df_continuum  a la priorite : ses raies ecrasent celles de df_final_fits.
    - fit_type = 'FORCED' pour df_continuum, 'SIMPLE' pour df_final_fits.
    - Les colonnes manquantes dans l'un ou l'autre sont remplies de NaN.
    - Les raies avec success=False sont exclues des deux sources.

    Parametres
    ----------
    df_final_fits : DataFrame issu de Analyse_Spectrum_SingleGaussian
                    (ou de merge_spectral_dataframesV2 si tu as deja des blends)
    df_continuum  : DataFrame issu de fit_manual_lines2

    Retour
    ------
    DataFrame fusionne, trie par wavelength_obs, sans doublons sur 'Line'.
    """
    # -- Nettoyage et tag des deux sources ------------------------------------
    cols_required = ['Line', 'wavelength_th', 'wavelength_obs', 'amplitude',
                     'sigma', 'Int_cgs', 'continuum_m', 'continuum_c',
                     'Velocity_km_s', 'snr', 'snr2', 'success']

    def _prepare(df, fit_type_label):
        d = df.copy()
        # Garder uniquement les fits reussis
        if 'success' in d.columns:
            d = d[d['success'] == True]
        # Tagger le type
        if 'fit_type' not in d.columns:
            d['fit_type'] = fit_type_label
        else:
            # Si fit_type existe mais est NaN ou vide, on le renseigne
            mask_empty = d['fit_type'].isna() | (d['fit_type'] == '')
            d.loc[mask_empty, 'fit_type'] = fit_type_label
        return d

    d_forced = _prepare(df_continuum,  'FORCED')
    d_simple = _prepare(df_final_fits, 'SIMPLE')

    # -- Concatenation dans l'ordre de priorite (FORCED en premier) ----------
    df_tot = pd.concat([d_forced, d_simple], ignore_index=True)

    # -- Suppression des doublons : keep='first' garde FORCED sur SIMPLE ------
    df_merged = (df_tot
                 .drop_duplicates(subset=['Line'], keep='first')
                 .sort_values('wavelength_obs')
                 .reset_index(drop=True))

    # -- S'assurer que les colonnes blend/unresolved existent (pour la compat) -
    for col in ['blended_with', 'amp_blend', 'mu_blend', 'sig_blend']:
        if col not in df_merged.columns:
            df_merged[col] = np.nan

    n_forced = (df_merged['fit_type'] == 'FORCED').sum()
    n_simple = (df_merged['fit_type'] == 'SIMPLE').sum()
    print(f"[build_df_merged_for_plot] {len(df_merged)} raies total : "
          f"{n_forced} FORCED + {n_simple} SIMPLE.")

    return df_merged




_C_KMS = 299792.458

# ---- Couleurs par categorie -------------------------------------------------
# (couleur_principale, couleur_secondaire, couleur_titre)
_PAL = {
    'SIMPLE':     ('crimson',    'crimson',    'crimson'),
    'FORCED':     ('magenta',    'magenta',    'magenta'),
    'BLEND':      ('royalblue',  'teal',       'royalblue'),
    'DOUBLE':     ('purple',     'orchid',     'purple'),
    'UNRESOLVED': ('dodgerblue', 'darkorange', 'saddlebrown'),
}
_ORDER = ['SIMPLE', 'FORCED', 'BLEND', 'DOUBLE', 'UNRESOLVED']


def plot_smart_gallery_from_mergedV5(df_spectrum, df_merged,
                                      n_cols=6, window_view=0.008,
                                      clean_mode=False, style_plot="plot",
                                      figsize_per_panel=(5, 4)):
    """
    Parametres
    ----------
    df_spectrum       : DataFrame 'Wavelength' / 'Int'
    df_merged         : DataFrame fusionne avec colonnes fit_type, blended_with
    n_cols            : nombre de colonnes de la galerie
    window_view       : largeur de fenetre en um (None = auto)
    clean_mode        : True -> pas d'annotations, monochrome
    style_plot        : 'plot' ou 'step'
    figsize_per_panel : (w, h) en pouces par vignette

    Retour
    ------
    (fig, axes)
    """
    if df_merged is None or df_merged.empty:
        print("[INFO] df_merged vide.")
        return None, None

    # =========================================================================
    # 1. Construction de la liste ordonnee des items a tracer
    # =========================================================================
    processed_pairs = set()
    items = {cat: [] for cat in _ORDER}

    for _, row in df_merged.sort_values('wavelength_obs').iterrows():
        ft = str(row.get('fit_type', 'SIMPLE'))

        if ft in ('DOUBLE', 'UNRESOLVED'):
            line1 = row['Line']
            line2 = row.get('blended_with', None)
            if pd.isna(line2) or line2 is None:
                # partenaire manquant -> traite comme SIMPLE
                items['SIMPLE'].append({'type': 'SINGLE', 'ft': ft, 'r1': row})
                continue
            pair_key = frozenset([line1, line2])
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            sub2 = df_merged[df_merged['Line'] == line2]
            if sub2.empty:
                items['SIMPLE'].append({'type': 'SINGLE', 'ft': ft, 'r1': row})
                continue
            row2 = sub2.iloc[0]

            # r1 = raie la plus bleue
            if float(row['wavelength_th']) <= float(row2['wavelength_th']):
                r1, r2 = row, row2
            else:
                r1, r2 = row2, row

            cat = ft  # 'DOUBLE' ou 'UNRESOLVED'
            items[cat].append({'type': 'PAIR', 'ft': ft, 'r1': r1, 'r2': r2})

        else:
            cat = ft if ft in _ORDER else 'SIMPLE'
            items[cat].append({'type': 'SINGLE', 'ft': ft, 'r1': row})

    # Aplatir dans l'ordre des categories
    items_flat = []
    for cat in _ORDER:
        items_flat.extend(items[cat])

    n_plots = len(items_flat)
    if n_plots == 0:
        print("[INFO] Rien a tracer.")
        return None, None

    # =========================================================================
    # 2. Figure
    # =========================================================================
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False
    )
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    axes_flat = axes.flatten()

    # =========================================================================
    # 3. Boucle de tracé
    # =========================================================================
    for i, item in enumerate(items_flat):
        ax    = axes_flat[i]
        itype = item['type']   # 'SINGLE' ou 'PAIR'
        ft    = item['ft']
        r1    = item['r1']
        r2    = item.get('r2', None)

        c_main, c_sec, c_title = _PAL.get(ft, _PAL['SIMPLE'])

        # ── Extraction des parametres scalaires ──────────────────────────────
        mu1   = float(r1['wavelength_obs'])
        sig1  = float(r1['sigma'])
        amp1  = float(r1['amplitude'])
        lth1  = float(r1['wavelength_th'])
        m1    = float(r1['continuum_m'])
        c1v   = float(r1['continuum_c'])
        snr1  = float(r1.get('snr2', r1.get('snr', 0)))

        if itype == 'PAIR':
            mu2   = float(r2['wavelength_obs'])
            sig2  = float(r2['sigma'])
            amp2  = float(r2['amplitude'])
            lth2  = float(r2['wavelength_th'])
            m2    = float(r2['continuum_m'])
            c2v   = float(r2['continuum_c'])
            snr2  = float(r2.get('snr2', r2.get('snr', 0)))
            lc    = (lth1 + lth2) / 2.0

        # ── Fenetre d'affichage ──────────────────────────────────────────────
        if itype == 'SINGLE':
            half  = max((window_view or 0) / 2, 4.5 * sig1)
            w_ctr = mu1
        else:
            half  = max((window_view or 0) / 2,
                        abs(lth2 - lth1) * 2.5,
                        4.5 * max(sig1, sig2),
                        0.003)
            w_ctr = lc

        w_min, w_max = w_ctr - half, w_ctr + half

        # ── Spectre local ────────────────────────────────────────────────────
        mask   = ((df_spectrum["Wavelength"] >= w_min) &
                  (df_spectrum["Wavelength"] <= w_max))
        x_obs  = df_spectrum.loc[mask, "Wavelength"].values
        y_obs  = df_spectrum.loc[mask, "Int"].values

        if len(x_obs) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='gray')
            continue

        if style_plot == 'step':
            ax.step(x_obs, y_obs, c='black', lw=1.4, alpha=0.85, where='mid')
        else:
            ax.plot(x_obs, y_obs, c='black', lw=1.4, alpha=0.85,
                    marker='o', ms=2, markerfacecolor='black')

        x_plot = np.linspace(w_min, w_max, 400)
        y_span = np.max(y_obs) - np.min(y_obs)

        # =====================================================================
        # SINGLE  (SIMPLE, FORCED, BLEND)
        # =====================================================================
        if itype == 'SINGLE':
            cont  = m1 * x_plot + c1v
            g_main = amp1 * np.exp(-(x_plot - mu1)**2 / (2 * sig1**2))
            g_sec  = np.zeros_like(x_plot)

            if ft == 'BLEND':
                ab = float(r1.get('amp_blend', 0) or 0)
                mb = float(r1.get('mu_blend', mu1) or mu1)
                sb = float(r1.get('sig_blend', sig1) or sig1)
                if ab > 0:
                    g_sec = ab * np.exp(-(x_plot - mb)**2 / (2 * sb**2))

            ax.plot(x_plot, cont,                  ls=':', lw=1.0, color='gray', alpha=0.5)
            ax.plot(x_plot, cont + g_main,         ls='--', lw=1.8, color=c_main, alpha=0.85)
            if ft == 'BLEND' and np.any(g_sec > 0):
                ax.plot(x_plot, cont + g_sec,      ls='--', lw=1.4, color=c_sec, alpha=0.75)
            ax.plot(x_plot, cont + g_main + g_sec, ls='-',  lw=2.2, color=c_main, alpha=0.75)

            # Shift Doppler
            sc = 'salmon' if mu1 > lth1 else 'cornflowerblue'
            ax.axvspan(min(lth1, mu1), max(lth1, mu1), color=sc, alpha=0.14, zorder=0)
            ax.axvline(lth1, color='gray',  ls='--', lw=1.0, alpha=0.6)
            ax.axvline(mu1,  color=c_main,  ls='-',  lw=1.0, alpha=0.55)

            # Titre
            ax.set_title(r1['Line'], fontsize=8.5, fontweight='bold', color=c_title)

            # Boite info
            if not clean_mode:
                vel = _C_KMS * (mu1 - lth1) / lth1
                ax.text(0.97, 0.97,
                        f"v = {vel:+.1f} km/s\nSNR = {snr1:.1f}",
                        transform=ax.transAxes, fontsize=7.5,
                        va='top', ha='right',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec=c_main, alpha=0.88))

        # =====================================================================
        # DOUBLE  -- 2 gaussiennes independantes, vitesse unique, 2 SNR
        # =====================================================================
        elif ft == 'DOUBLE':
            # Continuum de r1 (les deux partagent le meme fit de continuum)
            cont  = m1 * x_plot + c1v
            g1    = amp1 * np.exp(-(x_plot - mu1)**2 / (2 * sig1**2))
            g2    = amp2 * np.exp(-(x_plot - mu2)**2 / (2 * sig2**2))

            ax.plot(x_plot, cont,              ls=':', lw=1.0, color='gray', alpha=0.5)
            ax.plot(x_plot, cont + g1,         ls='--', lw=1.8, color=c_main, alpha=0.85)
            ax.plot(x_plot, cont + g2,         ls='--', lw=1.6, color=c_sec,  alpha=0.80)
            ax.plot(x_plot, cont + g1 + g2,    ls='-',  lw=2.4, color=c_main, alpha=0.70)

            # Shifts (lth1 seul = raie principale)
            sc1 = 'salmon' if mu1 > lth1 else 'cornflowerblue'
            ax.axvspan(min(lth1, mu1), max(lth1, mu1), color=sc1, alpha=0.14, zorder=0)
            ax.axvline(lth1, color='gray',  ls='--', lw=1.1, alpha=0.6)
            ax.axvline(mu1,  color=c_main,  ls='-',  lw=1.0, alpha=0.55)
            ax.axvline(mu2,  color=c_sec,   ls='-',  lw=0.9, alpha=0.45)

            # Titre
            ax.set_title(f"{r1['Line']} + blend", fontsize=8.5,
                         fontweight='bold', color=c_title)

            # Boite info : 1 vitesse, 2 SNR
            if not clean_mode:
                vel = _C_KMS * (mu1 - lth1) / lth1
                ax.text(0.97, 0.97,
                        f"v = {vel:+.1f} km/s\n"
                        f"SNR({r1['Line']}) = {snr1:.1f}\n"
                        f"SNR(blend) = {snr2:.1f}",
                        transform=ax.transAxes, fontsize=7.5,
                        va='top', ha='right',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec=c_main, alpha=0.88))

        # =====================================================================
        # UNRESOLVED -- 2 raies individuelles, 2 vitesses, 1 SNR commun
        # =====================================================================
        elif ft == 'UNRESOLVED':
            c_r1, c_r2 = 'dodgerblue', 'darkorange'

            # Continuum moyen
            m_avg = (m1 + m2) / 2.0
            c_avg = (c1v + c2v) / 2.0
            cont  = m_avg * x_plot + c_avg

            g1 = amp1 * np.exp(-(x_plot - mu1)**2 / (2 * sig1**2))
            g2 = amp2 * np.exp(-(x_plot - mu2)**2 / (2 * sig2**2))

            # Shading zone blend
            ax.axvspan(min(lth1, lth2) - 0.4 * max(sig1, sig2),
                       max(lth1, lth2) + 0.4 * max(sig1, sig2),
                       color='orange', alpha=0.07, zorder=0)

            # Positions theoriques
            ax.axvline(lth1, color=c_r1, ls='--', lw=1.4, alpha=0.70)
            ax.axvline(lth2, color=c_r2, ls='--', lw=1.4, alpha=0.70)

            # Pic gaussien commun (mu du fit -- identique si fit unique, sinon proche)
            mu_pic = (mu1 + mu2) / 2.0  # centre observe
            ax.axvline(mu_pic, color='black', ls='-', lw=1.1, alpha=0.50)

            # Courbes
            ax.plot(x_plot, cont,           ls=':', lw=1.0, color='gray', alpha=0.5)
            ax.plot(x_plot, cont + g1,      ls='--', lw=1.8, color=c_r1, alpha=0.85)
            ax.plot(x_plot, cont + g2,      ls='--', lw=1.8, color=c_r2, alpha=0.85)
            ax.plot(x_plot, cont + g1 + g2, ls='-',  lw=2.4, color='black', alpha=0.65)

            # Accolade entre lth1 et lth2
            y_top   = np.max(y_obs)
            y_brace = y_top + 0.05 * y_span
            b_tick  = 0.018 * y_span
            ax.annotate("",
                        xy=(max(lth1, lth2), y_brace),
                        xytext=(min(lth1, lth2), y_brace),
                        arrowprops=dict(arrowstyle="-", color='dimgray',
                                        lw=1.6,
                                        connectionstyle="bar,fraction=0.0"),
                        zorder=8)
            for xb in [lth1, lth2]:
                ax.plot([xb, xb], [y_brace - b_tick, y_brace + b_tick],
                        color='dimgray', lw=1.6, zorder=8)

            # Titre
            sep_a = abs(lth2 - lth1) * 1e4
            ax.set_title(f"{r1['Line']} / {r2['Line']}  "
                         f"({sep_a:.1f} A)",
                         fontsize=8.5, fontweight='bold', color=c_title)

            # Boite info : v(raie1) et v(raie2) par rapport au meme pic mu_pic,
            # et SNR commun (pic unique)
            if not clean_mode:
                # Pic utilise = mu du fit individuel de chaque raie
                vel1 = _C_KMS * (mu1 - lth1) / lth1
                vel2 = _C_KMS * (mu2 - lth2) / lth2
                snr_common = (snr1 + snr2) / 2.0   # moyenne des deux fits

                # Boite raie 1 -- gauche
                ax.text(0.02, 0.97,
                        f"{r1['Line']}\n"
                        f"$\\lambda_{{th}}$={lth1:.4f}\n"
                        f"v = {vel1:+.1f} km/s",
                        transform=ax.transAxes, fontsize=7.2,
                        va='top', ha='left', color=c_r1,
                        bbox=dict(boxstyle="round,pad=0.28", fc="white",
                                  ec=c_r1, alpha=0.90), zorder=10)

                # Boite raie 2 -- droite
                ax.text(0.98, 0.97,
                        f"{r2['Line']}\n"
                        f"$\\lambda_{{th}}$={lth2:.4f}\n"
                        f"v = {vel2:+.1f} km/s",
                        transform=ax.transAxes, fontsize=7.2,
                        va='top', ha='right', color=c_r2,
                        bbox=dict(boxstyle="round,pad=0.28", fc="white",
                                  ec=c_r2, alpha=0.90), zorder=10)

                # SNR commun -- bas centre
                ax.text(0.50, 0.03,
                        f"SNR = {snr_common:.1f}",
                        transform=ax.transAxes, fontsize=7.5,
                        va='bottom', ha='center',
                        bbox=dict(boxstyle="round,pad=0.22", fc="lightyellow",
                                  ec='goldenrod', alpha=0.88), zorder=10)

        # ── Finalisation subplot ─────────────────────────────────────────────
        ax.set_xlim(w_min, w_max)
        ax.grid(True, alpha=0.13, lw=0.5)

        if clean_mode:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.set_xlabel(r"$\lambda$ (um)", fontsize=7)
            ax.tick_params(axis='x', labelrotation=30)

    # ── Supprimer les axes vides ─────────────────────────────────────────────
    for j in range(n_plots, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # ── Legende globale des categories ───────────────────────────────────────
    if not clean_mode:
        legend_handles = [
            Line2D([0], [0], color='crimson',    lw=2, label='SIMPLE'),
            Line2D([0], [0], color='magenta',    lw=2, label='FORCED'),
            Line2D([0], [0], color='royalblue',  lw=2, label='BLEND'),
            Line2D([0], [0], color='purple',     lw=2, label='DOUBLE  (v unique, 2 SNR)'),
            mpatches.Patch(facecolor='dodgerblue', alpha=0.6,
                           label='UNRESOLVED raie 1'),
            mpatches.Patch(facecolor='darkorange', alpha=0.6,
                           label='UNRESOLVED raie 2  (2 v, 1 SNR)'),
        ]
        fig.legend(handles=legend_handles,
                   loc='lower center', ncol=6,
                   fontsize=8, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()
    return fig, axes


# =============================================================================
# TAG UNRESOLVED
# =============================================================================

def mark_unresolved_pairs(df_merged, unresolved_pairs):
    """
    Dans df_merged (deja fusionee), retague les raies des paires non-resolues :
      fit_type    --> 'UNRESOLVED'
      blended_with --> nom de la raie partenaire

    Les parametres gaussiens (amplitude, sigma, continuum_m/c, Int_cgs...)
    restent ceux du fit individuel. Seul le tag change.

    Parametres
    ----------
    df_merged        : DataFrame issu de build_df_merged_for_plot
    unresolved_pairs : liste de tuples (line1, line2)

    Retour
    ------
    Copie de df_merged avec les colonnes 'fit_type' et 'blended_with' mises a jour.
    """
    df = df_merged.copy()

    if 'blended_with' not in df.columns:
        df['blended_with'] = np.nan

    missing = []
    n_tagged = 0
    for line1, line2 in unresolved_pairs:
        for ln, partner in [(line1, line2), (line2, line1)]:
            mask = df['Line'] == ln
            if mask.sum() == 0:
                missing.append(ln)
            else:
                df.loc[mask, 'fit_type']     = 'UNRESOLVED'
                df.loc[mask, 'blended_with'] = partner
                n_tagged += mask.sum()

    if missing:
        print(f"[WARN] mark_unresolved_pairs : raies absentes de df_merged : "
              f"{list(set(missing))}")
    print(f"[mark_unresolved_pairs] {n_tagged} raies taguees UNRESOLVED "
          f"({len(unresolved_pairs)} paires).")
    return df