#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Librairie d'Analyse Spectrale JWST - NIRSpec
Méthodes de fit Gaussien (Local et Global robuste), Continua et Visualisations.
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import curve_fit, least_squares
from astropy.modeling import models, fitting
from astropy.stats import mad_std

# --- Constantes Physiques ---
C_MICRON_S = 2.99792458e14
C_KMS = 299792.458


# =============================================================================
# 1. UTILITAIRES ET BRUIT
# =============================================================================

def filter_fitted_lines(df, col_vrad="Velocity_km_s", min_vel=None, max_vel=None, col_snr="SNR_noise", min_snr=None):
    """ Filtre le DataFrame selon la vitesse radiale et le SNR. """
    df_filtered = df.copy()
    if min_vel is not None:
        df_filtered = df_filtered[df_filtered[col_vrad] >= min_vel]
    if max_vel is not None:
        df_filtered = df_filtered[df_filtered[col_vrad] <= max_vel]
    if min_snr is not None:
        df_filtered = df_filtered[df_filtered[col_snr] >= min_snr]
    return df_filtered

def estimate_local_noise(df_spectrum, col_wave="Wavelength", col_flux="Int", wave_min=None, wave_max=None):
    """ Estime le bruit RMS sur une section plate du spectre via détendring linéaire. """
    mask = (df_spectrum[col_wave] >= wave_min) & (df_spectrum[col_wave] <= wave_max)
    df_noise = df_spectrum.loc[mask].copy()
    
    if len(df_noise) < 3:
        raise ValueError("Section de bruit trop petite.")
        
    x, y = df_noise[col_wave].values, df_noise[col_flux].values
    coeffs = np.polyfit(x, y, 1)
    continuum_local = coeffs[0] * x + coeffs[1]
    
    residuals = y - continuum_local
    noise_rms = np.std(residuals, ddof=1)
    mean_background = np.mean(y)
    
    df_noise['Continuum_Fit'] = continuum_local
    df_noise['Residuals'] = residuals
    
    return noise_rms, mean_background, df_noise


# =============================================================================
# 2. CONTINUUM GLOBAL
# =============================================================================

def fit_continuum_with_history(df, wave_col="Wavelength", flux_col="Int", degree=5, sigma_upper=1.5, sigma_lower=5.0, maxiters=20):
    """ Ajuste un continuum global par un polynôme de Tchebychev et un écrêtage itératif (Sigma-Clipping). """
    df_sub = df.dropna(subset=[wave_col, flux_col]).copy()
    x, y = df_sub[wave_col].values, df_sub[flux_col].values

    fitter = fitting.LinearLSQFitter()
    current_model = models.Chebyshev1D(degree=degree, domain=[x.min(), x.max()])
    
    history_models = [] 
    mask = np.zeros(len(x), dtype=bool) 

    for i in range(maxiters):
        current_model = fitter(current_model, x[~mask], y[~mask])
        history_models.append(current_model.copy())
        
        residuals = y - current_model(x)
        std_dev = mad_std(residuals[~mask])
        
        if std_dev == 0: break 
            
        new_mask = (residuals > sigma_upper * std_dev) | (residuals < -sigma_lower * std_dev)
        if np.array_equal(new_mask, mask): break
        mask = new_mask

    df_sub['Continuum'] = current_model(x)
    return df_sub, history_models

def plot_continuum_multiline(df_spectrum, history_models, wl_min=4.57, wl_max=4.964, 
                             ylim_min=80, ylim_max=200, n_plots=2, show_history=True):
    """
    Affiche le spectre, l'historique des itérations du fit, et met en valeur
    les raies d'émission rejetées par le sigma-clipping, répartis sur plusieurs lignes.
    """
    delta_wl = (wl_max - wl_min) / n_plots
    fig, axes = plt.subplots(n_plots, 1, figsize=(20, 6 * n_plots))
    
    # Sécurité si n_plots = 1 (axes n'est pas une liste par défaut)
    if n_plots == 1: 
        axes = [axes]
        
    n_iters = len(history_models)

    for i, ax in enumerate(axes):
        # Définition de la fenêtre de longueur d'onde pour ce sous-graphique
        c_min = wl_min + i * delta_wl
        c_max = wl_min + (i + 1) * delta_wl
        
        mask_range = (df_spectrum["Wavelength"] >= c_min) & (df_spectrum["Wavelength"] <= c_max)
        df_plot = df_spectrum[mask_range]
        
        if df_plot.empty:
            continue
            
        x = df_plot["Wavelength"].values
        y = df_plot["Int"].values
        
        # 1. Tracé du spectre de base (step)
        label_spectre = "Spectre brut" if i == 0 else ""
        ax.step(x, y, "-", c="black", alpha=0.7, where="mid", linewidth=1.5, label=label_spectre, zorder=1)
        
        # 2. Tracé des points rejetés (les raies identifiées)
        if 'Is_Line' in df_plot.columns:
            x_lines = x[df_plot['Is_Line']]
            y_lines = y[df_plot['Is_Line']]
            if len(x_lines) > 0:
                label_rejet = "Points rejetés (raies)" if i == 0 else ""
                ax.scatter(x_lines, y_lines, color="blue", s=15, zorder=4, 
                           label=label_rejet, alpha=0.6)

        # 3. Tracé de l'historique des itérations
        if show_history and n_iters > 1:
            for j, model in enumerate(history_models[:-1]):
                # Opacité progressive : de plus en plus visible
                alpha_val = 0.2 + 0.6 * (j / (n_iters - 1))
                label_hist = "Itérations intermédiaires" if (i == 0 and j == 0) else ""
                # Les modèles stockés sont appelables : model(x)
                ax.plot(x, model(x), "--", c="gray", alpha=alpha_val, linewidth=1.5, label=label_hist, zorder=2)

        # 4. Tracé du continuum final (en rouge)
        if n_iters > 0:
            label_cont = "Continuum final" if i == 0 else ""
            ax.plot(x, history_models[-1](x), "-", c="red", alpha=1.0, 
                    linewidth=3, label=label_cont, zorder=5)

        # Mise en forme du sous-graphique
        ax.set_xlim([c_min, c_max])
        ax.set_ylim([ylim_min, ylim_max])
        ax.set_ylabel("Surface brightness [MJy/sr]", size=16)
        ax.grid(linestyle="--", alpha=0.5)
        ax.tick_params(axis='both', labelsize=12)
        
        # Légende et titre uniquement sur le premier graphique
        if i == 0:
            ax.legend(loc='upper right', fontsize=14, framealpha=0.9, facecolor='white')
            ax.set_title(f"Ajustement du continuum par Sigma-Clipping Itératif ({n_iters} itérations)", 
                         fontsize=18, fontweight='bold', pad=15)

    # Label X uniquement sur le dernier graphique pour ne pas surcharger
    axes[-1].set_xlabel("Wavelength [microns]", size=16)
    
    fig.tight_layout()
    return fig, axes


def plot_local_noise(df_noise, noise_rms, col_wave="Wavelength", col_flux="Int"):
    """ Trace la région de bruit, le continuum affine ajusté, et les bandes RMS. """
    fig, ax = plt.subplots(figsize=(10, 6))
    x, y = df_noise[col_wave].values, df_noise[col_flux].values
    continuum = df_noise['Continuum_Fit'].values
    
    ax.step(x, y, where='mid', color='black', label='Données (Bruit pur)', linewidth=1.5, alpha=0.8)
    ax.plot(x, continuum, color='blue', linestyle='--', linewidth=2, label='Continuum fitté')
    
    ax.fill_between(x, continuum - noise_rms, continuum + noise_rms, color='blue', alpha=0.3, label=r'$\pm 1 \sigma_{bruit}$')
    ax.fill_between(x, continuum - 3*noise_rms, continuum + 3*noise_rms, color='blue', alpha=0.1, label=r'$\pm 3 \sigma_{bruit}$')
    
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


def plot_final_summary(df_spectrum, df_merged, global_model, wl_range=(4.55, 4.93), ylim=(100, 240), 
                       n_plots=3, type_plot="step"):
    """
    Master Plot Rigoureux : 
    - Affiche TOUTES les composantes (individuelles en pointillés + aire, somme en plein).
    - Local = Bleu (avec remplissage semi-transparent). Global = Vert.
    - InfoBox indépendante pour chaque composante d'un blend.
    """
    wl_min, wl_max = wl_range
    delta_wl = (wl_max - wl_min) / n_plots
    fig, axes = plt.subplots(n_plots, 1, figsize=(25, 9 * n_plots))
    if n_plots == 1: axes = [axes]
    
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    for i, ax in enumerate(axes):
        ax.set_ylim(ylim)
        c_min, c_max = wl_min + i * delta_wl, wl_min + (i + 1) * delta_wl
        plot_width = c_max - c_min
        
        # 1. Tracé du spectre
        mask = (df_spectrum["Wavelength"] >= c_min) & (df_spectrum["Wavelength"] <= c_max)
        if mask.sum() == 0: continue
        w_v, f_v = df_spectrum.loc[mask, "Wavelength"].values, df_spectrum.loc[mask, "Int"].values
        
        if type_plot == "step":
            ax.step(w_v, f_v, color="black", lw=1.2, alpha=0.9, where='mid', zorder=1)
        else:
            ax.plot(w_v, f_v, color="black", lw=1.2, alpha=0.9, zorder=1)

        # 2. Continuum Global
        x_g = np.linspace(c_min, c_max, 1000)
        ax.plot(x_g, global_model(x_g), color="red", lw=1.5, alpha=0.7, zorder=2)

        subs = df_merged[(df_merged['wavelength_obs'] >= c_min) & (df_merged['wavelength_obs'] <= c_max)].sort_values('wavelength_obs')
        
        last_x, tog = -999, 0
        y_range = ylim[1] - ylim[0]
        y_lvls = [ylim[0] + 0.90 * y_range, ylim[0] + 0.75 * y_range, ylim[0] + 0.60 * y_range, ylim[0] + 0.45 * y_range]

        for _, r in subs.iterrows():
            mu, sig, A = r['wavelength_obs'], r['sigma'], r['amplitude']
            m, c = r['continuum_m'], r['continuum_c']
            
            A_g, s_g = r.get('amp_global', A), r.get('sig_global', sig)
            has_blend = pd.notna(r.get('amp_blend')) and r.get('amp_blend', 0) > 0
            
            if has_blend:
                x_m = np.linspace(min(mu - 8*sig, r['mu_blend'] - 8*r['sig_blend']), 
                                  max(mu + 8*sig, r['mu_blend'] + 8*r['sig_blend']), 500)
            else:
                x_m = np.linspace(mu - 8*sig, mu + 8*sig, 400)

            c_loc = m * x_m + c
            c_glob = global_model(x_m)
            
            # --- COMPOSANTES LOCALES ---
            g1_loc = A * np.exp(-(x_m - mu)**2 / (2 * sig**2))
            g2_loc = np.zeros_like(x_m)
            if has_blend:
                g2_loc = r['amp_blend'] * np.exp(-(x_m - r['mu_blend'])**2 / (2 * r['sig_blend']**2))

            # --- COMPOSANTES GLOBALES ---
            g1_glob = A_g * np.exp(-(x_m - mu)**2 / (2 * s_g**2))
            g2_glob = np.zeros_like(x_m)
            if has_blend:
                A2_g, s2_g = r.get('amp_blend_global', r['amp_blend']), r.get('sig_blend_global', r['sig_blend'])
                g2_glob = A2_g * np.exp(-(x_m - r['mu_blend'])**2 / (2 * s2_g**2))

            # ==============================================================
            # TRACÉS
            # ==============================================================
            
            # LOCAL (Bleu)
            ax.plot(x_m, c_loc, color='blue', ls=':', lw=1.2, alpha=0.8, zorder=3)
            
            # Individuelles locales (Pointillés + Remplissage)
            ax.fill_between(x_m, c_loc, c_loc + g1_loc, color='dodgerblue', alpha=0.25, zorder=3)
            ax.plot(x_m, c_loc + g1_loc, color='blue', ls='--', lw=1.5, alpha=0.9, zorder=4)
            if has_blend:
                ax.fill_between(x_m, c_loc, c_loc + g2_loc, color='deepskyblue', alpha=0.25, zorder=3)
                ax.plot(x_m, c_loc + g2_loc, color='deepskyblue', ls='--', lw=1.5, alpha=0.9, zorder=4)
                
            # Somme locale (Trait plein)
            ax.plot(x_m, c_loc + g1_loc + g2_loc, color='darkblue', ls='-', lw=2, alpha=0.9, zorder=5)

            # GLOBAL (Vert)
            y_glob_tot = c_glob + g1_glob + g2_glob
            
            # Individuelles globales (Pointillés + Remplissage)
            ax.fill_between(x_m, c_glob, c_glob + g1_glob, color='mediumseagreen', alpha=0.35, zorder=3)
            ax.plot(x_m, c_glob + g1_glob, color='seagreen', ls='--', lw=1.5, alpha=0.9, zorder=4)
            if has_blend:
                ax.fill_between(x_m, c_glob, c_glob + g2_glob, color='lightgreen', alpha=0.35, zorder=3)
                ax.plot(x_m, c_glob + g2_glob, color='limegreen', ls='--', lw=1.5, alpha=0.9, zorder=4)
                
            # Somme globale (Trait plein)
            ax.plot(x_m, y_glob_tot, color='darkgreen', ls='-', lw=2.5, zorder=6)


            # --- GESTION DES BOÎTES D'ANNOTATION ---
            def draw_box(x_pos, peak_pos, line_n, i_loc, i_glob, err, snr):
                nonlocal last_x, tog
                if abs(x_pos - last_x) < (plot_width * 0.04): tog = (tog + 1) % 4
                else: tog = 0
                last_x = x_pos
                
                lbl = f"{line_n}\nI_loc: {i_loc:.1e}\nI_glb: {i_glob:.1e}\n dI: {err:.1e} / SNR: {snr:.1f}"
                ax.annotate(lbl, xy=(x_pos, peak_pos), xytext=(x_pos, y_lvls[tog]), 
                            arrowprops=dict(arrowstyle='-', color='black', alpha=0.5, lw=1),
                            fontsize=8.5, color='black', ha='center', va='bottom', rotation=0, 
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.9, zorder=20))

            # Boîte Principale
            peak_y1 = c_loc[np.argmin(np.abs(x_m - mu))] + A
            draw_box(mu, peak_y1, r['Line'], r.get('Int_cgs',0), r.get('Int_cgs_global',0), r.get('Int_err_final',0), r.get('SNR_noise',0))
            
            # Boîte Blend (si existante)
            if has_blend:
                mu2 = r['mu_blend']
                peak_y2 = c_loc[np.argmin(np.abs(x_m - mu2))] + r['amp_blend']
                nom_blend = r.get('blended_with') if pd.notna(r.get('blended_with')) else f"{r['Line']} (Blend)"
                
                # Calcul de l'intensité approximative du blend pour la boîte
                C_MICRON_S = 2.99792458e14
                i_loc2 = (r['amp_blend'] * r['sig_blend'] * np.sqrt(2 * np.pi)) * ((C_MICRON_S / (mu2**2)) * 1e-17)
                i_glb2 = (A2_g * s2_g * np.sqrt(2 * np.pi)) * ((C_MICRON_S / (mu2**2)) * 1e-17)
                err2 = abs(i_glb2 - i_loc2) / 2.0
                
                draw_box(mu2, peak_y2, nom_blend, i_loc2, i_glb2, err2, r.get('SNR_noise',0))

        ax.set_xlim(c_min, c_max)
        ax.set_ylabel("Surface brightness [MJy/sr]", size=14)
        ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)

        if i == 0:
            custom = [Line2D([0],[0], color='black', lw=1.5, ls='-'), 
                      Line2D([0],[0], color='red', lw=1.5, alpha=0.7),
                      Line2D([0],[0], color='blue', ls='--', lw=1.5), 
                      Line2D([0],[0], color='darkblue', ls='-', lw=2), 
                      Line2D([0],[0], color='seagreen', ls='--', lw=1.5), 
                      Line2D([0],[0], color='darkgreen', ls='-', lw=2.5),
                      Patch(facecolor='dodgerblue', alpha=0.25),
                      Patch(facecolor='mediumseagreen', alpha=0.35)]
            ax.legend(custom, ['NIRSpec Spectrum', 'Global continuum', 'Local continuum', 'Local gaussian', '', 'Global gaussian', 'Local area', 'Global area'], 
                      loc='upper right', fontsize=10, framealpha=0.95, ncol=4)

    axes[-1].set_xlabel(r"Wavelength $\lambda$ [$\mu m$]", size=14)
    plt.tight_layout()
    return fig, axes


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_publication_spectrum(
    df_spectrum, 
    df_merged, 
    global_model, 
    wl_range=(4.55, 4.93), 
    ylim=(100, 240), 
    n_plots=2, 
    type_plot="step",
    # --- Configuration de la Figure & Style Global ---
    figsize=(22, 13),      
    font_family="serif",   
    fontsize_labels=18,    # Titres des axes encore plus grands
    fontsize_ticks=16,     # Graduations d'axes (chiffres) très grands
    fontsize_legend=13,    
    fontsize_box=9.5,      # Texte légèrement plus condensé pour limiter la taille des boîtes
    # --- Épaisseurs de lignes ---
    lw_spec=1.0,           
    lw_global_cont=1.2,    
    lw_fit_indiv=1.2,      
    lw_fit_sum=2.0,        
    # --- Transparences (Alphas) ---
    alpha_spec=0.8,
    alpha_fill_loc=0.15,   
    alpha_fill_glob=0.20,  
    alpha_cont=0.6,
    # --- Palettes de Couleurs ---
    color_spec="black",
    color_global_cont="crimson",
    color_loc_sum="darkblue",
    color_loc_indiv="dodgerblue",
    color_glob_sum="forestgreen",
    color_glob_indiv="mediumseagreen"
):
    """
    Plots a highly polished spectral summary with an advanced 1D collision-avoidance 
    algorithm for annotation boxes. Labels are placed dynamically to never overlap.
    """
    plt.rcParams['font.family'] = font_family
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    
    wl_min, wl_max = wl_range
    delta_wl = (wl_max - wl_min) / n_plots
    
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=False)
    if n_plots == 1: 
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.set_ylim(ylim)
        c_min, c_max = wl_min + i * delta_wl, wl_min + (i + 1) * delta_wl
        plot_width = c_max - c_min
        
        # Ticks et sous-ticks beaucoup plus épais et visibles
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks, length=10, width=1.5)
        ax.tick_params(axis='both', which='minor', length=5, width=1.0)
        
        # 1. Tracé du spectre
        mask = (df_spectrum["Wavelength"] >= c_min) & (df_spectrum["Wavelength"] <= c_max)
        if mask.sum() == 0: 
            continue
        w_v, f_v = df_spectrum.loc[mask, "Wavelength"].values, df_spectrum.loc[mask, "Int"].values
        
        if type_plot == "step":
            ax.step(w_v, f_v, color=color_spec, lw=lw_spec, alpha=alpha_spec, where='mid', zorder=1)
        else:
            ax.plot(w_v, f_v, color=color_spec, lw=lw_spec, alpha=alpha_spec, zorder=1)

        # 2. Continuum Global
        x_g = np.linspace(c_min, c_max, 1000)
        ax.plot(x_g, global_model(x_g), color=color_global_cont, lw=lw_global_cont, alpha=alpha_cont, zorder=2)

        # 3. Composantes et fits
        subs = df_merged[(df_merged['wavelength_obs'] >= c_min) & (df_merged['wavelength_obs'] <= c_max)].sort_values('wavelength_obs')
        
        # --- ALGORITHME ANTI-COLLISION ROBUSTE ---
        # On définit plusieurs paliers verticaux d'accueil pour les boîtes
        y_range = ylim[1] - ylim[0]
        y_levels = [
            ylim[0] + 0.90 * y_range, 
            ylim[0] + 0.79 * y_range, 
            ylim[0] + 0.68 * y_range, 
            ylim[0] + 0.57 * y_range,
            ylim[0] + 0.46 * y_range
        ]
        
        # Historique des boîtes placées sur ce subplot : list of dict {'x_min': , 'x_max': , 'level_idx': }
        # Une boîte à l'abscisse x_pos a une certaine largeur physique estimée à environ 3.8% de la largeur du graphique.
        placed_boxes = []
        box_half_width = plot_width * 0.019  

        def get_best_level(x_pos):
            box_xmin = x_pos - box_half_width
            box_xmax = x_pos + box_half_width
            
            # Essayer chaque niveau du haut vers le bas
            for lvl_idx in range(len(y_levels)):
                collision = False
                for pb in placed_boxes:
                    if pb['level_idx'] == lvl_idx:
                        # Si chevauchement horizontal sur le même niveau vertical -> collision
                        if not (box_xmax < pb['x_min'] or box_xmin > pb['x_max']):
                            collision = True
                            break
                if not collision:
                    # Trouvé un niveau libre à cette coordonnée x !
                    placed_boxes.append({'x_min': box_xmin, 'x_max': box_xmax, 'level_idx': lvl_idx})
                    return y_levels[lvl_idx]
            
            # Si tous les niveaux sont pris (rare), on force le premier et on applique un micro-décalage horizontal
            placed_boxes.append({'x_min': box_xmin, 'x_max': box_xmax, 'level_idx': 0})
            return y_levels[0]

        for _, r in subs.iterrows():
            mu, sig, A = r['wavelength_obs'], r['sigma'], r['amplitude']
            m, c = r['continuum_m'], r['continuum_c']
            
            A_g, s_g = r.get('amp_global', A), r.get('sig_global', sig)
            has_blend = pd.notna(r.get('amp_blend')) and r.get('amp_blend', 0) > 0
            
            if has_blend:
                x_m = np.linspace(min(mu - 8*sig, r['mu_blend'] - 8*r['sig_blend']), 
                                  max(mu + 8*sig, r['mu_blend'] + 8*r['sig_blend']), 500)
            else:
                x_m = np.linspace(mu - 8*sig, mu + 8*sig, 400)

            c_loc = m * x_m + c
            c_glob = global_model(x_m)
            
            # --- Composantes Locales ---
            g1_loc = A * np.exp(-(x_m - mu)**2 / (2 * sig**2))
            g2_loc = np.zeros_like(x_m)
            if has_blend:
                g2_loc = r['amp_blend'] * np.exp(-(x_m - r['mu_blend'])**2 / (2 * r['sig_blend']**2))

            # --- Composantes Globales ---
            g1_glob = A_g * np.exp(-(x_m - mu)**2 / (2 * s_g**2))
            g2_glob = np.zeros_like(x_m)
            if has_blend:
                A2_g, s2_g = r.get('amp_blend_global', r['amp_blend']), r.get('sig_blend_global', r['sig_blend'])
                g2_glob = A2_g * np.exp(-(x_m - r['mu_blend'])**2 / (2 * s2_g**2))

            # --- Tracé Local (Bleu) ---
            ax.plot(x_m, c_loc, color=color_loc_sum, ls=':', lw=0.8, alpha=alpha_cont, zorder=3)
            ax.fill_between(x_m, c_loc, c_loc + g1_loc, color=color_loc_indiv, alpha=alpha_fill_loc, zorder=3)
            ax.plot(x_m, c_loc + g1_loc, color=color_loc_indiv, ls='--', lw=lw_fit_indiv, alpha=0.8, zorder=4)
            if has_blend:
                ax.fill_between(x_m, c_loc, c_loc + g2_loc, color=color_loc_indiv, alpha=alpha_fill_loc, zorder=3)
                ax.plot(x_m, c_loc + g2_loc, color=color_loc_indiv, ls='--', lw=lw_fit_indiv, alpha=0.8, zorder=4)
                
            ax.plot(x_m, c_loc + g1_loc + g2_loc, color=color_loc_sum, ls='-', lw=lw_fit_sum, alpha=0.9, zorder=5)

            # --- Tracé Global (Vert) ---
            y_glob_tot = c_glob + g1_glob + g2_glob
            ax.fill_between(x_m, c_glob, c_glob + g1_glob, color=color_glob_indiv, alpha=alpha_fill_glob, zorder=3)
            if has_blend:
                ax.fill_between(x_m, c_glob, c_glob + g2_glob, color=color_glob_indiv, alpha=alpha_fill_glob, zorder=3)
                
            ax.plot(x_m, y_glob_tot, color=color_glob_sum, ls='-', lw=lw_fit_sum + 0.3, zorder=6)

            # --- Boîtes d'annotation améliorées ---
            def draw_box(x_pos, peak_pos, line_n, i_loc, i_glob, err):
                # Récupération dynamique du meilleur niveau vertical sans collision
                y_target = get_best_level(x_pos)
                
                i_moy = (i_loc + i_glob) / 2.0
                
                # Format ultra-condensé et lisible
                lbl = f"{line_n}\n$I_{{loc}}$: {i_loc:.1e}\n$I_{{glb}}$: {i_glob:.1e}\n$dI$: {err:.1e}\n$I_{{moy}}$: {i_moy:.1e}"
                
                ax.annotate(
                    lbl, xy=(x_pos, peak_pos), xytext=(x_pos, y_target), 
                    arrowprops=dict(arrowstyle='-', color='black', alpha=0.2, lw=0.6),
                    fontsize=fontsize_box, color='black', ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="darkgray", lw=0.6, alpha=0.95, zorder=20)
                )

            # Annotation principale
            peak_y1 = c_loc[np.argmin(np.abs(x_m - mu))] + A
            draw_box(mu, peak_y1, r['Line'], r.get('Int_cgs',0), r.get('Int_cgs_global',0), r.get('Int_err_final',0))
            
            # Annotation Blend
            if has_blend:
                mu2 = r['mu_blend']
                peak_y2 = c_loc[np.argmin(np.abs(x_m - mu2))] + r['amp_blend']
                nom_blend = r.get('blended_with') if pd.notna(r.get('blended_with')) else f"{r['Line']} (B)"
                
                C_MICRON_S = 2.99792458e14
                i_loc2 = (r['amp_blend'] * r['sig_blend'] * np.sqrt(2 * np.pi)) * ((C_MICRON_S / (mu2**2)) * 1e-17)
                i_glb2 = (A2_g * s2_g * np.sqrt(2 * np.pi)) * ((C_MICRON_S / (mu2**2)) * 1e-17)
                err2 = abs(i_glb2 - i_loc2) / 2.0
                
                draw_box(mu2, peak_y2, nom_blend, i_loc2, i_glb2, err2)

        ax.set_xlim(c_min, c_max)
        ax.set_ylabel(r"$I_\nu$ [MJy / sr]", size=fontsize_labels)

        # --- AJOUT DE LA LÉGENDE SUR LE DEUXIÈME GRAPHIQUE (Index i == 1) ---
        if i == 1:
            legend_elements = [
                Line2D([0], [0], color=color_spec, lw=lw_spec, ls='-'), 
                Line2D([0], [0], color=color_global_cont, lw=lw_global_cont, ls='-'),
                Line2D([0], [0], color=color_loc_indiv, lw=lw_fit_indiv, ls='--'), 
                Line2D([0], [0], color=color_loc_sum, lw=lw_fit_sum, ls='-'), 
                Line2D([0], [0], color=color_glob_sum, lw=lw_fit_sum, ls='-'),
                Patch(facecolor=color_loc_indiv, alpha=alpha_fill_loc, edgecolor='none'),
                Patch(facecolor=color_glob_indiv, alpha=alpha_fill_glob, edgecolor='none')
            ]
            legend_labels = [
                'Spectrum', 'Global Continuum', 
                'Local Comp.', 'Local Fit', 'Global Fit', 
                'Local Area', 'Global Area'
            ]
            ax.legend(
                legend_elements, legend_labels, 
                loc='upper right', fontsize=fontsize_legend, 
                framealpha=0.95, frameon=True, edgecolor='darkgray',
                ncol=4, columnspacing=1.0, handletextpad=0.5
            )

    axes[-1].set_xlabel(r"Observed Wavelength $\lambda$ [$\mu\mathrm{m}$]", size=fontsize_labels)
    plt.tight_layout()
    return fig, axes

# =============================================================================
# 3. FITS LOCAUX (Gaussienne + Continuum affine local)
# =============================================================================

def model_gauss_linear(x, A, mu, sigma, m, c):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + m * x + c

def model_frozen_cont(x, A, mu, sigma, m_fixed, c_fixed):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + m_fixed * x + c_fixed

def fit_manual_lines_local(df_spectrum, df_lines_ref, target_names, manual_guesses={}, R=2700, window_width=0.008, plot=False):
    """ Fit local d'une gaussienne simple avec possibilité d'imposer pente et ordonnée. """
    df_subset = df_lines_ref[df_lines_ref['Line'].isin(target_names)].copy()
    if df_subset.empty: return pd.DataFrame(), None, None
    df_subset['Line'] = pd.Categorical(df_subset['Line'], categories=target_names, ordered=True)
    df_subset = df_subset.sort_values('Line').reset_index(drop=True)

    wave, flux = df_spectrum["Wavelength"].values, df_spectrum["Int"].values
    results = []
    fwhm_factor = 2 * np.sqrt(2 * np.log(2))

    for _, row in df_subset.iterrows():
        line_id, l0 = row['Line'], row['wavelength']
        mask = (wave >= l0 - window_width / 2) & (wave <= l0 + window_width/2)
        x_loc, y_loc = wave[mask], flux[mask]
        if len(x_loc) < 5: continue

        sig_g = l0 / R / fwhm_factor
        amp_g = np.max(y_loc) - np.min(y_loc)
        m_g = (y_loc[-1] - y_loc[0]) / (x_loc[-1] - x_loc[0])
        c_g = np.mean(y_loc) - (m_g * np.mean(x_loc))

        is_fixed = False
        if line_id in manual_guesses:
            p = manual_guesses[line_id]
            amp_g, sig_g = p.get('amp', amp_g), p.get('sigma', sig_g)
            if 'slope' in p or 'intercept' in p:
                is_fixed = True
                m_fix = p.get('slope', 0.0)
                c_fix = p.get('intercept', np.mean(y_loc)) - (m_fix * l0)

        try:
            if is_fixed:
                popt, _ = curve_fit(lambda x, A, mu, s: model_frozen_cont(x, A, mu, s, m_fix, c_fix), 
                                       x_loc, y_loc, p0=[amp_g, l0, sig_g], maxfev=10000)
                A_f, mu_f, sig_f = popt[0], popt[1], abs(popt[2])
                m_f, c_f = m_fix, c_fix
            else:
                popt, _ = curve_fit(model_gauss_linear, x_loc, y_loc, p0=[amp_g, l0, sig_g, m_g, c_g], maxfev=10000)
                A_f, mu_f, sig_f, m_f, c_f = popt[0], popt[1], abs(popt[2]), popt[3], popt[4]

            flux_cgs = (A_f * sig_f * np.sqrt(2 * np.pi)) * ((C_MICRON_S / (mu_f**2)) * 1e-17)
            results.append({
                'Line': line_id, 'wavelength_th': l0, 'wavelength_obs': mu_f,
                'amplitude': A_f, 'sigma': sig_f, 'continuum_m': m_f, 'continuum_c': c_f, 
                'Velocity_km_s': C_KMS * (mu_f - l0) / l0, 'Int_cgs': flux_cgs, 'success': True
            })
        except: pass

    return pd.DataFrame(results), None, None


def analyze_double_lines_flexible(df_spectrum, df_lines_params, R=2700):
    """ Ajuste un blend complet dans une même fenêtre avec paramètres flexibles. """
    wave, flux = df_spectrum["Wavelength"].values, df_spectrum["Int"].values
    results = []

    for _, row in df_lines_params.iterrows():
        l0, line_id = row['wavelength'], row['Line']
        win = row.get('custom_window')
        if isinstance(win, (list, tuple)) and len(win) == 2:
            w_min, w_max = win[0], win[1]
        else:
            w_min, w_max = l0 - row['win_width']/2, l0 + row['win_width']/2

        x_l, y_l = wave[(wave >= w_min) & (wave <= w_max)], flux[(wave >= w_min) & (wave <= w_max)]
        if len(x_l) < 10: continue

        sig_in = (l0 / R / 2.355) * row['width_guess']
        is_fixed_c = pd.notna(row['fixed_m']) and pd.notna(row['fixed_c'])
        f_m, f_c = (row['fixed_m'], row['fixed_c'] - row['fixed_m']*l0) if is_fixed_c else (np.nan, np.nan)
        is_fixed_b = pd.notna(row['fixed_blend_wave'])
        f_mb = row['fixed_blend_wave'] if is_fixed_b else np.nan

        try:
            m_g, c_g = (f_m, f_c) if is_fixed_c else ((y_l[-1]-y_l[0])/(x_l[-1]-x_l[0]), np.mean(y_l))
            cont = m_g * x_l + c_g
            a1 = max(np.max(y_l - cont), np.max(y_l)*0.1)
            
            res = y_l - (cont + a1 * np.exp(-(x_l - l0)**2 / (2 * sig_in**2)))
            mu2_g = f_mb if is_fixed_b else x_l[np.argmax(res)]
            a2 = max(res[np.argmin(np.abs(x_l - mu2_g))], a1*0.3)

            p0 = [a1, l0, sig_in, a2]
            b_l, b_h = [0, x_l[0], 1e-4, 0], [np.inf, x_l[-1], (w_max-w_min), np.inf]
            
            if not is_fixed_b: p0.append(mu2_g); b_l.append(x_l[0]); b_h.append(x_l[-1])
            p0.append(sig_in); b_l.append(1e-4); b_h.append((w_max-w_min))
            if not is_fixed_c: p0.extend([m_g, c_g]); b_l.extend([-np.inf, -np.inf]); b_h.extend([np.inf, np.inf])

            def dynamic_model(x, *args):
                i = 0
                A1, M1, S1, A2 = args[i:i+4]; i += 4
                M2 = f_mb if is_fixed_b else args[i]; i += (0 if is_fixed_b else 1)
                S2 = args[i]; i += 1
                M = f_m if is_fixed_c else args[i]; i += (0 if is_fixed_c else 1)
                C = f_c if is_fixed_c else args[i]
                return A1*np.exp(-(x-M1)**2/(2*S1**2)) + A2*np.exp(-(x-M2)**2/(2*S2**2)) + M*x + C

            popt, _ = curve_fit(dynamic_model, x_l, y_l, p0=p0, bounds=(b_l, b_h), maxfev=10000)

            fp = np.zeros(8); i = 0
            fp[0:4] = popt[i:i+4]; i += 4
            fp[4] = f_mb if is_fixed_b else popt[i]; i += (0 if is_fixed_b else 1)
            fp[5] = popt[i]; i += 1
            fp[6:8] = [f_m, f_c] if is_fixed_c else popt[i:i+2]

            d1, d2 = abs(fp[1] - l0), abs(fp[4] - l0)
            im, ib = (0, 3) if d1 <= d2 else (3, 0)
            
            flux_cgs = (fp[im] * fp[im+2] * np.sqrt(2*np.pi)) * ((C_MICRON_S / (fp[im+1]**2)) * 1e-17)
            results.append({
                'Line': line_id, 'wavelength_th': l0, 'wavelength_obs': fp[im+1],
                'amplitude': fp[im], 'sigma': fp[im+2], 'amp_blend': fp[ib], 'mu_blend': fp[ib+1], 'sig_blend': fp[ib+2],
                'continuum_m': fp[6], 'continuum_c': fp[7], 'Velocity_km_s': C_KMS * (fp[im+1] - l0) / l0,
                'Int_cgs': flux_cgs, 'success': True
            })
        except: pass

    return pd.DataFrame(results)


def analyze_double_lines_split(df_spectrum, df_lines, params_delta, R=2700):
    """ Ajuste une double gaussienne contrainte par un Delta Lambda théorique et sépare le résultat. """
    wave, flux = df_spectrum["Wavelength"].values, df_spectrum["Int"].values
    res = []; l_dict = df_lines.set_index('Line').to_dict('index')

    for p in params_delta:
        L1, L2, w_win, w_g, f_m, f_c, c_win = p
        if L1 not in l_dict or L2 not in l_dict: continue
        lth1, lth2 = l_dict[L1]['wavelength'], l_dict[L2]['wavelength']
        delta_th = lth2 - lth1
        # Remplace la ligne 494 par ce bloc :
        if isinstance(c_win, (list, tuple)) and len(c_win) == 2:
            w_min, w_max = c_win[0], c_win[1]
        else:
            w_min, w_max = lth1 - w_win/2, lth1 + w_win/2
        #w_min, w_max = (c_win[0], c_win[1]) if c_win else (lth1 - w_win/2, lth1 + w_win/2)
        x_l, y_l = wave[(wave >= w_min) & (wave <= w_max)], flux[(wave >= w_min) & (wave <= w_max)]
        if len(x_l) < 10: continue

        sig = (lth1 / R / 2.355) * w_g
        is_f = pd.notna(f_m) and pd.notna(f_c)
        m, c = (f_m, f_c - f_m*lth1) if is_f else (np.nan, np.nan)

        try:
            m_g, c_g = (m, c) if is_f else ((y_l[-1]-y_l[0])/(x_l[-1]-x_l[0]), np.mean(y_l))
            cont = m_g * x_l + c_g
            a1 = max(np.max(y_l - cont), np.max(y_l)*0.1)
            a2 = max(y_l[np.argmin(np.abs(x_l - (lth1+delta_th)))] - cont[np.argmin(np.abs(x_l - (lth1+delta_th)))], a1*0.3)

            p0 = [a1, lth1, sig, a2, sig]
            b_l, b_h = [0, x_l[0], 1e-4, 0, 1e-4], [np.inf, x_l[-1], w_win, np.inf, w_win]
            if not is_f: p0.extend([m_g, c_g]); b_l.extend([-np.inf, -np.inf]); b_h.extend([np.inf, np.inf])

            def mod(x, *a):
                A1, M1, S1, A2, S2 = a[0:5]
                M, C = (f_m, c) if is_f else a[5:7]
                return A1*np.exp(-(x-M1)**2/(2*S1**2)) + A2*np.exp(-(x-(M1+delta_th))**2/(2*S2**2)) + M*x + C

            popt, _ = curve_fit(mod, x_l, y_l, p0=p0, bounds=(b_l, b_h), maxfev=10000)
            A1, M1, S1, A2, S2 = popt[0:5]
            M2 = M1 + delta_th
            M, C = (f_m, c) if is_f else popt[5:7]

            def make_row(Ln, Lb, lth, mu, A, S):
                f_cgs = (A * S * np.sqrt(2*np.pi)) * ((C_MICRON_S / (mu**2)) * 1e-17)
                return {'Line': Ln, 'blended_with': Lb, 'wavelength_th': lth, 'wavelength_obs': mu, 'amplitude': A, 
                        'sigma': S, 'continuum_m': M, 'continuum_c': C, 'Velocity_km_s': C_KMS*(mu-lth)/lth, 'Int_cgs': f_cgs, 'success': True}
            
            res.extend([make_row(L1, L2, lth1, M1, A1, S1), make_row(L2, L1, lth2, M2, A2, S2)])
        except: pass
    return pd.DataFrame(res)

def imposer_parametres_unresolved(df, raie_source, raie_cible):
    """ Duplique les paramètres de fit d'une raie vers l'autre (partageant le même pic). """
    cols = ['wavelength_obs', 'sigma', 'amplitude', 'continuum_m', 'continuum_c', 'Int_cgs']
    if raie_source not in df['Line'].values or raie_cible not in df['Line'].values: return df
    
    val_dict = df[df['Line'] == raie_source][cols].iloc[0].to_dict()
    for col, val in val_dict.items():
        df.loc[df['Line'] == raie_cible, col] = val
        
    lth_cible = df.loc[df['Line'] == raie_cible, 'wavelength_th'].values[0]
    df.loc[df['Line'] == raie_cible, 'Velocity_km_s'] = C_KMS * (val_dict['wavelength_obs'] - lth_cible) / lth_cible
    return df

def mark_unresolved_pairs(df_merged, unresolved_pairs):
    """ Tague les raies comme UNRESOLVED dans le DataFrame. """
    df = df_merged.copy()
    df['blended_with'] = pd.NA
    for L1, L2 in unresolved_pairs:
        for ln, p in [(L1, L2), (L2, L1)]:
            mask = df['Line'] == ln
            if mask.sum() > 0:
                df.loc[mask, 'fit_type'] = 'UNRESOLVED'
                df.loc[mask, 'blended_with'] = p
    return df


# =============================================================================
# 4. FUSION ET FIT GLOBAL ROBUSTE
# =============================================================================

def merge_spectral_dataframes(df_simple, df_blend, df_double, df_unres):
    """ Concatène tous les DataFrames en respectant l'ordre de priorité. """
    dfs = []
    if df_double is not None: dfs.append(df_double[df_double['success'] == True].copy())
    if df_unres is not None: dfs.append(df_unres[df_unres['success'] == True].copy())
    if df_blend is not None: dfs.append(df_blend[df_blend['success'] == True].copy())
    if df_simple is not None: dfs.append(df_simple[df_simple['success'] == True].copy())
    
    if not dfs: return pd.DataFrame()
    
    df_tot = pd.concat(dfs, ignore_index=True)
    df_merged = df_tot.drop_duplicates(subset=['Line'], keep='first').reset_index(drop=True)
    
    for c in ['amp_blend', 'mu_blend', 'sig_blend', 'blended_with']:
        if c not in df_merged.columns: df_merged[c] = np.nan
            
    return df_merged

# =============================================================================
# FONCTIONS DE DIAGNOSTIC LOCAL RESTAURÉES (Aurélien Piluso - Mai 2026)
# =============================================================================

def analyse_velocity_detailed_local(df_local, n_sigma=2.5, relative_to_median=True):
    """
    Restaure l'analyse de vitesse détaillée sur les fits locaux avant le fit global.
    Affiche les vitesses relatives ou absolues avec identification des ailes et des outliers.
    """
    df_analyse = df_local.copy()
    if df_analyse.empty or 'Velocity_km_s' not in df_analyse.columns:
        print("⚠️ DataFrame vide ou colonne 'Velocity_km_s' absente.")
        return None, None

    v = df_analyse['Velocity_km_s'].dropna()
    median_v = np.median(v)
    mean_v = np.mean(v)
    std_v = np.std(v)

    print(f"--- Statistiques des Vitesses (Fit Local) ---")
    print(f"Nombre de pics mesurés : {len(v)}")
    print(f"Moyenne : {mean_v:.2f} km/s / Médiane : {median_v:.2f} km/s")
    print(rf"Écart-type ($\sigma$) : {std_v:.2f} km/s")

    if relative_to_median:
        df_analyse['Plot_V'] = df_analyse['Velocity_km_s'] - median_v
        center_val = 0.0
        mean_offset = mean_v - median_v
        y_label = r"Relative Velocity ($V - V_{median}$) [km/s]"
    else:
        df_analyse['Plot_V'] = df_analyse['Velocity_km_s']
        center_val = median_v
        mean_offset = mean_v
        y_label = r"Absolute Velocity [km/s]"

    seuil_bas = center_val - (n_sigma * std_v)
    seuil_haut = center_val + (n_sigma * std_v)

    mask_outlier = (df_analyse['Plot_V'] < seuil_bas) | (df_analyse['Plot_V'] > seuil_haut)
    df_outliers = df_analyse[mask_outlier].copy()
    df_valid = df_analyse[~mask_outlier].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [2, 1]})

    # Scatter Plot
    ax1.scatter(df_valid['wavelength_th'], df_valid['Plot_V'], c='dodgerblue', alpha=0.8, s=100, edgecolors='k', label='Valid Local Fits', zorder=3)
    ax1.scatter(df_outliers['wavelength_th'], df_outliers['Plot_V'], c='crimson', s=140, marker='D', edgecolors='white', label='Outliers Rejetés', zorder=4)

    ax1.axhline(center_val, color='black', linestyle='-', linewidth=2, label=f'Médiane ({median_v:.1f} km/s)')
    ax1.axhline(mean_offset, color='green', linestyle='--', alpha=0.6, label='Moyenne')
    ax1.axhline(seuil_haut, color='red', linestyle=':', lw=2, label=f'Limite +{n_sigma}sig')
    ax1.axhline(seuil_bas, color='red', linestyle=':', lw=2, label=f'Limite -{n_sigma}sig')
    ax1.fill_between([df_analyse['wavelength_th'].min(), df_analyse['wavelength_th'].max()], center_val - std_v, center_val + std_v, color='gray', alpha=0.2, label=r'Zone $\pm 1\sigma$')

    for idx, row in df_analyse.iterrows():
        color_txt = 'darkred' if mask_outlier[idx] else 'black'
        weight = 'bold' if mask_outlier[idx] else 'normal'
        ax1.text(row['wavelength_th'], row['Plot_V'] + (0.5 if row['Plot_V']>=0 else -0.8), str(row['Line']),
                 fontsize=8, color=color_txt, fontweight=weight, ha='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    ax1.set_xlabel(r"Theoretical Wavelength [$\mu$m]", fontsize=14)
    ax1.set_ylabel(y_label, fontsize=14)
    ax1.set_title("Analyse des Vitesses Radiales par Pic (Ajustement Local)", fontsize=16, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right')

    # Histogramme simple
    bins_v = np.arange(np.floor(df_analyse['Plot_V'].min())-2, np.ceil(df_analyse['Plot_V'].max())+2, 1.5)
    ax2.hist(df_valid['Plot_V'], bins=bins_v, color='dodgerblue', alpha=0.6, edgecolor='white', label='Valid')
    ax2.hist(df_outliers['Plot_V'], bins=bins_v, color='crimson', alpha=0.8, edgecolor='white', label='Outliers')
    ax2.set_xlabel(y_label, fontsize=14)
    ax2.set_ylabel("Count", fontsize=14)
    ax2.set_title("Distribution des vitesses locales", fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_h_and_co_distributions_v3(df_co, bin_width=5):
    """
    Génère l'histogramme comparatif de la dynamique cinématique entre les phases gazeuses :
    H I (Atomique), H2 (Moléculaire dense) et vos pics de CO mesurés.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Reconstruction des données historiques Orion Bar (Peeters / Van de Putte et al.)
    orig_bins = np.arange(-5, 40, 5)
    hi_counts = [1, 4, 16, 14, 6, 3, 1, 0]
    h2_counts = [0, 0, 0, 2, 8, 2, 0, 1]
    
    hi_data = np.repeat(orig_bins[:-1] + 2.5, hi_counts)
    h2_data = np.repeat(orig_bins[:-1] + 2.5, h2_counts)
    v_co = df_co['Velocity_km_s'].dropna().values

    # Tracé des distributions
    plot_bins = np.arange(-5, 40 + bin_width, bin_width)
    ax.hist(hi_data, bins=plot_bins, histtype='step', edgecolor='#0044cc', linewidth=2.5, alpha=0.8, label='H I (Atomic PDR)')
    ax.hist(h2_data, bins=plot_bins, histtype='step', edgecolor='#ff7f0e', linewidth=2.5, alpha=0.8, label='H$_2$ (Molecular)')
    ax.hist(v_co, bins=plot_bins, histtype='step', edgecolor='forestgreen', linewidth=3.0, label=f'CO (This Study, bin={bin_width} km/s)')

    # Calculs et lignes des médianes
    datasets = [('H I', hi_data, '#0044cc'), ('H$_2$', h2_data, '#ff7f0e'), ('CO', v_co, 'forestgreen')]
    stats_text_lines = []
    
    for name, data, color in datasets:
        if len(data) > 0:
            med_v = np.median(data)
            ax.axvline(med_v, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
            stats_text_lines.append(f"{name} (N={len(data)}):\n  Médiane = {med_v:.2f} km/s\n  sig = {np.std(data):.2f} km/s")

    ax.set_xlabel(r"Velocity $v_{LSR}$ [km s$^{-1}$]", fontsize=16)
    ax.set_ylabel("Number of lines", fontsize=16)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)
    ax.legend(loc='upper right', fontsize=12, framealpha=1)
    
    stats_text = "\n\n".join(stats_text_lines)
    ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, va='top', ha='left', 
            fontsize=11, family='monospace', bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="gray", alpha=0.9))
            
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(right=0.75) 
    return fig, ax


def compare_filter_effects(df_before, df_after):
    """
    Point de contrôle critique pour comprendre exactement quelles raies ont été perdues
    lors du filtrage qualité et pourquoi.
    """
    print("\n========== Diagnostic de Filtrage ==========")
    lines_before = set(df_before['Line'].unique())
    lines_after = set(df_after['Line'].unique())
    eliminated = lines_before - lines_after
    
    print(f"Raies initialement disponibles (Fit Global) : {len(lines_before)}")
    print(f"Raies conservées après filtre : {len(lines_after)}")
    print(f"Raies éliminées : {len(eliminated)}")
    print("---------------------------------------------")
    
    if len(eliminated) == 0:
        print("✅ Aucune raie n'a été éliminée par les critères.")
        return
        
    records = []
    for line in eliminated:
        row = df_before[df_before['Line'] == line].iloc[0]
        records.append({
            'Line': line,
            'Velocity': row.get('Velocity_km_s', np.nan),
            'SNR_noise': row.get('SNR_noise', np.nan),
            'fit_type': row.get('fit_type', 'UNKNOWN')
        })
    df_elim = pd.DataFrame(records)
    print(df_elim.to_string(index=False))
    print("=============================================")

def fit_global_profile_robust(df_spectrum, df_merged, global_model, window_sigma=6.0):
    """
    Fit rigoureux des Gaussiennes sur le continuum global via `least_squares` (loss='soft_l1').
    Cette méthode minimise rigoureusement (I_obs - C_global) - Modèle_Gaussien.
    """
    df_res = df_merged.copy()
    a_glob, s_glob, i_glob = [], [], []
    C_MICRON_S = 2.99792458e14

    for idx, row in df_res.iterrows():
        mu_loc = row['wavelength_obs']
        sig_loc = row['sigma']
        amp_loc = row['amplitude']
        
        is_blend = row.get('fit_type') in ['BLEND', 'DOUBLE'] and pd.notna(row.get('amp_blend'))
        
        # --- 1. DÉFINITION DE LA FENÊTRE DE FIT ---
        if is_blend:
            mu2 = row['mu_blend']
            sig2 = row['sig_blend']
            w_min = min(mu_loc - window_sigma*sig_loc, mu2 - window_sigma*sig2)
            w_max = max(mu_loc + window_sigma*sig_loc, mu2 + window_sigma*sig2)
        else:
            w_min, w_max = mu_loc - window_sigma*sig_loc, mu_loc + window_sigma*sig_loc

        mask = (df_spectrum['Wavelength'] >= w_min) & (df_spectrum['Wavelength'] <= w_max)
        x_d = df_spectrum.loc[mask, 'Wavelength'].values
        y_d = df_spectrum.loc[mask, 'Int'].values
        
        if len(x_d) < 5:
            # Sécurité si manque de données
            a_glob.append(amp_loc); s_glob.append(sig_loc); i_glob.append(row.get('Int_cgs', 0))
            continue
            
        # --- 2. ISOLATION DU SIGNAL (Soustraction du Continuum Global) ---
        y_target = y_d - global_model(x_d)
        
        # Point de départ intelligent : L'amplitude attendue est l'ancienne + l'écart des continus
        c_loc_mu = row['continuum_m'] * mu_loc + row['continuum_c']
        c_glob_mu = global_model(np.array([mu_loc]))[0] if isinstance(global_model(np.array([mu_loc])), np.ndarray) else global_model(mu_loc)
        A_guess = max(amp_loc + (c_loc_mu - c_glob_mu), amp_loc * 0.1)

        # --- 3. OPTIMISATION NON-LINÉAIRE ROBUSTE ---
        if is_blend:
            amp2_loc = row['amp_blend']
            c_loc_mu2 = row['continuum_m'] * mu2 + row['continuum_c']
            c_glob_mu2 = global_model(np.array([mu2]))[0] if isinstance(global_model(np.array([mu2])), np.ndarray) else global_model(mu2)
            A2_guess = max(amp2_loc + (c_loc_mu2 - c_glob_mu2), amp2_loc * 0.1)

            p0 = [A_guess, mu_loc, sig_loc, A2_guess, mu2, sig2]

            def residuals_blend(p, x, y):
                A1, m1, s1, A2, m2, s2 = p
                return y - (A1 * np.exp(-(x - m1)**2 / (2 * s1**2)) + A2 * np.exp(-(x - m2)**2 / (2 * s2**2)))

            # Les positions et largeurs sont très contraintes par rapport au fit local, l'amplitude est libre
            bounds = (
                [0, mu_loc - sig_loc*0.5, sig_loc*0.8, 0, mu2 - sig2*0.5, sig2*0.8],
                [A_guess*3.0, mu_loc + sig_loc*0.5, sig_loc*1.2, A2_guess*3.0, mu2 + sig2*0.5, sig2*1.2]
            )

            try:
                res = least_squares(residuals_blend, p0, args=(x_d, y_target), bounds=bounds, loss='soft_l1')
                A_opt, _, sig_opt, A2_opt, _, sig2_opt = res.x
                
                a_glob.append(A_opt); s_glob.append(sig_opt)
                i_glob.append((A_opt * sig_opt * np.sqrt(2 * np.pi)) * ((C_MICRON_S / (mu_loc**2)) * 1e-17))
                df_res.at[idx, 'amp_blend_global'] = A2_opt
                df_res.at[idx, 'sig_blend_global'] = sig2_opt
            except:
                # Fallback si explosion de l'algorithme
                a_glob.append(A_guess); s_glob.append(sig_loc); i_glob.append(row.get('Int_cgs', 0))
                df_res.at[idx, 'amp_blend_global'] = A2_guess
                df_res.at[idx, 'sig_blend_global'] = sig2

        else:
            p0 = [A_guess, mu_loc, sig_loc]

            def residuals_single(p, x, y):
                A, mu, sigma = p
                return y - (A * np.exp(-(x - mu)**2 / (2 * sigma**2)))

            bounds = (
                [0, mu_loc - sig_loc*0.5, sig_loc*0.8],
                [A_guess*3.0, mu_loc + sig_loc*0.5, sig_loc*1.2]
            )

            try:
                res = least_squares(residuals_single, p0, args=(x_d, y_target), bounds=bounds, loss='soft_l1')
                A_opt, _, sig_opt = res.x
                
                a_glob.append(A_opt); s_glob.append(sig_opt)
                i_glob.append((A_opt * sig_opt * np.sqrt(2 * np.pi)) * ((C_MICRON_S / (mu_loc**2)) * 1e-17))
            except:
                a_glob.append(A_guess); s_glob.append(sig_loc); i_glob.append(row.get('Int_cgs', 0))

    df_res['amp_global'] = a_glob
    df_res['sig_global'] = s_glob
    df_res['Int_cgs_global'] = i_glob
    
    df_res['Int_err_final'] = np.abs(df_res['Int_cgs'] - df_res['Int_cgs_global']) / 2.0
    return df_res







def compute_final_intensities(df, noise_rms):
    """ Calcule I_moy, I_min, I_max et propage l'erreur totale. """
    df_final = df.copy()
    
    if 'Int_cgs_global' in df_final.columns:
        df_final['I_max'] = df_final[['Int_cgs', 'Int_cgs_global']].max(axis=1)
        df_final['I_min'] = df_final[['Int_cgs', 'Int_cgs_global']].min(axis=1)
        
        df_final['Int_cgs_final'] = (df_final['Int_cgs'] + df_final['Int_cgs_global']) / 2.0
        err_sys = (df_final['I_max'] - df_final['I_min']) / 2.0
    else:
        df_final['Int_cgs_final'] = df_final['Int_cgs']
        df_final['I_max'] = df_final['Int_cgs']
        df_final['I_min'] = df_final['Int_cgs']
        err_sys = 0.0
        
    err_stat = df_final['Int_cgs_final'] / df_final['SNR_noise']
    df_final['Int_err_final'] = np.sqrt(err_stat**2 + err_sys**2)
    
    return df_final


# =============================================================================
# 5. VISUALISATIONS HAUTE QUALITÉ
# =============================================================================

def plot_smart_gallery(df_spectrum, df_merged, n_cols=6, window_view=0.005, style_plot="step", clean_mode=False):
    """ Galerie triée, compacte et intelligente pour chaque raie fittée. """
    if df_merged.empty: return
    
    items, seen = [], set()
    for _, row in df_merged.iterrows():
        ft = row['fit_type']
        if ft in ['DOUBLE', 'UNRESOLVED']:
            line1, line2 = row['Line'], row.get('blended_with')
            if pd.isna(line2): items.append(('SINGLE', row)); continue
            key = frozenset([line1, line2])
            if key not in seen:
                try:
                    r2 = df_merged[(df_merged['Line'] == line2) & (df_merged['fit_type'] == ft)].iloc[0]
                    items.append((ft, row, r2))
                    seen.add(key)
                except: items.append(('SINGLE', row))
        else: items.append(('SINGLE', row))

    items.sort(key=lambda x: x[1]['wavelength_th'] if x[0]=='SINGLE' else min(x[1]['wavelength_th'], x[2]['wavelength_th']))
    
    n_plots = len(items)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    plt.rc('xtick', labelsize=7); plt.rc('ytick', labelsize=7)
    axes = np.array(axes).flatten() if n_plots > 1 else [axes]

    for i, item in enumerate(items):
        ax, itype = axes[i], item[0]
        
        if itype == 'SINGLE':
            r = item[1]; mu, sig = r['wavelength_obs'], r['sigma']
            w_min, w_max = mu - max(window_view/2, 4*sig), mu + max(window_view/2, 4*sig)
            tit = f"{r['Line']} ({r['fit_type']})"
        else:
            r1, r2 = item[1], item[2]
            w_min = min(r1['wavelength_obs'], r2['wavelength_obs']) - max(window_view/4, 3*max(r1['sigma'], r2['sigma']))
            w_max = max(r1['wavelength_obs'], r2['wavelength_obs']) + max(window_view/4, 3*max(r1['sigma'], r2['sigma']))
            tit = f"{r1['Line']} / {r2['Line']} ({itype})"

        mask = (df_spectrum["Wavelength"] >= w_min) & (df_spectrum["Wavelength"] <= w_max)
        x_obs, y_obs = df_spectrum.loc[mask, "Wavelength"].values, df_spectrum.loc[mask, "Int"].values
        if len(x_obs) == 0: continue
            
        ax.step(x_obs, y_obs, c='black', lw=1.5, alpha=0.8, where='mid') if style_plot == "step" else ax.plot(x_obs, y_obs, c='black', lw=1.5, alpha=0.8, marker='o', ms=2)
        x_m = x_obs if style_plot == "step" else np.linspace(w_min, w_max, 200)

        if itype == 'SINGLE':
            cont = r['continuum_m'] * x_m + r['continuum_c']
            g_main = r['amplitude'] * np.exp(-(x_m - mu)**2 / (2 * sig**2))
            g_sec = r['amp_blend'] * np.exp(-(x_m - r['mu_blend'])**2 / (2 * r['sig_blend']**2)) if r['fit_type'] == 'BLEND' else np.zeros_like(x_m)
            tot = cont + g_main + g_sec
            c_main, c_sec, c_tot = ('tab:orange', 'darkorange', 'tab:orange') if r['fit_type']=='BLEND' else ('tab:green', None, 'tab:green')
        else:
            cont = r1['continuum_m'] * x_m + r1['continuum_c']
            g_main = r1['amplitude'] * np.exp(-(x_m - r1['wavelength_obs'])**2 / (2 * r1['sigma']**2))
            g_sec = r2['amplitude'] * np.exp(-(x_m - r2['wavelength_obs'])**2 / (2 * r2['sigma']**2)) if itype == 'DOUBLE' else np.zeros_like(x_m)
            tot = cont + g_main + g_sec
            c_main, c_sec, c_tot = ('blue', 'teal', 'blue') if itype == 'DOUBLE' else ('red', 'red', 'red')

        p_k = {'where': 'mid'} if style_plot == "step" else {}
        plot_f = ax.step if style_plot == "step" else ax.plot
        
        plot_f(x_m, cont, ls=':', color='gray', alpha=0.5, **p_k)
        plot_f(x_m, cont + g_main, ls='--', color=c_main, lw=2, alpha=0.8, **p_k)
        if np.any(g_sec > 0): plot_f(x_m, cont + g_sec, ls='--', color=c_sec, lw=2, alpha=0.8, **p_k)
        plot_f(x_m, tot, ls='-', color=c_tot, lw=2.5, alpha=0.5, **p_k)

        ax.set_title(tit, fontsize=8, color=c_tot, fontweight='bold')
        ax.set_xlim(w_min, w_max)
        ax.tick_params(axis='x', rotation=45, labelsize=7)

    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
"""
def plot_final_summary(df_spectrum, df_merged, global_model, wl_range=(4.55, 4.93), ylim=(100, 240), 
                       n_plots=3, type_plot="step", plot_unresolved=True, df_unresolved=None):

    wl_min, wl_max = wl_range
    delta_wl = (wl_max - wl_min) / n_plots
    fig, axes = plt.subplots(n_plots, 1, figsize=(25, 8 * n_plots))
    if n_plots == 1: axes = [axes]
    
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)

    for i, ax in enumerate(axes):
        ax.set_ylim(ylim)
        c_min, c_max = wl_min + i * delta_wl, wl_min + (i + 1) * delta_wl
        
        # 1. Tracé du spectre observé
        mask = (df_spectrum["Wavelength"] >= c_min) & (df_spectrum["Wavelength"] <= c_max)
        if mask.sum() == 0: continue
        w_v, f_v = df_spectrum.loc[mask, "Wavelength"].values, df_spectrum.loc[mask, "Int"].values
        
        if type_plot == "step":
            ax.step(w_v, f_v, color="black", lw=1.5, alpha=0.9, where='mid', zorder=1)
        else:
            ax.plot(w_v, f_v, color="black", lw=1.5, alpha=0.9, zorder=1)

        # 2. Continuum Global (Rouge plein)
        x_g = np.linspace(c_min, c_max, 1000)
        ax.plot(x_g, global_model(x_g), color="red", lw=2, alpha=0.6, zorder=2)

        # 3. Récupération des raies fittées dans la fenêtre
        subs = df_merged[(df_merged['wavelength_obs'] >= c_min) & (df_merged['wavelength_obs'] <= c_max)].sort_values('wavelength_obs')
        
        last_x, tog = -999, 0
        y_range = ylim[1] - ylim[0]
        # Trois niveaux de hauteur pour empêcher les textes de se chevaucher
        y_lvls = [ylim[0] + 0.85 * y_range, ylim[0] + 0.65 * y_range, ylim[0] + 0.45 * y_range]

        for _, r in subs.iterrows():
            mu, sig, A = r['wavelength_obs'], r['sigma'], r['amplitude']
            m, c = r['continuum_m'], r['continuum_c']
            A_g, s_g = r.get('amp_global', A), r.get('sig_global', sig)
            
            # Vérification de la présence d'un blend dans la même fenêtre d'ajustement
            has_blend = 'amp_blend' in r and pd.notna(r['amp_blend'])
            if has_blend:
                A_b, mu_b, sig_b = r['amp_blend'], r['mu_blend'], r['sig_blend']
            else:
                A_b = mu_b = sig_b = 0

            x_m = np.linspace(mu - 8*sig, mu + 8*sig, 300)
            
            # --- MODÈLE LOCAL ---
            c_loc = m * x_m + c
            g_main_loc = A * np.exp(-(x_m - mu)**2 / (2 * sig**2))
            g_sec_loc = A_b * np.exp(-(x_m - mu_b)**2 / (2 * sig_b**2)) if has_blend else np.zeros_like(x_m)
            
            y_loc_tot = c_loc + g_main_loc + g_sec_loc # Somme des gaussiennes
            
            # Continuum local (Pointillés bleus)
            ax.plot(x_m, c_loc, color='blue', ls=':', lw=1.5, alpha=0.8, zorder=3)
            # Fit local total (Tirets bleus)
            ax.plot(x_m, y_loc_tot, color='blue', ls='--', lw=1.5, alpha=0.8, zorder=4)
            
            # --- MODÈLE GLOBAL ---
            c_glob = global_model(x_m)
            g_main_glob = A_g * np.exp(-(x_m - mu)**2 / (2 * s_g**2))
            
            # Reconstitution de la 2nde gaussienne globale (translatée sur le continuum global)
            if has_blend:
                c_glob_mu_b = global_model(np.array([mu_b]))[0] if isinstance(global_model(np.array([mu_b])), np.ndarray) else global_model(mu_b)
                c_loc_mu_b = m * mu_b + c
                A_b_g = max(A_b + (c_loc_mu_b - c_glob_mu_b), 0)
                g_sec_glob = A_b_g * np.exp(-(x_m - mu_b)**2 / (2 * sig_b**2))
            else:
                g_sec_glob = np.zeros_like(x_m)

            y_glob_tot = c_glob + g_main_glob + g_sec_glob # Somme des gaussiennes globales
            
            # Remplissage et ligne pleine pour le global
            ax.fill_between(x_m, c_glob, y_glob_tot, color='mediumseagreen', alpha=0.25, zorder=4)
            ax.plot(x_m, y_glob_tot, color='seagreen', ls='-', lw=2, zorder=5)

            # --- ANNOTATIONS ---
            # Alternance des hauteurs de texte
            if abs(mu - last_x) < (c_max - c_min) * 0.05:
                tog = (tog + 1) % 3
            else:
                tog = 0
            last_x = mu
            
            Err = r.get('Int_err_final', 0)
            SNR = r.get('SNR_noise', 0)
            
            # Formatage du texte "Droit"
            lbl = (f"{r['Line']}\n"
                   f"I_loc (Bleu) : {r.get('Int_cgs', 0):.2e}\n"
                   f"I_glob (Vert): {r.get('Int_cgs_global', 0):.2e}\n"
                   f"deltaI: {Err:.2e} / SNR: {SNR:.1f}")
            
            peak_y = c_loc[len(c_loc)//2] + A
            
            ax.annotate(lbl, xy=(mu, peak_y), xytext=(mu, y_lvls[tog]), 
                        arrowprops=dict(arrowstyle='-', color='black', alpha=0.4),
                        fontsize=8, color='black', ha='center', va='bottom', rotation=0, 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.95, zorder=20))

        ax.set_xlim(c_min, c_max)
        ax.set_ylabel("Surface brightness [MJy/sr]", size=14)
        ax.tick_params(axis='both', which='major', direction='in', length=8, width=1.5)

        if i == 0:
            custom = [Line2D([0],[0], color='black', lw=1.5), 
                      Line2D([0],[0], color='red', lw=2, alpha=0.6),
                      Line2D([0],[0], color='blue', ls=':', lw=1.5),
                      Line2D([0],[0], color='blue', ls='--', lw=1.5), 
                      Patch(facecolor='mediumseagreen', edgecolor='seagreen', alpha=0.5)]
            ax.legend(custom, ['Spectre Observé', 'Continuum Global', 'Continuum Local', 'Fit Local (Somme)', 'Fit Global (Somme)'], 
                      loc='upper right', fontsize=12, framealpha=0.9)

    axes[-1].set_xlabel(r"Wavelength $\lambda$ [$\mu m$]", size=14)
    plt.tight_layout()
    return fig, axes
"""
def plot_velocity_histogram(df_final, relative_to_median=False, size=(10, 6)):
    """ Histogramme simple des vitesses validées. """
    v = df_final['Velocity_km_s'].dropna()
    med, std = np.median(v), np.std(v)
    
    fig, ax = plt.subplots(figsize=size)
    ax.hist(v, bins=np.arange(np.floor(v.min()), np.ceil(v.max()) + 2, 2), color='dodgerblue', edgecolor='black', alpha=0.7)
    ax.axvline(med, color='red', ls='--', lw=2, label=f"Médiane : {med:.1f} km/s")
    ax.set_xlabel("Vitesse [km/s]", fontsize=14)
    ax.set_ylabel("Nombre de raies", fontsize=14)
    ax.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return fig, ax


# =============================================================================
# 6. EXPORTATION
# =============================================================================

def exporter_vers_dat(df, nom_fichier_sortie, col_int="Int_cgs_final", col_err="Int_err_final", info=""):
    """ Export standardisé au format DAT pour lecture rapide. """
    df_tri = df.sort_values(by='wavelength_th', ascending=True)
    with open(nom_fichier_sortie, 'w') as f:
        f.write("# Observations JWST - Orion Bar DF3\n")
        f.write(f"# Info: {info}\n")
        f.write(f"# {'Line':<13} {'I_corr':<12} {'Err_corr':<12}\n")
        for _, row in df_tri.iterrows():
            nom = str(row['Line'])
            i_val, e_val = row.get(col_int, 0.0), row.get(col_err, 0.0)
            f.write(f"{nom + ',':<14}{i_val:.2e},   {e_val:.2e}\n")
    print(f"Export réussi : {nom_fichier_sortie}")
    
    

    

def fit_global_profile_robust_v2(df_spectrum, df_merged, global_model, window_sigma=5.0):
    """
    Ajustement robuste d'une gaussienne pure sur le continuum global.
    Utilise least_squares (loss='soft_l1') avec f_scale dynamique (invariance d'échelle).
    """
    df_res = df_merged.copy()
    a_glob, s_glob, i_glob = [], [], []
    
    for idx, row in df_res.iterrows():
        mu_loc = row['wavelength_obs']
        sig_loc = row['sigma']
        amp_loc = row['amplitude']
        
        w_min, w_max = mu_loc - window_sigma*sig_loc, mu_loc + window_sigma*sig_loc
        mask = (df_spectrum['Wavelength'] >= w_min) & (df_spectrum['Wavelength'] <= w_max)
        x_d = df_spectrum.loc[mask, 'Wavelength'].values
        y_d = df_spectrum.loc[mask, 'Int'].values
        
        if len(x_d) < 5:
            a_glob.append(amp_loc); s_glob.append(sig_loc); i_glob.append(row['Int_cgs'])
            continue
            
        y_target = y_d - global_model(x_d)
        
        c_loc_mu = row['continuum_m'] * mu_loc + row['continuum_c']
        c_glob_mu = global_model(np.array([mu_loc]))[0] if isinstance(global_model(np.array([mu_loc])), np.ndarray) else global_model(mu_loc)
        
        # Le pic local par rapport au global donne le guess de la nouvelle amplitude
        A_guess = max(amp_loc + (c_loc_mu - c_glob_mu), amp_loc)
        p0 = [A_guess, mu_loc, sig_loc]
        
        def residuals(p, x, y):
            A, mu, sigma = p
            return y - (A * np.exp(-(x - mu)**2 / (2 * sigma**2)))
            
        # Bornes modérées : on laisse le fit trouver sa place, mais on contraint sigma
        bounds = (
            [amp_loc * 0.5, mu_loc - 1.0*sig_loc, sig_loc * 0.7],
            [A_guess * 2.5, mu_loc + 1.0*sig_loc, sig_loc * 1.5]
        )
        
        try:
            # f_scale dynamique : crucial pour les raies faibles sur piédestal
            scale_factor = max(0.2 * A_guess, 1e-3)
            res = least_squares(residuals, p0, args=(x_d, y_target), bounds=bounds, loss='soft_l1', f_scale=scale_factor)
            A_opt, mu_opt, sig_opt = res.x
            
            a_glob.append(A_opt)
            s_glob.append(sig_opt)
            
            flux = A_opt * sig_opt * np.sqrt(2 * np.pi)
            i_glob.append(flux * ((C_MICRON_S / (mu_opt**2)) * 1e-17))
            
        except Exception as e:
            print(f"Échec du fit global pour {row['Line']}: {e}")
            a_glob.append(amp_loc); s_glob.append(sig_loc); i_glob.append(row['Int_cgs'])

    df_res['amp_global'] = a_glob
    df_res['sig_global'] = s_glob
    df_res['Int_cgs_global'] = i_glob
    
    return df_res


def fit_global_profile_robust_v3(df_spectrum, df_merged, global_model, window_sigma=5.0):
    """
    Fit robuste V4 : Verrouille fermement les largeurs (sigma) et les centres (mu) 
    aux valeurs du fit local pour forcer le maintien du 'creux' entre les raies.
    L'algorithme n'a que l'amplitude (A) pour s'ajuster au continuum global.
    """
    df_res = df_merged.copy()
    a_glob, s_glob, i_glob = [], [], []
    C_MICRON_S = 2.99792458e14
    
    for idx, row in df_res.iterrows():
        mu_loc = row['wavelength_obs']
        sig_loc = row['sigma']
        amp_loc = row['amplitude']
        
        is_blend = row.get('fit_type') in ['BLEND', 'DOUBLE'] and pd.notna(row.get('amp_blend')) and row.get('amp_blend') > 0
        
        # Fenêtre d'ajustement
        if is_blend:
            mu2_loc = row['mu_blend']
            sig2_loc = row['sig_blend']
            amp2_loc = row['amp_blend']
            w_min = min(mu_loc - window_sigma*sig_loc, mu2_loc - window_sigma*sig2_loc)
            w_max = max(mu_loc + window_sigma*sig_loc, mu2_loc + window_sigma*sig2_loc)
        else:
            w_min, w_max = mu_loc - window_sigma*sig_loc, mu_loc + window_sigma*sig_loc
            
        mask = (df_spectrum['Wavelength'] >= w_min) & (df_spectrum['Wavelength'] <= w_max)
        x_d = df_spectrum.loc[mask, 'Wavelength'].values
        y_d = df_spectrum.loc[mask, 'Int'].values
        
        if len(x_d) < 5:
            a_glob.append(amp_loc); s_glob.append(sig_loc); i_glob.append(row.get('Int_cgs', 0))
            continue
            
        y_target = y_d - global_model(x_d)
        
        c_loc_mu = row['continuum_m'] * mu_loc + row['continuum_c']
        c_glob_mu = global_model(np.array([mu_loc]))[0] if isinstance(global_model(np.array([mu_loc])), np.ndarray) else global_model(mu_loc)
        A_guess = max(amp_loc + (c_loc_mu - c_glob_mu), amp_loc)

        if is_blend:
            c_loc_mu2 = row['continuum_m'] * mu2_loc + row['continuum_c']
            c_glob_mu2 = global_model(np.array([mu2_loc]))[0] if isinstance(global_model(np.array([mu2_loc])), np.ndarray) else global_model(mu2_loc)
            A2_guess = max(amp2_loc + (c_loc_mu2 - c_glob_mu2), amp2_loc)

            p0 = [A_guess, mu_loc, sig_loc, A2_guess, mu2_loc, sig2_loc]

            def residuals_blend(p, x, y):
                A1, m1, s1, A2, m2, s2 = p
                return y - (A1 * np.exp(-(x - m1)**2 / (2 * s1**2)) + A2 * np.exp(-(x - m2)**2 / (2 * s2**2)))
                
            # LIMITES TRÈS STRICTES : mu bloqué à ±5%, sigma bloqué à ±10%. Seul A est libre.
            bounds = (
                [amp_loc * 0.5, mu_loc - 0.05*sig_loc, sig_loc * 0.9, amp2_loc * 0.5, mu2_loc - 0.05*sig2_loc, sig2_loc * 0.9],
                [A_guess * 3.0, mu_loc + 0.05*sig_loc, sig_loc * 1.1, A2_guess * 3.0, mu2_loc + 0.05*sig2_loc, sig2_loc * 1.1]
            )
            
            try:
                res = least_squares(residuals_blend, p0, args=(x_d, y_target), bounds=bounds, loss='soft_l1')
                A_opt, mu_opt, sig_opt, A2_opt, mu2_opt, sig2_opt = res.x
                
                a_glob.append(A_opt)
                s_glob.append(sig_opt)
                flux = A_opt * sig_opt * np.sqrt(2 * np.pi)
                i_glob.append(flux * ((C_MICRON_S / (mu_opt**2)) * 1e-17))
                
                df_res.at[idx, 'amp_blend_global'] = A2_opt
                df_res.at[idx, 'sig_blend_global'] = sig2_opt
                
            except Exception as e:
                a_glob.append(amp_loc); s_glob.append(sig_loc); i_glob.append(row.get('Int_cgs', 0))
                df_res.at[idx, 'amp_blend_global'] = amp2_loc
                df_res.at[idx, 'sig_blend_global'] = sig2_loc

        else:
            p0 = [A_guess, mu_loc, sig_loc]
            
            def residuals_single(p, x, y):
                A, mu, sigma = p
                return y - (A * np.exp(-(x - mu)**2 / (2 * sigma**2)))
                
            bounds = (
                [amp_loc * 0.5, mu_loc - 0.05*sig_loc, sig_loc * 0.9],
                [A_guess * 3.0, mu_loc + 0.05*sig_loc, sig_loc * 1.1]
            )
            
            try:
                res = least_squares(residuals_single, p0, args=(x_d, y_target), bounds=bounds, loss='soft_l1')
                A_opt, mu_opt, sig_opt = res.x
                
                a_glob.append(A_opt)
                s_glob.append(sig_opt)
                flux = A_opt * sig_opt * np.sqrt(2 * np.pi)
                i_glob.append(flux * ((C_MICRON_S / (mu_opt**2)) * 1e-17))
                
            except Exception as e:
                a_glob.append(amp_loc); s_glob.append(sig_loc); i_glob.append(row.get('Int_cgs', 0))

    df_res['amp_global'] = a_glob
    df_res['sig_global'] = s_glob
    df_res['Int_cgs_global'] = i_glob
    
    df_res['Int_err_final'] = np.abs(df_res['Int_cgs'] - df_res['Int_cgs_global']) / 2.0
    return df_res







def fit_global_profile_robust_v4(df_spectrum, df_merged, global_model, window_sigma=5.0):
    """
    V4 Révisée - Ajustement élastique : Permet à l'amplitude ET à la largeur (sigma)
    de s'ajuster simultanément sur le continuum global pour éviter la surestimation des sommets.
    Fit robuste V4 : Verrouille fermement les largeurs (sigma) et les centres (mu) 
    aux valeurs du fit local pour forcer le maintien du 'creux' entre les raies.
    L'algorithme n'a que l'amplitude (A) pour s'ajuster au continuum global.
    """
    df_res = df_merged.copy()
    a_glob, s_glob, i_glob = [], [], []
    C_MICRON_S = 2.99792458e14
    
    for idx, row in df_res.iterrows():
        mu_loc = float(row['wavelength_obs'])
        sig_loc = float(row['sigma'])
        amp_loc = float(row['amplitude'])
        m_loc = float(row['continuum_m'])
        c_loc = float(row['continuum_c'])
        
        # 1. Fenêtre d'analyse
        is_blend = row.get('fit_type') in ['BLEND', 'DOUBLE'] and pd.notna(row.get('amp_blend')) and row.get('amp_blend') > 0
        if is_blend:
            # si blend on élargi la fenetre our considérer la double guasienne
            mu2_loc = float(row['mu_blend'])
            sig2_loc = float(row['sig_blend'])
            w_min = min(mu_loc - window_sigma*sig_loc, mu2_loc - window_sigma*sig2_loc)
            w_max = max(mu_loc + window_sigma*sig_loc, mu2_loc + window_sigma*sig2_loc)
        else:
            w_min = mu_loc - window_sigma * sig_loc
            w_max = mu_loc + window_sigma * sig_loc
            
        mask = df_spectrum['Wavelength'].between(w_min, w_max)
        x_d = df_spectrum.loc[mask, 'Wavelength'].values
        y_d = df_spectrum.loc[mask, 'Int'].values
        
        if len(x_d) < 5:
            a_glob.append(amp_loc); s_glob.append(sig_loc); i_glob.append(row.get('Int_cgs', 0))
            continue
            
        y_global_cont = global_model(x_d)
        
        # Estimation de l'amplitude sur le fond global en se basant sur l'amplitude au pic de la gaussienne locale 
        c_loc_mu = m_loc * mu_loc + c_loc
        c_glob_mu = global_model(np.array([mu_loc]))[0] if isinstance(global_model(np.array([mu_loc])), np.ndarray) else global_model(mu_loc)
        A_guess = max(amp_loc + (c_loc_mu - c_glob_mu), amp_loc * 0.5)



        # --- CLOSURE INTERNE ---
        if is_blend:
            amp2_loc = float(row['amp_blend'])
            c_loc_mu2 = m_loc * mu2_loc + c_loc
            c_glob_mu2 = global_model(np.array([mu2_loc]))[0] if isinstance(global_model(np.array([mu2_loc])), np.ndarray) else global_model(mu2_loc)
            A2_guess = max(amp2_loc + (c_loc_mu2 - c_glob_mu2), amp2_loc * 0.5)
            
            # Paramètres libres : [A1, sigma1, A2, sigma2] -> mu reste bloqué pour la cinématique
            p0 = [A_guess, sig_loc, A2_guess, sig2_loc]
            bounds = (
                [amp_loc * 0.4, sig_loc * 0.95, amp2_loc * 0.4, sig2_loc * 0.95],
                [A_guess * 2.0, sig_loc * 1.25, A2_guess * 2.0, sig2_loc * 1.25]
            )
            #amplitude peut varier de 40% de l'amplitude initale et 200% de Amp guess
            # Sigma largeur doit rester entre 95% et 125% de la baleur initiale 
            
            
            
            #calcul de la déviation (résiduals)  : écart entre les données du spectre et le modele mathématique 
            # Model = Contglob + A * exp(-x-mu^2/2sigma^2)
            # -> Résiuds = yspectre-modele
            #Soft L1 est plus douce pour les valeurs aberrantes (ailes mal estimées)
            def residuals_blend(p):
                A1, s1, A2, s2 = p
                model = y_global_cont + A1 * np.exp(-(x_d - mu_loc)**2 / (2 * s1**2)) + A2 * np.exp(-(x_d - mu2_loc)**2 / (2 * s2**2))
                return y_d - model
                
            try:
                res = least_squares(residuals_blend, p0, bounds=bounds, loss='soft_l1')
                A_opt, s_opt, A2_opt, s2_opt = res.x
                
                a_glob.append(A_opt)
                s_glob.append(s_opt)
                flux = A_opt * s_opt * np.sqrt(2 * np.pi)
                i_glob.append(flux * ((C_MICRON_S / (mu_loc**2)) * 1e-17))
                
                df_res.at[idx, 'amp_blend_global'] = A2_opt
                df_res.at[idx, 'sig_blend_global'] = s2_opt
            except:
                a_glob.append(amp_loc); s_glob.append(sig_loc); i_glob.append(row.get('Int_cgs', 0))
        else:
            # Paramètres libres : [A, sigma]
            p0 = [A_guess, sig_loc]
            bounds = (
                [amp_loc * 0.4, sig_loc * 0.95],
                [A_guess * 2.0, sig_loc * 1.25] # Autorise une expansion modérée de la largeur (+25% max)
            )
            
            def residuals_single(p):
                A, s = p
                model = y_global_cont + A * np.exp(-(x_d - mu_loc)**2 / (2 * s**2))
                return y_d - model
                
            try:
                res = least_squares(residuals_single, p0, bounds=bounds, loss='soft_l1')
                A_opt, s_opt = res.x
                
                a_glob.append(A_opt)
                s_glob.append(s_opt)
                flux = A_opt * s_opt * np.sqrt(2 * np.pi)
                i_glob.append(flux * ((C_MICRON_S / (mu_loc**2)) * 1e-17))
            except:
                a_glob.append(amp_loc); s_glob.append(sig_loc); i_glob.append(row.get('Int_cgs', 0))

    df_res['amp_global'] = a_glob
    df_res['sig_global'] = s_glob
    df_res['Int_cgs_global'] = i_glob
    df_res['Int_err_final'] = np.abs(df_res['Int_cgs'] - df_res['Int_cgs_global']) / 2.0
    return df_res












def fit_global_profile_robust_v5(df_spectrum, df_merged, global_model, window_sigma=5.0):
    """
    V5 Révisée : Ajustement d'une Gaussienne pure directement ancrée sur le continuum global.
    Utilise la minimisation robuste Soft-L1 par une closure (pas de problème d'arguments).
    Verrouille la cinématique locale (mu, sigma) et ajuste l'amplitude depuis le fond global.
    """
    df_res = df_merged.copy()
    a_glob, s_glob, i_glob = [], [], []
    C_MICRON_S = 2.99792458e14
    
    for idx, row in df_res.iterrows():
        mu_loc = float(row['wavelength_obs'])
        sig_loc = float(row['sigma'])
        amp_loc = float(row['amplitude'])
        m_loc = float(row['continuum_m'])
        c_loc = float(row['continuum_c'])
        
        # 1. Définition de la fenêtre d'ajustement autour de la raie
        is_blend = row.get('fit_type') in ['BLEND', 'DOUBLE'] and pd.notna(row.get('amp_blend')) and row.get('amp_blend') > 0
        if is_blend:
            mu2_loc = float(row['mu_blend'])
            sig2_loc = float(row['sig_blend'])
            w_min = min(mu_loc - window_sigma*sig_loc, mu2_loc - window_sigma*sig2_loc)
            w_max = max(mu_loc + window_sigma*sig_loc, mu2_loc + window_sigma*sig2_loc)
        else:
            w_min = mu_loc - window_sigma * sig_loc
            w_max = mu_loc + window_sigma * sig_loc
            
        mask = df_spectrum['Wavelength'].between(w_min, w_max)
        x_d = df_spectrum.loc[mask, 'Wavelength'].values
        y_d = df_spectrum.loc[mask, 'Int'].values
        
        if len(x_d) < 5:
            a_glob.append(amp_loc); s_glob.append(sig_loc); i_glob.append(row.get('Int_cgs', 0))
            continue
            
        # 2. Calcul du piédestal (la différence de niveau entre le continu local et global)
        # On évalue le modèle global sur les points X
        y_global_cont = global_model(x_d)
        
        # 3. Estimation de l'amplitude attendue sur le continuum global
        c_loc_mu = m_loc * mu_loc + c_loc
        c_glob_mu = global_model(np.array([mu_loc]))[0] if isinstance(global_model(np.array([mu_loc])), np.ndarray) else global_model(mu_loc)
        # L'amplitude par rapport au fond global est l'amplitude locale augmentée de la différence des continus
        A_guess = max(amp_loc + (c_loc_mu - c_glob_mu), amp_loc * 0.5)

        # --- CLOSURE DYNAMIQUE ---
        if is_blend:
            amp2_loc = float(row['amp_blend'])
            c_loc_mu2 = m_loc * mu2_loc + c_loc
            c_glob_mu2 = global_model(np.array([mu2_loc]))[0] if isinstance(global_model(np.array([mu2_loc])), np.ndarray) else global_model(mu2_loc)
            A2_guess = max(amp2_loc + (c_loc_mu2 - c_glob_mu2), amp2_loc * 0.5)
            
            p0 = [A_guess, A2_guess]
            bounds = ([0, 0], [A_guess * 3.0, A2_guess * 3.0])
            
            def residuals_blend(p):
                A1, A2 = p
                # Gaussiennes pures basées sur la cinématique locale, posées sur le continuum global
                model = y_global_cont + A1 * np.exp(-(x_d - mu_loc)**2 / (2 * sig_loc**2)) + A2 * np.exp(-(x_d - mu2_loc)**2 / (2 * sig2_loc**2))
                return y_d - model
                
            try:
                res = least_squares(residuals_blend, p0, bounds=bounds, loss='soft_l1')
                A_opt, A2_opt = res.x
                
                a_glob.append(A_opt)
                s_glob.append(sig_loc)
                flux = A_opt * sig_loc * np.sqrt(2 * np.pi)
                i_glob.append(flux * ((C_MICRON_S / (mu_loc**2)) * 1e-17))
                
                df_res.at[idx, 'amp_blend_global'] = A2_opt
                df_res.at[idx, 'sig_blend_global'] = sig2_loc
            except:
                a_glob.append(amp_loc); s_glob.append(sig_loc); i_glob.append(row.get('Int_cgs', 0))
                df_res.at[idx, 'amp_blend_global'] = amp2_loc
                df_res.at[idx, 'sig_blend_global'] = sig2_loc
        else:
            p0 = [A_guess]
            bounds = ([0], [A_guess * 3.0])
            
            def residuals_single(p):
                A = p[0]
                model = y_global_cont + A * np.exp(-(x_d - mu_loc)**2 / (2 * sig_loc**2))
                return y_d - model
                
            try:
                res = least_squares(residuals_single, p0, bounds=bounds, loss='soft_l1')
                A_opt = res.x[0]
                
                a_glob.append(A_opt)
                s_glob.append(sig_loc)
                flux = A_opt * sig_loc * np.sqrt(2 * np.pi)
                i_glob.append(flux * ((C_MICRON_S / (mu_loc**2)) * 1e-17))
            except:
                a_glob.append(amp_loc); s_glob.append(sig_loc); i_glob.append(row.get('Int_cgs', 0))

    df_res['amp_global'] = a_glob
    df_res['sig_global'] = s_glob
    df_res['Int_cgs_global'] = i_glob
    
    df_res['Int_err_final'] = np.abs(df_res['Int_cgs'] - df_res['Int_cgs_global']) / 2.0
    return df_res























def plot_kinematic_summary(df_spectrum, df_fits, global_model, 
                           wl_range=(4.55, 4.89), n_plots=2, ylim=(100, 250)):
    """
    Visualisation panoramique :
    - Affiche le spectre, le fit global (rouge) et le fit local.
    - Annotations : Ligne th (gris), Ligne obs (couleur raie).
    - Fond coloré entre th et obs (Rouge=Redshift, Bleu=Blueshift).
    - Texte : Nom de raie + Vitesse LSR (km/s).
    """
    wl_min, wl_max = wl_range
    delta_wl = (wl_max - wl_min) / n_plots
    fig, axes = plt.subplots(n_plots, 1, figsize=(23, 7 * n_plots))
    if n_plots == 1: axes = [axes]

    for i, ax in enumerate(axes):
        c_min = wl_min + i * delta_wl
        c_max = wl_min + (i + 1) * delta_wl
        ax.set_ylim(ylim)
        
        # Spectres et modèle global
        mask = (df_spectrum["Wavelength"] >= c_min) & (df_spectrum["Wavelength"] <= c_max)
        ax.plot(df_spectrum.loc[mask, "Wavelength"], df_spectrum.loc[mask, "Int"], c="black", lw=1.5, alpha=0.8, zorder=1)
        x_g = np.linspace(c_min, c_max, 500)
        ax.plot(x_g, global_model(x_g), color="red", lw=2, alpha=0.4, zorder=2)

        # Raies dans la zone
        subset = df_fits[(df_fits['wavelength_obs'] >= c_min) & (df_fits['wavelength_obs'] <= c_max)]
        y_levels = [ylim[0] + 0.85 * (ylim[1]-ylim[0]), ylim[0] + 0.65 * (ylim[1]-ylim[0])]

        for idx, row in subset.iterrows():
            mu, lth, v = row['wavelength_obs'], row['wavelength_th'], row['Velocity_km_s']
            sigma, amp, m, c = row['sigma'], row['amplitude'], row['continuum_m'], row['continuum_c']

            # Modèle gaussien
            x_plot = np.linspace(mu - 6*sigma, mu + 6*sigma, 100)
            y_fit = (amp * np.exp(-(x_plot - mu)**2 / (2 * sigma**2))) + (m * x_plot + c)
            
            # Couleur du shift (Redshift > 0, Blueshift < 0)
            shift_color = 'salmon' if (mu > lth) else 'cornflowerblue'
            
            ax.axvspan(lth, mu, color=shift_color, alpha=0.2)
            ax.axvline(lth, color='gray', ls='--', lw=1)
            ax.axvline(mu, color='black', ls='-', lw=1)
            ax.plot(x_plot, y_fit, color='darkgreen', lw=2, alpha=0.7)

            # Annotation
            label_txt = f"{row['Line']}\nv={v:.1f} km/s"
            ax.annotate(label_txt, xy=(mu, y_fit.max()), xytext=(mu, y_levels[idx % 2]),
                        arrowprops=dict(arrowstyle='-', color='black', alpha=0.3),
                        fontsize=8, ha='center', va='bottom', rotation=45,
                        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))

        ax.set_xlim(c_min, c_max)
        ax.grid(True, alpha=0.2)
        ax.set_ylabel("Brightness [MJy/sr]")
    
    axes[-1].set_xlabel("Wavelength [$\mu$m]")
    plt.tight_layout()
    return fig, axes

# def save_plot(name, out_dir, save_figs=False):
#     """Sauvegarde la figure actuelle si save_figs est True."""
#     if save_figs and out_dir and name:
#         path = os.path.join(out_dir, name)
#         plt.gcf().savefig(path, dpi=300, bbox_inches='tight')
#         print(f"   📸 Figure sauvegardée : {name}")
#     plt.show() # Appel standard à la fin
    
def fit_and_plot_temperature(df, ax, color='red', label_prefix=''):
    """
    Calcule la température d'excitation et trace la droite de régression sur l'axe donné.
    Utilise la relation : ln(Nu/gu) = - (1/Tex) * Eu/k + Constante
    """
    import numpy as np
    
    # 1. Filtrer les données valides
    mask = (df['Nu_gu'] > 0) & (df['En(K)_u'].notna()) & (df['Nu_gu'].notna())
    df_clean = df[mask]
    
    if df_clean.empty:
        print(f"⚠️ Pas de données valides pour {label_prefix}")
        return None

    x = df_clean['En(K)_u'].values
    y = np.log(df_clean['Nu_gu'].values)
    
    # 2. Fit linéaire (régression simple)
    # Si tu as des erreurs de colonne ('Nu_gu_err_30pc'), on pourrait pondérer ici, 
    # mais une polyfit simple suffit pour la robustesse du fit de T.
    a, b = np.polyfit(x, y, 1)

    # 3. Calcul de la température d'excitation (Tex)
    # Pente 'a' = -1/Tex => Tex = -1/a
    Tex = -1.0 / a

    # 4. Création de la droite d'ajustement
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = np.exp(a * x_fit + b)

    # 5. Traçage
    ax.plot(x_fit, y_fit, color=color, linestyle='--', zorder=0, 
            label=f'{label_prefix} ($T_{{ex}} \\approx {Tex:.0f}$ K)')
    
    ax.legend(fontsize=10)

    print(f"✅ Température calculée pour {label_prefix} : {Tex:.1f} K")
    return Tex