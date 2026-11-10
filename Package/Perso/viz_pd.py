#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:27:56 2025

@author: aurelien
"""

import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np 
from math import floor
import pandas as pd 
from sklearn.linear_model import LinearRegression
import pyismtools as ism

from labellines import labelLine, labelLines
from pyismtools.classes import *

import re
import os
import imageio.v2 as imageio # Use imageio.v2
import shutil # For managing temporary directory

from matplotlib import ticker
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import plotly.express as px

import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"
import read
import modify as mod 

import glob

plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['text.usetex'] = True


"""
chemin_modules = "/Users/aurelien/Desktop/PhD/Codes/Package/Perso/"
sys.path.append(chemin_modules)
import read as read 

"""

def processObs_df (Obs) : 
    df = add_allInfo_df_mol(Obs)
    df = mod.add_freq_and_lambda(df)
    df["lambda_micron"] = df["lambda_m"]*1e6
    df_final = compute_final_pdr_columns(df)
    return df_final

def processMod_df (Obs) : 
    df = add_allInfo_df_mol(Obs,Observation=False)
    df = mod.add_freq_and_lambda(df)
    df["lambda_micron"] = df["lambda_m"]*1e6
    df_final = compute_final_pdr_columns(df,Observation=False)
    return df_final



def add_allInfo_df_mol (Obs,mol="co",version="HITRAN",Observation=True): 
    Line = Obs.get_observable_names()
    Int =  Obs.get_observable_vector()
    noms_colonnes =["Line", "Int"]
    df = pd.DataFrame(list(zip(Line, Int)), 
                  columns=noms_colonnes)
    if Observation == True :
        Err = Obs.get_error_vector()
        noms_colonnes += ["Err"]
        df = pd.DataFrame(list(zip(Line, Int, Err)), 
                      columns=noms_colonnes)

    import to_PDR
    df = df.join(df['Line'].apply(read.parse_line_properties)) #lit le format spectro et découpe en vu,Ju,vl,Jl
    df_level = to_PDR.Level_to_pd(mol)
    df_lines_co = to_PDR.Lines_to_pd(mol,version)
    
    df_final = df.merge( #Ajout des gu, gl, depuis level
        df_level.add_suffix('_u'), left_on=['vu', 'Ju'], right_on=['v_u', 'J_u']
        ).merge(df_level.add_suffix('_l'), left_on=['vl', 'Jl'], right_on=['v_l', 'J_l'])
    #Ajout des Aij par Lines 
    df_final = pd.merge(df_final,df_lines_co,on=["vu","Ju","vl","Jl"])
    
    return df_final



def compute_final_pdr_columns(df, use_wavelength=True, Observation=True):
    PI = 3.141592653589793
    KB = 1.380649e-16      # Boltzmann (erg/K)
    H_PLANCK = 6.62607015e-27 # Planck (erg s)
    C_LIGHT = 2.99792458e10   # Vitesse lumière (cm/s)

    # --- 1. Calcul de l'énergie (Indépendant de l'observation/modèle) ---
    if use_wavelength or 'E_K' not in df.columns:
        if 'lambda_micron' not in df.columns:
            raise ValueError("Colonne 'lambda_micron' manquante.")
        
        lambda_cm = df['lambda_micron'] * 1e-4
        energy_transition_erg = (H_PLANCK * C_LIGHT) / lambda_cm
        print("Calcul basé sur : Longueur d'onde (lambda_micron)")
        
    else:
        energy_transition_erg = df['E_K'] * KB
        print("Calcul basé sur : Écart d'énergie (E_K)")

    # --- 2. Calculs des valeurs principales (Nu, Nu/gu) ---
    # On le fait ici pour tout le monde (Obs et Modèles)
    df['Nu'] = (4 * PI * df['Int']) / (df['Aij_s-1'] * energy_transition_erg) 
    df['Nu_gu'] = df['Nu'] / df['g_u']

    # --- 3. Calcul des erreurs (Seulement si Observation) ---
    if Observation: 
        if "Err" in df.columns : 
            threshold = 1.0e-2      
            # Gestion erreur relative vs absolue
            sigma_I = np.where(df['Err'] > threshold, 
                               (df['Err'] - 1.0) * df['Int'], 
                               df['Err'])
            df['Int_sigma'] = sigma_I
            # Erreur obs
            df['Nu_err'] = (4 * PI * sigma_I) / (df['Aij_s-1'] * energy_transition_erg)
            # Erreur 30% Fixe (0.3 * Int)
            sigma_I_30pc = 0.30 * df['Int']
            df['Nu_err_30pc'] = (4 * PI * sigma_I_30pc) / (df['Aij_s-1'] * energy_transition_erg)
     
            # Propagation sur Nu/gu
            df['Nu_gu_err'] = df['Nu_err'] / df['g_u'] 
            df['Nu_gu_err_30pc'] = df['Nu_err_30pc'] / df['g_u']
        else : 
            print("No Err Column found, we generate one at 30% uncertainties")
            sigma_I_30pc = 0.30 * df['Int']
            df['Nu_err_30pc'] = (4 * PI * sigma_I_30pc) / (df['Aij_s-1'] * energy_transition_erg)
            df['Nu_gu_err_30pc'] = df['Nu_err_30pc'] / df['g_u']

    return df














def plot_excitation_diagram(df, 
                            title="Excitation Diagram", 
                            save_path=None,
                            show_labels=True,
                            show_factor2=True,
                            color_by_v=True,
                            show_top_levels=False,
                            Show_err = False,
                            Label="Observation",
                            Size=(16,8)): 
    
    fig, ax = plt.subplots(figsize=Size)
    palette = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    
    if color_by_v and 'vu' in df.columns:
        unique_vs = sorted(df['vu'].dropna().unique())
        color_map = {v: palette[int(v) % len(palette)] for v in unique_vs}
    else:
        color_map = None

    groups = df.groupby('vu') if (color_by_v and 'vu' in df.columns) else [('All', df)]
    energy_counter = {}
    
    for v_val, group in groups:
        c = color_map[v_val] if color_map else 'black'
        label_v = f"v={int(v_val)}" if v_val != 'All' else Label

        x = group['En(K)_u']
        y = group['Nu_gu']
        # 1. Facteur 2 (Fond - zorder=1)
        if show_factor2:
            yerr_low = 0.5 * y
            yerr_high = 1.0 * y
            ax.errorbar(x, y, yerr=[yerr_low, yerr_high], 
                        fmt='none', ecolor=c, alpha=0.2, 
                        capsize=0, elinewidth=2, zorder=1)    
        if 'Nu_gu_err' in group.columns:
            


            # 2. Erreur Observationnelle (Pointillés Gris - zorder=2)
            if Show_err == True : 
                yerr_obs = group['Nu_gu_err']
                ax.errorbar(x, y, yerr=yerr_obs, 
                            fmt='none', ecolor='dimgray', linestyle='-', alpha=0.7,
                            capsize=6, elinewidth=1.5, zorder=2)



        else:
            ax.scatter(x, y, color=c, label=label_v, s=50, zorder=10)
            # 3. Erreur 30% (Noir Solide - zorder=3)
        if 'Nu_gu_err_30pc' in group.columns:
            yerr_30 = group['Nu_gu_err_30pc']
            ax.errorbar(x, y, yerr=yerr_30, 
                        fmt='none', ecolor='black', 
                        capsize=3, elinewidth=2, zorder=3)

        # 4. Marqueurs (Devant - zorder=10)
        ax.scatter(x, y, s=60, color=c, edgecolors='black', zorder=10, label=Label)

        for i, row in group.iterrows():
            eu = row['En(K)_u']
            eu_key = round(eu, 1)
            offset_level = energy_counter.get(eu_key, 0)
            energy_counter[eu_key] = offset_level + 1
            
            if show_labels:
                ax.annotate(row['Line_x'], 
                            xy=(eu, row['Nu_gu']),
                            xytext=(10, 30),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=8, color=c, rotation=90)

            if show_top_levels:
                ax.vlines(x=eu, ymin=0.85, ymax=0.9, transform=ax.get_xaxis_transform(), 
                          color='black', lw=0.8, alpha=0.6)
                
                label_top = f"(v{int(row['vu'])},J{int(row['Ju'])})"
                y_pos_top = 0.92 + (offset_level * 0.06) 
                
                ax.text(eu, y_pos_top, label_top, 
                        transform=ax.get_xaxis_transform(),
                        rotation=90, ha='center', va='bottom', fontsize=7, color='black')

    ax.set_yscale('log') 
    ax.set_xlabel(r'$E_u / k_B$ (K)', fontsize=25)
    ax.set_ylabel(r'$N_u / g_u$ (cm$^{-2}$)', fontsize=25)
    ax.set_title(title, fontsize=20)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    
    # Astuce pour éviter les doublons dans la légende si v=All
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=15, loc='best')

    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        
    return ax


def plot_excitation_diagramV2(df, 
                            title="Excitation Diagram", 
                            save_path=None,
                            show_labels=True,
                            show_factor2=True,
                            color_by_v=True,
                            show_top_levels=False,
                            Show_err=False,
                            Label="Observation",
                            Size=(16,8),
                            detection_limit_snr3=None): # NOUVEAU PARAMÈTRE
    
    fig, ax = plt.subplots(figsize=Size)
    palette = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    
    if color_by_v and 'vu' in df.columns:
        unique_vs = sorted(df['vu'].dropna().unique())
        color_map = {v: palette[int(v) % len(palette)] for v in unique_vs}
    else:
        color_map = None

    groups = df.groupby('vu') if (color_by_v and 'vu' in df.columns) else [('All', df)]
    energy_counter = {}
    
    for v_val, group in groups:
        c = color_map[v_val] if color_map else 'black'
        label_v = f"v={int(v_val)}" if v_val != 'All' else Label

        x = group['En(K)_u']
        y = group['Nu_gu']
        
        # 1. Facteur 2 (Fond - zorder=1)
        if show_factor2:
            yerr_low = 0.5 * y
            yerr_high = 1.0 * y
            ax.errorbar(x, y, yerr=[yerr_low, yerr_high], 
                        fmt='none', ecolor=c, alpha=0.2, 
                        capsize=0, elinewidth=2, zorder=1)    
            
        if 'Nu_gu_err' in group.columns:
            # 2. Erreur Observationnelle (Pointillés Gris - zorder=2)
            if Show_err == True: 
                yerr_obs = group['Nu_gu_err']
                ax.errorbar(x, y, yerr=yerr_obs, 
                            fmt='none', ecolor='dimgray', linestyle='-', alpha=0.7,
                            capsize=6, elinewidth=1.5, zorder=2)
        else:
            ax.scatter(x, y, color=c, label=label_v, s=50, zorder=10)
            
        # 3. Erreur 30% (Noir Solide - zorder=3)
        if 'Nu_gu_err_30pc' in group.columns:
            yerr_30 = group['Nu_gu_err_30pc']
            ax.errorbar(x, y, yerr=yerr_30, 
                        fmt='none', ecolor='black', 
                        capsize=3, elinewidth=2, zorder=3)

        # 4. Marqueurs (Devant - zorder=10)
        ax.scatter(x, y, s=60, color=c, edgecolors='black', zorder=10, label=Label)

        for i, row in group.iterrows():
            eu = row['En(K)_u']
            eu_key = round(eu, 1)
            offset_level = energy_counter.get(eu_key, 0)
            energy_counter[eu_key] = offset_level + 1
            
            if show_labels:
                ax.annotate(row['Line_x'], 
                            xy=(eu, row['Nu_gu']),
                            xytext=(10, 30),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=8, color=c, rotation=90)

            if show_top_levels:
                ax.vlines(x=eu, ymin=0.85, ymax=0.9, transform=ax.get_xaxis_transform(), 
                          color='black', lw=0.8, alpha=0.6)
                
                label_top = f"(v{int(row['vu'])},J{int(row['Ju'])})"
                y_pos_top = 0.92 + (offset_level * 0.06) 
                
                ax.text(eu, y_pos_top, label_top, 
                        transform=ax.get_xaxis_transform(),
                        rotation=90, ha='center', va='bottom', fontsize=7, color='black')

    # --- NOUVEAUTÉ : Ajout de la limite de détection ---
    if detection_limit_snr3 is not None:
        # Ligne horizontale
        ax.axhline(y=detection_limit_snr3, color='red', linestyle='--', linewidth=1.5, zorder=4, label='Limite de détection')
        
        # Récupération de la limite basse actuelle pour ombrer jusqu'en bas
        ymin, _ = ax.get_ylim()
        
        # Zone grisée (rouge très clair)
        ax.axhspan(ymin, detection_limit_snr3, color='red', alpha=0.05, zorder=0)
        
        # Texte explicatif (positionné à 2% de l'axe X, et aligné juste en dessous de la ligne)
        ax.text(0.02, detection_limit_snr3, "Zone indétectable (SNR $\leq$ 3)", 
                transform=ax.get_yaxis_transform(), # Permet de mélanger fraction de l'axe X (0.02) et valeur réelle sur Y
                color='red', fontsize=12, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # On s'assure que le span ne fausse pas l'échelle par défaut
        ax.set_ylim(bottom=ymin)
    # ---------------------------------------------------

    ax.set_yscale('log') 
    ax.set_xlabel(r'$E_u / k_B$ (K)', fontsize=25)
    ax.set_ylabel(r'$N_u / g_u$ (cm$^{-2}$)', fontsize=25)
    ax.set_title(title, fontsize=20)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    
    # Astuce pour éviter les doublons dans la légende si v=All
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=15, loc='best')

    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200)
        
    return ax


def plot_models_on_axis(ax, 
                        models_list, 
                        models_names=None,
                        color_by_model=True,
                        ColorPalette = ['red', 'green', 'blue', 'purple', 'orange']):
    """
    Superpose des modèles sur un axe existant.
    
    Arguments:
    - ax : L'objet axes matplotlib retourné par plot_excitation_diagram
    - models_list : Une liste de DataFrames. Chaque DF doit avoir 'En(K)_u', 'Nu_gu' et 'vu'.
    - models_names : Liste de noms (str) pour la légende (ex: ['Model A', 'Model B'])
    - color_by_model : 
        True -> Chaque modèle a une couleur unique (tous les v de ce modèle ont la même couleur).
        False -> La couleur dépend de v (comme pour les obs), le style de ligne dépend du modèle.
    """
    
    # Palettes et styles
    palette_v = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    colors_models = ColorPalette #['red', 'green', 'blue', 'purple', 'orange'] # Couleurs distinctes pour modèles
    linestyles = ['-', '--', '-.', ':']
    
    for i, df_model in enumerate(models_list):
        # Nom du modèle
        name = models_names[i] if (models_names and i < len(models_names)) else f"Model {i+1}"
        
        # Style de ligne pour ce modèle
        ls = linestyles[i % len(linestyles)]
        
        # Groupement par v
        if 'vu' in df_model.columns:
            groups = df_model.groupby('vu')
        else:
            groups = [('All', df_model)]
            
        for v_val, group in groups:
            # Trier par énergie pour avoir de belles lignes
            group = group.sort_values('En(K)_u')
            
            # Détermination de la couleur
            if color_by_model:
                c = colors_models[i % len(colors_models)]
                # On ne met le label qu'une fois par modèle (sur le premier v rencontré)
                lbl = name if v_val == list(groups.groups.keys())[0] else None
            else:
                # Couleur dépend de v (cohérent avec obs)
                idx = int(v_val) % len(palette_v) if v_val != 'All' else 0
                c = palette_v[idx]
                lbl = f"{name} v={v_val}" if i == 0 else None # Légende complexe à gérer sinon
            
            # Tracé : Ligne + Croix pour les points calculés
            ax.plot(group['En(K)_u'], group['Nu_gu'], 
                    linestyle=ls, color=c, marker='+', markersize=6, 
                    alpha=0.7, label=lbl, zorder=5)
            
    ax.legend(fontsize=25, loc='best')
    plt.draw()
    