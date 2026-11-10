#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:39:52 2026

@author: aurelien
"""

import pandas as pd
import numpy as np
import pyismtools as ism
import matplotlib.pyplot as plt

# --- Constantes physiques (CGS) ---
H_CGS = 6.62607015e-27
C_CGS = 2.99792458e10


##########################LEXTURE DE DONNEES ################
def load_physics_database(filepath):
    """Charge la base de données HITRAN formatée en DataFrame propre."""
    # Assure-toi que les noms de colonnes correspondent à ceux de ton fichier généré
    cols = ['Line', 'vu', 'Ju', 'vl', 'Jl', 'gu', 'gl', 'Eu_K', 'El_K', 'Eul_K', 'Aul', 'Wave_um']
    df_db = pd.read_csv(filepath, comment='#', names=cols, sep=r',\s*', engine='python')
    df_db['Line'] = df_db['Line'].str.replace(',', '') # Nettoyage si besoin
    return df_db





def build_translation_dict(obs_lines, mol="co"):
    """Crée le dictionnaire de traduction entre la nomenclature Obs et Modèle."""
    trans_dict = {}
    for x in obs_lines:
        try:
            # Nettoyage de base (au cas où il y ait des virgules)
            clean_x = x.replace(',', '').strip()
            vup, temp = clean_x.split('-')
            vlow, temp2 = temp.split('_')
            trtype, temp3 = temp2.split('(')
            
            # strip() est plus sûr que split(' ')[1] s'il y a un nombre variable d'espaces
            Jlow = int(temp3.split(')')[0].strip())
            
            if trtype == 'S': Jup = Jlow + 2
            elif trtype == 'R': Jup = Jlow + 1
            elif trtype == 'Q': Jup = Jlow
            elif trtype == 'P': Jup = Jlow - 1
            elif trtype == 'O': Jup = Jlow - 2
            else: continue
            
            # Nom attendu dans la grille PDR/PyISM
            grid_name = f"inta_00_{mol}_v{vup}_j{Jup}__v{vlow}_j{Jlow}"
            trans_dict[x] = grid_name
            
        except Exception as e:
            print(f"⚠️ Impossible de traduire la raie {x}: {e}")
            
    return trans_dict


###########################PROCESSING DATA EN APPELLANT PYISMTOOLS##########
def pyism_to_dataframe(ism_obj, df_db, source_name="Observation", is_model=False, scaling=1.0, params=None):
    """Convertit un objet PyISM (Obs ou Modèle) en DataFrame enrichi avec la physique."""
    lines = ism_obj.get_observable_names()
    # Application du scaling directement sur les intensités
    intensities = np.array(ism_obj.get_observable_vector()) * scaling
    
    # Gestion des erreurs (les modèles n'en ont pas)
    errors = [0.0] * len(lines) if is_model else ism_obj.get_error_vector()

    # Création du DF initial
    df = pd.DataFrame({
        'Line': lines,
        'Intensity': intensities,
        'Error': errors,
        'Source': source_name,
        'Type': 'Model' if is_model else 'Obs'
    })

    # Fusion avec la base de données physique (inner join pour ne garder que ce qu'on connait)
    df = pd.merge(df, df_db, on='Line', how='inner')

    # Calcul de Nu et Nu/gu (Vectorisé, ultra rapide)
    # Formule : Nu = (4 * pi * lambda_cm * I) / (h * c * Aul)
    conversion_factor = (4 * np.pi * (df['Wave_um'] * 1e-4)) / (H_CGS * C_CGS * df['Aul'])
    
    df['Nu'] = conversion_factor * df['Intensity']
    df['Nu_err'] = conversion_factor * df['Error']
    df['Nu_gu'] = df['Nu'] / df['gu']
    df['Nu_gu_err'] = df['Nu_err'] / df['gu']

    # Ajout des propriétés du modèle si applicable
    if is_model and params is not None:
        df['Pth'] = params[0]
        df['G0'] = params[1]
        df['Omega'] = scaling

    # Détection Ortho/Para (Pair=Para, Impair=Ortho)
    df['Ortho_Para'] = np.where(df['Ju'] % 2 == 0, 'Para', 'Ortho')

    return df







def process_to_unified_df(obs_file, obs_name, grid_files, db_file, threshold=0.3, limscale=[0, 10], mol="co"):
    """Génère un DataFrame unifié ET renvoie les objets PyISM pour les contours."""
    df_db = load_physics_database(db_file)

    # 1. Observations
    obs = ism.Observation.from_ASCII_file(obs_file, obs_name)
    obs.set_multiplicative_noise_threshold(threshold=threshold)
    df_all = pyism_to_dataframe(obs, df_db, source_name=obs_name)

    # Traduction
    trans_dict = build_translation_dict(obs.get_observable_names(), mol=mol)
    translation = ism.Translation("obs_to_grid", trans_dict)
    grid_observable_names = translation.get_alternate_names()

    # Dictionnaire pour stocker les objets PyISM utiles
    pyism_objects = {'obs': obs, 'approximators': {}, 'best_models': {}}

    # 2. Modèles
    for grid_file in grid_files:
        grid_name = grid_file.split('/')[-1].replace('.dat', '')
        grid = ism.Model_grid.from_ASCII_file(grid_file, 3, grid_name)
        
        grid.reduce_observables(grid_observable_names)
        grid.translate_names(translation)
        
        approx = ism.Grid_approximator.interpolate_from_grid(grid)
        best_mod, best_chi2 = ism.fit_tools.find_best_interpolated_model(
            obs, approx, N_repetitions=20, scaling_factor_bounds=limscale, cost_function="reduced_chi2"
        )
        
        df_mod = pyism_to_dataframe(
            best_mod, df_db, source_name=grid_name, is_model=True, 
            scaling=best_mod.get_scaling_factor(), params=best_mod.get_parameter_vector()
        )
        
        df_all = pd.concat([df_all, df_mod], ignore_index=True)
        
        # Sauvegarde des objets
        pyism_objects['approximators'][grid_name] = approx
        pyism_objects['best_models'][grid_name] = best_mod
        
        print(f"✅ Modèle {grid_name} ajouté. Chi2 réduit = {best_chi2:.2f}")
    
    # On renvoie le DF principal et les objets PyISM
    return df_all, pyism_objects, df_db


def get_custom_model_df(approximator, params, omega, df_db, model_name="Custom_Model"):
    """
    Génère un DataFrame au format standard pour n'importe quel point (Pth, G0) de la grille.
    params = [Pth, G0]
    """
    # Évaluation du modèle sur la grille interpolée
    custom_model = approximator(params)
    
    # Transformation en DataFrame avec la physique
    df_custom = pyism_to_dataframe(
        custom_model, df_db, source_name=model_name, is_model=True, 
        scaling=omega, params=params
    )
    return df_custom




############################TRACE FIFGURES ################
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def plot_excitation_df(df_unified, title="Diagramme d'Excitation", figsize=(16, 9), 
                       full=False, reg=False, annotate=False, coeff_y=2.0, 
                       op_visible=True, matchcolor=False, thick=2, sizepoint=8):
    """
    Trace le diagramme d'excitation robuste à partir du DataFrame unifié.
    Retrouve TOUTES les fonctionnalités de l'ancien viz.py.
    """
    plt.rcParams.update({'font.size': 18, 'axes.linewidth': 2})
    
    # Récupération des niveaux v présents
    v_levels = sorted(df_unified['vu'].dropna().unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(10, len(v_levels))))
    styles = ['-', '--', ':', '-.']
    marqueur = {'Para': '^', 'Ortho': 'o'}

    # Configuration de la figure globale si full=True
    if full:
        fig_full, ax_full = plt.subplots(figsize=figsize)
        if op_visible:
            ax_full.scatter([], [], c='black', marker='^', label="Para levels")
            ax_full.scatter([], [], c='black', marker='o', label="Ortho levels")

    for v_idx, v in enumerate(v_levels):
        df_v = df_unified[df_unified['vu'] == v]
        if df_v.empty: continue
        
        c_obs = palette[v_idx % len(palette)]
        ax = ax_full if full else plt.subplots(figsize=figsize)[1]

        # --- 1. OBSERVATIONS ---
        df_obs = df_v[df_v['Type'] == 'Obs']
        if not df_obs.empty:
            for op in (['Para', 'Ortho'] if op_visible else ['Para']):
                subset = df_obs if not op_visible else df_obs[df_obs['Ortho_Para'] == op]
                if not subset.empty:
                    # Point et barre d'erreur principale
                    ax.errorbar(subset['Eu_K'], subset['Nu_gu'], yerr=subset['Nu_gu_err'],
                                fmt=marqueur[op] if op_visible else 'o', color=c_obs, 
                                markersize=10, markeredgecolor='white', capsize=3, 
                                elinewidth=2, ecolor=(0, 0, 0, 0.7), zorder=2)
                    
                    # Barre d'erreur design transparente (Sécurisée contre le log(0))
                    # L'erreur basse est maxée à 90% de la valeur pour éviter la chute vers -l'infini
                    yerr_bas = np.minimum(0.9 * subset['Nu_gu'], subset['Nu_gu']) 
                    yerr_haut = 1.0 * subset['Nu_gu']
                    ax.errorbar(subset['Eu_K'], subset['Nu_gu'], yerr=[yerr_bas, yerr_haut],
                                fmt="none", color=c_obs, capsize=1, elinewidth=2, alpha=0.2, zorder=1)

            # Annotation du niveau v
            ax.annotate(f"v={int(v)}", xy=(df_obs['Eu_K'].min(), df_obs['Nu_gu'].max()), 
                        xytext=(df_obs['Eu_K'].min() + 500, df_obs['Nu_gu'].max() * 1.5),
                        color=c_obs, fontweight='bold', fontsize=20)
            
            # Annotation textuelle des raies (J=...)
            if annotate:
                for _, row in df_obs.iterrows():
                    ax.annotate(f"{row['Line']}", (row['Eu_K'], row['Nu_gu'] * 1.25), 
                                textcoords="data", rotation=90, fontsize=10)

            # Régression Linéaire (Température d'excitation)
            if reg and len(df_obs) > 2:
                X = df_obs['Eu_K'].values.reshape(-1, 1)
                y = df_obs['Nu_gu'].values
                model_reg = LinearRegression().fit(X, np.log10(y))
                coef = float(model_reg.coef_[0])
                T_ex = -int(1 / coef) if coef != 0 else 0
                
                y_pred = 10 ** model_reg.predict(X)
                ax.plot(X, y_pred, color='red', linestyle=":", label=f"Vup={int(v)}, T={T_ex} K")

        # --- 2. MODÈLES ---
        models = df_v[df_v['Type'] == 'Model']['Source'].unique()
        for idx, mod_name in enumerate(models):
            df_mod = df_v[df_v['Source'] == mod_name].sort_values(by='Eu_K')
            if not df_mod.empty:
                c_mod = palette[v_idx % len(palette)] if matchcolor else plt.cm.Set1(idx % 9)
                label_mod = f"{mod_name} (Pth={df_mod['Pth'].iloc[0]:.1e}, G0={df_mod['G0'].iloc[0]:.1e})"
                
                # N'ajoute la légende qu'une seule fois si full=True
                label = label_mod if (not full or v_idx == 0) else "_nolegend_"
                ax.plot(df_mod['Eu_K'], df_mod['Nu_gu'], linestyle=styles[idx % len(styles)], 
                        color=c_mod, marker='x', linewidth=thick, markersize=sizepoint, 
                        alpha=0.9, label=label, zorder=3)

        # --- 3. FORMATAGE DE L'AXE ---
        ax.set_yscale('log')
        ax.set_xlabel(r"Energie du niveau supérieur $E_u$ [K]", fontsize=22)
        ax.set_ylabel(r"Densité $N_u/g_u$ [cm$^{-2}$]", fontsize=22)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        # Ajustement des limites Y
        if not df_v.empty:
            ymin, ymax = df_v['Nu_gu'].min(), df_v['Nu_gu'].max()
            ax.set_ylim(ymin / coeff_y, ymax * coeff_y)

        if not full:
            ax.set_title(f"{title} - Bande v={int(v)}", fontsize=24)
            ax.legend(fontsize=16)
            plt.tight_layout()
            plt.show()

    if full:
        ax_full.set_title(f"{title} (Tous les niveaux)", fontsize=24)
        ax_full.legend(fontsize=16)
        plt.tight_layout()
        plt.show()


def plot_intensity_df(df_unified, title="Diagramme d'Intensité", figsize=(21, 9), 
                      op_visible=True):
    """
    Trace le diagramme d'intensité comparatif à partir du DataFrame unifié.
    Respecte l'ordre des énergies pour que les points soient classés logiquement.
    """
    plt.rcParams.update({'font.size': 18, 'axes.linewidth': 2})
    fig, ax = plt.subplots(figsize=figsize)
    
    # Tri par Eu_K pour que l'axe X (les raies) soit ordonné physiquement
    df_sorted = df_unified.sort_values(by=['vu', 'Eu_K'])
    line_order = df_sorted['Line'].unique()
    
    marqueur = {'Para': '^', 'Ortho': 'o'}

    # --- 1. OBSERVATIONS ---
    df_obs = df_sorted[df_sorted['Type'] == 'Obs'].set_index('Line').reindex(line_order)
    
    if op_visible:
        for op, color in zip(['Para', 'Ortho'], ['red', 'blue']):
            subset = df_obs[df_obs['Ortho_Para'] == op]
            if not subset.empty:
                ax.errorbar(subset.index, subset['Intensity'], yerr=subset['Error'], 
                            fmt=marqueur[op], color=color, ecolor='black', markersize=10, 
                            label=f"Obs ({op})", capsize=4, zorder=3)
    else:
        ax.errorbar(df_obs.index, df_obs['Intensity'], yerr=df_obs['Error'], 
                    fmt='s', color='black', ecolor='black', markersize=8, 
                    label="Observations", capsize=4, zorder=3)

    # --- 2. MODÈLES ---
    models = df_unified[df_unified['Type'] == 'Model']['Source'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
    styles = ['-', '--', ':', '-.']
    
    for idx, mod_name in enumerate(models):
        df_mod = df_sorted[df_sorted['Source'] == mod_name].set_index('Line').reindex(line_order)
        ax.plot(df_mod.index, df_mod['Intensity'], linestyle=styles[idx % len(styles)], 
                color=colors[idx], marker='x', linewidth=2, label=mod_name, zorder=2)

    # --- 3. FORMATAGE DE L'AXE ---
    ax.set_yscale('log')
    ax.set_ylabel(r"Intensité [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", fontsize=20)
    ax.set_title(title, fontsize=24)
    plt.xticks(rotation=90, fontsize=12) # Rotation des noms de raies pour lisibilité
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=16)
    
    # On étend un peu les limites X pour ne pas coller les points aux bords
    ax.set_xlim(-1, len(line_order))
    
    plt.tight_layout()
    plt.show()
        
    


def plot_contour_2d(approx, obs, best_model, param_pair=("Go", "Pgas"), 
                    limscale=[0.01, 10], title="Carte de Chi2", color="purple"):
    """Trace une carte de contour 2D entre deux paramètres (ex: Pgas et Go, ou Omega et Pgas)."""
    # On utilise l'outil natif ou personnalisé de ton environnement (ism.ap_tools)
    try:
        contour = ism.ap_tools.fit_quality_contour_plots_singlepair2(
            approx, obs, param_pair, best_model, 
            couleur=color, Title=title, scaling_factor_bounds=limscale
        )
        plt.tight_layout()
        plt.show()
        return contour
    except AttributeError:
        print("⚠️ Fonction fit_quality_contour_plots_singlepair2 introuvable. Assure-toi qu'elle est bien dans pyismtools.ap_tools ou viz.")

def plot_contour_corner(approx, obs, best_model, limscale=[0.01, 10], title="Corner Plot"):
    """Trace le 'corner plot' complet (triangle) pour tous les paramètres + Omega."""
    try:
        contour_all = ism.ap_tools.fit_quality_contour_plots2(
            approx, obs, best_model, scaling_factor_bounds=limscale
        )
        ax = contour_all.gca()
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return contour_all
    except AttributeError:
        print("⚠️ Fonction fit_quality_contour_plots2 introuvable.")
        