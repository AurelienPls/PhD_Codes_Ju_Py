#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Librairie Modulaire pour Modèles PDR (IDAT)
- Lecture et filtrage des zones d'étude
- Moteurs graphiques (1, 2, 3, 4 panneaux découplés et personnalisables)
- Calculs physiques de l'épaisseur optique (tau_0) par intégration spatiale
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. LECTURE ET CONFIGURATION DES DONNÉES
# =============================================================================

def load_pdr_data(file_path, extractor_path):
    """Lit le fichier .dat d'IDAT et applique les noms de colonnes."""
    if not os.path.exists(extractor_path):
        raise FileNotFoundError(f"Extracteur introuvable : {extractor_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Données introuvables : {file_path}")
        
    with open(extractor_path, "r", encoding="utf-8") as f:
        col_names = [line.strip() for line in f if line.strip()]
        
    df = pd.read_csv(file_path, sep=r'\s+', comment='#', names=col_names, skiprows=6)
    df.columns = df.columns.str.replace(' ', '')  # Nettoyage des espaces critiques
    return df

def prepare_study_zone(df, av_min=0.0, av_max=10.0):
    """Filtre le DataFrame sur une plage d'AV et convertit en numérique."""
    df_zone = df[(df["AV"] >= av_min) & (df["AV"] <= av_max)].copy()
    cols_critical = ['AV', 'Distance', 'nH', 'n(H)', 'n(H2)', 'n(CO)', 'Temperature']
    for col in cols_critical:
        if col in df_zone.columns:
            df_zone[col] = pd.to_numeric(df_zone[col], errors='coerce')
    return df_zone.dropna(subset=['AV', 'Distance'])

def get_h2_transition_av(df_zone):
    """Calcule l'AV de la transition H/H2 (croisement n(H) et 2*n(H2))."""
    if "n(H)" in df_zone.columns and "n(H2)" in df_zone.columns:
        diff = (df_zone["n(H)"] - 2 * df_zone["n(H2)"]).abs()
        return df_zone.loc[diff.idxmin(), "AV"]
    return 1.0

#Transitos AX 

def load_electronic_lines_co(lines_a_path):
    """
    Lit de manière robuste le fichier de lignes électroniques line_co_A.dat
    en ignorant les en-têtes textuels et en nettoyant les lignes corrompues.
    """
    import os
    import pandas as pd
    import numpy as np

    if not os.path.exists(lines_a_path):
        raise FileNotFoundError(f"Fichier quantique introuvable : {lines_a_path}")
        
    try:
        # On lit sans assigner de noms de colonnes fixes au départ pour laisser le parseur C respirer.
        # on_bad_lines='skip' permet de sauter la ligne 10 (et toute autre ligne textuelle mal formatée).
        df = pd.read_csv(
            lines_a_path, 
            sep=r'\s+', 
            comment='#',        # Ignore les lignes de commentaires classiques
            header=None,        # Pas d'en-tête automatique
            on_bad_lines='skip' # SÉCURITÉ CRITIQUE : saute les lignes avec trop de champs
        )
        
        # Structure attendue d'un fichier de transitions PDR standards (5 ou 6 colonnes de base)
        expected_cols = ['vu', 'Ju', 'vl', 'Jl', 'Aij_s-1', 'wavenumber_cm1']
        
        # Renommage dynamique selon le nombre réel de colonnes lues
        rename_dict = {i: expected_cols[i] for i in range(min(len(df.columns), len(expected_cols)))}
        df = df.rename(columns=rename_dict)
        
        # Si le fichier contient des colonnes de texte supplémentaires en fin de ligne (ex: Description),
        # on les fusionne proprement sans faire crasher le tableau
        if len(df.columns) > len(expected_cols):
            df['Description'] = df.iloc[:, len(expected_cols):].astype(str).agg(' '.join, axis=1)
        else:
            df['Description'] = "No description"
            
        # Conversion de force des quanta en entiers et des constantes physiques en flottants
        for col in ['vu', 'Ju', 'vl', 'Jl']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        for col in ['Aij_s-1', 'wavenumber_cm1']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Nettoyage final : on supprime les résidus textuels qui n'ont pas pu devenir des entiers
        df = df.dropna(subset=['vu', 'Ju', 'vl', 'Jl']).reset_index(drop=True)
        
        # Conversion stricte en types natifs pour le filtrage
        df['vu'] = df['vu'].astype(int)
        df['Ju'] = df['Ju'].astype(int)
        df['vl'] = df['vl'].astype(int)
        df['Jl'] = df['Jl'].astype(int)

        print(f"✅ Fichier électronique chargé avec succès ({len(df)} transitions valides détectées).")
        return df

    except Exception as e:
        print(f"❌ Erreur critique lors du parsing du fichier de raies électroniques : {e}")
        return None
    
    
    


def load_pdr_lines_electronic(file_path):
    """
    Lit le fichier line_co_A.dat d'après le format natif du code Meudon PDR.
    Structure attendue :
    Col 0: Index | Col 1: J_index | Col 2: Level_index | Col 3: Wavenumber (cm-1)
    Col 4: Aij (s-1) | Col 5: vu | Col 6: Ju | Col 7: vl | Col 8: Jl
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier de lignes introuvable : {file_path}")
        
    clean_rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith('#') or line_str.startswith('!'):
                continue
            elements = line_str.split()
            try:
                # Sécurité : on valide que la ligne commence bien par un index numérique
                float(elements[0])
                # On extrait les 9 premières colonnes essentielles
                clean_rows.append(elements[:9])
            except (ValueError, IndexError):
                continue # Ligne textuelle parasite sautée automatiquement
                
    # Création du DataFrame avec le mapping astrophysique exact
    df = pd.DataFrame(clean_rows)
    df = df.rename(columns={
        3: 'wavenumber_cm1', 
        4: 'Aij_s-1', 
        5: 'vu', 
        6: 'Ju', 
        7: 'vl', 
        8: 'Jl'
    })
    
    # Conversion de force des types de données
    for col in ['vu', 'Ju', 'vl', 'Jl', 'Aij_s-1', 'wavenumber_cm1']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna(subset=['vu', 'Ju', 'vl', 'Jl']).reset_index(drop=True)
    
    for col in ['vu', 'Ju', 'vl', 'Jl']:
        df[col] = df[col].astype(int)
        
    print(f"✅ Catalogue UV line_co_A chargé : {len(df)} transitions valides.")
    return df


def load_pdr_levels_electronic(file_path):
    """
    Lit le fichier level_co_A.dat de manière robuste.
    Structure attendue : v | J | g | E (cm-1)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier de niveaux introuvable : {file_path}")
        
    clean_rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith('#') or line_str.startswith('!'):
                continue
            elements = line_str.split()
            try:
                float(elements[0])
                clean_rows.append(elements[:4])
            except (ValueError, IndexError):
                continue
                
    df = pd.DataFrame(clean_rows, columns=['v', 'J', 'g', 'E_cm1'])
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna(subset=['v', 'J', 'g']).reset_index(drop=True)
    
    for col in ['v', 'J', 'g']:
        df[col] = df[col].astype(int)
        
    print(f"✅ Catalogue UV level_co_A chargé : {len(df)} niveaux électroniques valides.")
    return df



def load_pdr_atomic_files(file_path, expected_cols, comment_char='#'):
    """
    Charge de manière générique les fichiers de structure quantique (.dat)
    en éliminant toutes les lignes de texte ou d'en-têtes parasites.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    try:
        clean_rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith(comment_char) or line_str.startswith('!'):
                    continue
                elements = line_str.split()
                try:
                    float(elements[0])  # Vérifie que la ligne commence par un nombre
                    clean_rows.append(elements[:len(expected_cols)])
                except (ValueError, IndexError):
                    continue
        df = pd.DataFrame(clean_rows, columns=expected_cols)
        for col in expected_cols:
            if col in ['v', 'J', 'vu', 'Ju', 'vl', 'Jl', 'g']:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna(subset=[expected_cols[0]]).reset_index(drop=True)
    except Exception as e:
        print(f"❌ Erreur lors du parsing de {os.path.basename(file_path)} : {e}")
        return None

# =============================================================================
# 3. MOTEURS GRAPHIQUES & STYLE
# =============================================================================

def apply_publication_style(font_size=14):
    """Applique une configuration matplotlib propre de qualité publication."""
    plt.rcParams.update({
        'font.size': font_size,
        'axes.labelsize': font_size + 2,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size - 2,
        'axes.linewidth': 1.5,
        'xtick.major.size': 6, 'ytick.major.size': 6,
        'xtick.direction': 'in', 'ytick.direction': 'in'
    })

def setup_double_top_axis(ax_base, df_zone, D_pc=414.0):
    """Génère les échelles supérieures synchronisées (pc et arcsec)."""
    ax_pc = ax_base.twiny()
    ax_arc = ax_base.twiny()
    ax_arc.spines['top'].set_position(('outward', 40))
    
    av_min, av_max = ax_base.get_xlim()
    idx_min = (df_zone["AV"] - av_min).abs().idxmin()
    idx_max = (df_zone["AV"] - av_max).abs().idxmin()
    
    dist_bounds = [df_zone.loc[idx_min, "Distance"], df_zone.loc[idx_max, "Distance"]]
    pc_bounds = [d / 3.085677581e18 for d in dist_bounds]
    arc_bounds = [(p / D_pc) * 206264.8 for p in pc_bounds]
    
    ax_pc.set_xlim(pc_bounds)
    ax_pc.set_xlabel("Physical Distance (pc)", labelpad=8)
    ax_arc.set_xlim(arc_bounds)
    ax_arc.set_xlabel(f"Angular Scale (arcsec) at {D_pc} pc", labelpad=8)
    return ax_pc, ax_arc

def plot_background_thermique(ax, df_zone, fields=None):
    """Trace le profil d'environnement (densités de fond)."""
    if fields is None:
        fields = [("n(CO)", r"$n(\mathrm{CO})$", "black", "-"),
                  ("n(H2)", r"$n(\mathrm{H}_2)$", "tab:blue", "-"),
                  ("n(H)", r"$n(\mathrm{H})$", "tab:green", "-")]
    for col, lbl, color, ls in fields:
        if col in df_zone.columns:
            ax.plot(df_zone["AV"], df_zone[col], label=lbl, color=color, linestyle=ls)
    ax.set_ylabel(r"Density ($\mathrm{cm}^{-3}$)")
    ax.set_yscale('log')

def plot_background_temperature(ax, df_zone, custom_grains=None):
    """Trace T_gas et le gradient des températures de grains."""
    ax_t = ax.twinx()
    ax_t.plot(df_zone["AV"], df_zone["Temperature"], label=r"$T_{\mathrm{gas}}$", color='tab:red', linewidth=2.5)
    
    cmap_g = plt.get_cmap('copper')
    grain_list = custom_grains if custom_grains is not None else [1, 6, 12]
    for i in grain_list:
        g_col = f"Graintemperature(r{i:02d})"
        if g_col in df_zone.columns:
            ax_t.plot(df_zone["AV"], df_zone[g_col], color=cmap_g(i/12), linestyle=':', alpha=0.7, label=f"$T_{{gr}}$ r{i:02d}")
            
    ax_t.set_ylabel("Temperature (K)", color='tab:red')
    ax_t.tick_params(axis='y', labelcolor='tab:red')
    return ax_t

def plot_quad_panel(df_zone, species="CO", normalize=True, D_pc=414.0, title_prefix=""):
    """Traceur complet à 4 panneaux empilés."""
    apply_publication_style(font_size=12)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(11, 14), sharex=True, gridspec_kw={'hspace': 0})
    
    plot_background_thermique(ax1, df_zone)
    ax1_t = plot_background_temperature(ax1, df_zone)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_t.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="lower left", ncols=2)
    
    if species.upper() == "CO":
        j_range, j_labels = range(1, 15), [0, 3, 6, 9, 13]
        v_configs = [(0, 'viridis', r"CO $v=0$"), (1, 'plasma', r"CO $v=1$"), (2, 'magma', r"CO $v=2$")]
        prefix = "n(COv={v},J={j})"
    else:
        j_range, j_labels = range(0, 15), [0, 1, 3, 5, 7, 10, 14]
        v_configs = [(0, 'YlGn', r"$\mathrm{H}_2\ v=0$"), (1, 'PuBu', r"$\mathrm{H}_2\ v=1$"), (2, 'Oranges', r"$\mathrm{H}_2\ v=2$")]
        prefix = "n(H2v={v},J={j})"
        
    av_trans = get_h2_transition_av(df_zone)
    for ax_sub, (v_num, cmap_name, lbl_t) in zip([ax2, ax3, ax4], v_configs):
        cmap = plt.get_cmap(cmap_name)
        valid_cols = [prefix.format(v=v_num, j=j) for j in j_range if prefix.format(v=v_num, j=j) in df_zone.columns]
        if not valid_cols: continue
        
        max_panel = max([pd.to_numeric(df_zone[c], errors='coerce').max() for c in valid_cols])
        for idx, j_val in enumerate(j_range):
            c_name = prefix.format(v=v_num, j=j_val)
            if c_name in df_zone.columns:
                y_data = pd.to_numeric(df_zone[c_name], errors='coerce').fillna(0).values
                m_loc = y_data.max()
                if m_loc <= 0: continue
                y_plot = y_data / m_loc if normalize else y_data
                lbl = f"$J={j_val}$" if idx in j_labels else ""
                ax_sub.plot(df_zone["AV"], y_plot, color=cmap(0.3 + 0.7*(idx/len(j_range))), label=lbl)
                
        ax_sub.set_ylabel("Norm. Pop" if normalize else "Pop (cm$^{-3}$)")
        if not normalize: ax_sub.set_yscale('log')
        ax_sub.legend(title=lbl_t, loc="upper right", ncols=2)
        
    ax4.set_xlabel(r"Visual Extinction $A_V$ (mag)")
    ax4.set_xlim([df_zone["AV"].min(), df_zone["AV"].max()])
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(x=av_trans, color='gray', linestyle='--', alpha=0.5)
    setup_double_top_axis(ax1, df_zone, D_pc=D_pc)
    fig.suptitle(f"{title_prefix} - {species}", y=0.96, fontsize=14, fontweight='bold')
    return fig, (ax1, ax2, ax3, ax4)


def plot_custom_tracers(ax, df_zone, tracers_list, normalize=True):
    """
    Trace une liste sur-mesure de molécules/transitions sur un axe donné.
    tracers_list: liste de tuples (nom_colonne, label_latex, couleur, style_ligne)
    """
    for col_name, label_latex, color, ls in tracers_list:
        if col_name in df_zone.columns:
            y_data = pd.to_numeric(df_zone[col_name], errors='coerce').fillna(0).values
            max_val = y_data.max()
            if max_val <= 0: 
                continue
            
            y_plot = y_data / max_val if normalize else y_data
            ax.plot(df_zone["AV"], y_plot, color=color, linestyle=ls, label=label_latex, linewidth=2.5)
            
            # Repère sur le pic d'émission
            idx_max = df_zone[col_name].idxmax()
            ax.scatter(df_zone.loc[idx_max, "AV"], y_plot[df_zone.index.get_loc(idx_max)], color=color, s=35, zorder=5)
        else:
            print(f"ℹ️ Traceur {col_name} absent des données. Ignoré.")
            
    ax.set_ylabel("Normalized Population" if normalize else r"Density $n_u$ ($\mathrm{cm}^{-3}$)")
    if not normalize: 
        ax.set_yscale('log')
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper right", ncols=2)


def plot_flexible_quad_panel(df_data, species="CO", normalize=True, av_max=7.0, title_prefix="", cmap_set=None):
    """
    Moteur à 4 panneaux ultra-flexible issu de IDAT_AV_easy.
    Permet de choisir la molécule, d'inclure automatiquement n(C) et n(O), 
    et de projeter la transition H/H2 et C/CO.
    """
    # Filtrage et nettoyage
    df_zone = df_data[df_data["AV"] <= av_max].copy()
    df_zone.columns = df_zone.columns.str.replace(' ', '')
    
    # Extraction des transitions structurelles
    diff_h = (df_zone["n(H)"] - 2 * df_zone["n(H2)"]).abs()
    av_trans_h = df_zone.loc[diff_h.idxmin(), "AV"]
    
    av_trans_c = None
    if "n(C)" in df_zone.columns and "n(CO)" in df_zone.columns:
        try:
            df_apres_h2 = df_zone[df_zone["AV"] > av_trans_h]
            if not df_apres_h2.empty:
                idx_c = (df_apres_h2["n(C)"] - df_apres_h2["n(CO)"]).abs().idxmin()
                av_trans_c = df_apres_h2.loc[idx_c, "AV"]
        except:
            pass

    # Configuration des familles de v et styles de sauts quantiques
    if species.upper() == "CO":
        j_range = range(1, 15)
        j_labels = [0, 3, 6, 9, 13]
        v_configs = cmap_set if cmap_set else [(0, 'viridis', r"CO $v=0$"), (1, 'plasma', r"CO $v=1$"), (2, 'magma', r"CO $v=2$")]
        prefix = "n(COv={v},J={j})"
    else:
        j_range = range(0, 15)
        j_labels = [0, 1, 3, 5, 7, 10, 14]
        v_configs = cmap_set if cmap_set else [(0, 'YlGn', r"$\mathrm{H}_2\ v=0$"), (1, 'PuBu', r"$\mathrm{H}_2\ v=1$"), (2, 'Oranges', r"$\mathrm{H}_2\ v=2$")]
        prefix = "n(H2v={v},J={j})"

    apply_publication_style(font_size=12)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True, gridspec_kw={'hspace': 0})
    
    # Panneau 1 : Physique globale avec C et O inclus
    plot_background_thermique(ax1, df_zone)
    if "n(C)" in df_zone.columns:
        ax1.plot(df_zone["AV"], df_zone["n(C)"], label=r"$n(\mathrm{C})$", color='tab:purple', linestyle='--')
    if "n(O)" in df_zone.columns:
        ax1.plot(df_zone["AV"], df_zone["n(O)"], label=r"$n(\mathrm{O})$", color='tab:brown', linestyle='--')
        
    ax1_t = plot_background_temperature(ax1, df_zone)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_t.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="lower left", ncols=2)
    
    # Panneaux d'excitation rotationnelle dynamique
    axes_v = [ax2, ax3, ax4]
    for ax, (v_num, cmap_name, lbl_t) in zip(axes_v, v_configs):
        cmap = plt.get_cmap(cmap_name)
        valid_cols = [prefix.format(v=v_num, j=j) for j in j_range if prefix.format(v=v_num, j=j) in df_zone.columns]
        if not valid_cols: 
            continue
            
        max_panel = max([pd.to_numeric(df_zone[c], errors='coerce').max() for c in valid_cols])
        for idx, j_val in enumerate(j_range):
            c_name = prefix.format(v=v_num, j=j_val)
            if c_name in df_zone.columns:
                y_data = pd.to_numeric(df_zone[c_name], errors='coerce').fillna(0).values
                m_loc = y_data.max()
                if m_loc <= 0: 
                    continue
                y_plot = y_data / m_loc if normalize else y_data
                lbl = f"$J={j_val}$" if idx in j_labels else ""
                color_frac = 0.25 + 0.75 * (idx / len(j_range))
                ax.plot(df_zone["AV"], y_plot, color=cmap(color_frac), label=lbl)
                ax.scatter(df_zone.loc[df_zone[c_name].idxmax(), "AV"], y_plot.max() if normalize else m_loc, color=cmap(color_frac), s=15, zorder=5)
                
        ax.set_ylabel("Norm. Pop" if normalize else "Pop (cm$^{-3}$)")
        if not normalize: 
            ax.set_yscale('log')
        ax.legend(title=lbl_t, loc="upper right", ncols=2)
        ax.grid(True, which="both", ls=":", alpha=0.4)

    ax4.set_xlabel(r"Visual Extinction $A_V$ (mag)")
    ax4.set_xlim([df_zone["AV"].min(), df_zone["AV"].max()])
    
    # Application transversale des lignes de transition de fronts
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(x=av_trans_h, color='gray', linestyle='--', alpha=0.7)
        if av_trans_c is not None:
            ax.axvline(x=av_trans_c, color='gray', linestyle='--', alpha=0.7)
    plt.suptitle(title_prefix)
    fig.tight_layout()
            
    return fig, (ax1, ax2, ax3, ax4)
def plot_h2_excitation_distribution(df_data, av_max=7.0, cmap_set=None):
    """
    Génère un Quad-Panel H2 optimisé pour la visualisation de l'excitation :
      - Panneau 1 : Densités globales + Sommes Ortho et Para H2 globales.
      - Panneaux 2, 3, 4 : Distribution rotationnelle interne pour v=0, 1, 2.
                           Chaque niveau J (à partir de J=2) est divisé par la densité 
                           totale de son propre état vibrationnel n(v) = Sum_J n(v, J).
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 1. Nettoyage et filtrage spatial
    df_zone = df_data[df_data["AV"] <= av_max].copy()
    df_zone.columns = df_zone.columns.str.replace(' ', '')
    
    # --- CALCUL DES SOMMES ORTHO / PARA GLOBALES (Pour le Panneau 1) ---
    all_h2_cols = [c for c in df_zone.columns if c.startswith("n(H2v=")]
    sum_para = np.zeros(len(df_zone))
    sum_ortho = np.zeros(len(df_zone))
    
    for col in all_h2_cols:
        try:
            j_val = int(col.split("J=")[1].replace(")", ""))
            vec_data = pd.to_numeric(df_zone[col], errors='coerce').fillna(0).values
            if j_val % 2 == 0:
                sum_para += vec_data
            else:
                sum_ortho += vec_data
        except:
            continue

    # Front de transition H / H2
    diff_h = (df_zone["n(H)"] - 2 * df_zone["n(H2)"]).abs()
    av_trans_h = df_zone.loc[diff_h.idxmin(), "AV"]

    # Configuration (On restreint volontairement J de 2 à 7 pour nettoyer le graphique)
    j_range = range(2, 8) 
    j_labels = [2, 3, 4, 5, 6, 7]
    
    v_configs = cmap_set if cmap_set else [
        (0, 'YlGn', r"$\mathrm{H}_2\ v=0$ (Intra-v Distribution, $J \geq 2$)"), 
        (1, 'PuBu', r"$\mathrm{H}_2\ v=1$ (Intra-v Distribution, $J \geq 2$)"), 
        (2, 'Oranges', r"$\mathrm{H}_2\ v=2$ (Intra-v Distribution, $J \geq 2$)")
    ]
    prefix = "n(H2v={v},J={j})"

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True, gridspec_kw={'hspace': 0})
    
    # =========================================================================
    # PANNEAU 1 : RÉSUMÉ THERMIQUE ET CHIMIQUE
    # =========================================================================
    if 'plot_background_thermique' in dir():
        plot_background_thermique(ax1, df_zone)
    else:
        ax1.plot(df_zone["AV"], df_zone["n(H2)"], label=r"$n(\mathrm{H}_2)$", color='tab:blue', linewidth=2)
        ax1.plot(df_zone["AV"], df_zone["n(H)"], label=r"$n(\mathrm{H})$", color='tab:green', linewidth=2)
        if "n(CO)" in df_zone.columns:
            ax1.plot(df_zone["AV"], df_zone["n(CO)"], label=r"$n(\mathrm{CO})$", color='black', linewidth=1.5)
            
    ax1.plot(df_zone["AV"], sum_para, label=r"$\sum \mathrm{H}_2\ \mathrm{Para}$", color='darkcyan', linestyle=':', linewidth=2)
    ax1.plot(df_zone["AV"], sum_ortho, label=r"$\sum \mathrm{H}_2\ \mathrm{Ortho}$", color='indigo', linestyle='-.', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_ylim([1e-1, ax1.get_ylim()[1] * 2])
    
    if 'plot_background_temperature' in dir():
        ax1_t = plot_background_temperature(ax1, df_zone)
    else:
        ax1_t = ax1.twinx()
        ax1_t.plot(df_zone["AV"], df_zone["Temperature"], color='tab:red', label=r"$T_{\mathrm{gas}}$")
        ax1_t.set_yscale('log')
        
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_t.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="lower left", ncols=2, fontsize=10)
    
    # =========================================================================
    # PANNEAUX 2, 3, 4 : DISTRIBUTION INTERNE PAR ÉTAT VIBRATIONNEL (n(v,J) / n(v))
    # =========================================================================
    axes_v = [ax2, ax3, ax4]
    
    for ax, (v_num, cmap_name, lbl_t) in zip(axes_v, v_configs):
        cmap = plt.get_cmap(cmap_name)
        
        # Calcul de la densité totale de cet état vibrationnel précis : n(v) = Sum_J n(v, J)
        # On somme sur TOUS les J (0 à 14) pour avoir le vrai réservoir de v
        n_v_total = np.zeros(len(df_zone))
        for j_all in range(0, 15):
            c_all = prefix.format(v=v_num, j=j_all)
            if c_all in df_zone.columns:
                n_v_total += pd.to_numeric(df_zone[c_all], errors='coerce').fillna(0).values
                
        # Tracé des fractions thermiques internes pour J >= 2
        for idx, j_val in enumerate(j_range):
            c_name = prefix.format(v=v_num, j=j_val)
            
            if c_name in df_zone.columns:
                y_raw = pd.to_numeric(df_zone[c_name], errors='coerce').fillna(0).values
                
                # Normalisation par la population de son propre état vibrationnel n(v)
                y_plot = np.where(n_v_total > 0, y_raw / n_v_total, 0.0)
                
                # Sécurité : on n'affiche pas les courbes horizontalement plates à 0 (niveaux vides)
                if y_plot.max() <= 1e-7:
                    continue
                
                lbl = f"$J={j_val}$" if j_val in j_labels else ""
                # Dégradé de couleur contrôlé : J bas = clair, J haut = sombre
                color_frac = 0.35 + 0.60 * (idx / (len(j_range) - 1))
                
                ax.plot(df_zone["AV"], y_plot, color=cmap(color_frac), label=lbl)
                
                # Positionnement du marqueur au sommet de la distribution interne
                idx_max_loc = np.argmax(y_plot)
                ax.scatter(df_zone.loc[df_zone.index[idx_max_loc], "AV"], y_plot[idx_max_loc], color=cmap(color_frac), s=15, zorder=5)
                
        ax.set_ylabel(r"Fraction $\frac{n(v,J)}{n(v)}$")
        ax.legend(title=lbl_t, loc="upper right", ncols=2, fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.4)

    # Axe de position final
    ax4.set_xlabel(r"Visual Extinction $A_V$ (mag)", fontsize=13)
    ax4.set_xlim([df_zone["AV"].min(), df_zone["AV"].max()])
    
    # Projection géométrique des frontières
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(x=av_trans_h, color='gray', linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', direction='in', which='both', top=True)
        
    return fig, (ax1, ax2, ax3, ax4)








def plot_h2_ortho_para_fraction_panel(df_data, av_max=7.0, cmap_set=None):
    """
    Génère un Quad-Panel dédié à l'H2 :
      - Panneau 1 : Densités globales + Courbes de la somme H2-Para (J pairs) 
                    et H2-Ortho (J impairs) calculées à la volée.
      - Panneaux 2, 3, 4 : Populations rotationnelles (v=0, 1, 2) normalisées 
                           d'abord par n(H2) total (donnant la fraction d'excitation), 
                           puis recalées au maximum (pic) du panneau.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 1. Nettoyage et filtrage spatial
    df_zone = df_data[df_data["AV"] <= av_max].copy()
    df_zone.columns = df_zone.columns.str.replace(' ', '')
    
    # Récupération de la densité totale de H2 pour la normalisation physique
    if "n(H2)" not in df_zone.columns:
        raise KeyError("La colonne globale 'n(H2)' est indispensable pour ce tracé.")
    n_h2_tot = pd.to_numeric(df_zone["n(H2)"], errors='coerce').fillna(0).values
    
    # -------------------------------------------------------------------------
    # 2. CALCUL AUTOMATIQUE DU REPARTITEUR ORTHO / PARA H2
    # -------------------------------------------------------------------------
    all_h2_cols = [c for c in df_zone.columns if c.startswith("n(H2v=")]
    
    sum_para = np.zeros(len(df_zone))
    sum_ortho = np.zeros(len(df_zone))
    
    for col in all_h2_cols:
        try:
            j_part = col.split("J=")[1].replace(")", "")
            j_val = int(j_part)
            vec_data = pd.to_numeric(df_zone[col], errors='coerce').fillna(0).values
            
            if j_val % 2 == 0:
                sum_para += vec_data   # J pair = Para-H2
            else:
                sum_ortho += vec_data  # J impair = Ortho-H2
        except:
            continue

    # Front de transition atomique/moléculaire H/H2
    diff_h = (df_zone["n(H)"] - 2 * df_zone["n(H2)"]).abs()
    av_trans_h = df_zone.loc[diff_h.idxmin(), "AV"]

    # Configuration des colormaps et quanta de rotation (J=0 à 7 pour l'H2 standard)
    j_range = range(0, 8)
    j_labels = [0, 1, 2, 3, 4, 5, 6, 7]
    v_configs = cmap_set if cmap_set else [
        (0, 'YlGn', r"$\mathrm{H}_2\ v=0$ [Fraction]"), 
        (1, 'PuBu', r"$\mathrm{H}_2\ v=1$ [Fraction]"), 
        (2, 'Oranges', r"$\mathrm{H}_2\ v=2$ [Fraction]")
    ]
    prefix = "n(H2v={v},J={j})"

    # Initialisation de la figure 4 panneaux collés
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True, gridspec_kw={'hspace': 0})
    
    # =========================================================================
    # PANNEAU 1 : PHYSIQUE GLOBALE AVEC AJOUTS DES REPARTITIONS ORTHO/PARA
    # =========================================================================
    if 'plot_background_thermique' in list(globals().keys()) or 'plot_background_thermique' in dir():
        plot_background_thermique(ax1, df_zone)
    else:
        ax1.plot(df_zone["AV"], df_zone["n(H2)"], label=r"$n(\mathrm{H}_2)$", color='tab:blue', linewidth=2)
        ax1.plot(df_zone["AV"], df_zone["n(H)"], label=r"$n(\mathrm{H})$", color='tab:green', linewidth=2)
        if "n(CO)" in df_zone.columns:
            ax1.plot(df_zone["AV"], df_zone["n(CO)"], label=r"$n(\mathrm{CO})$", color='black', linewidth=1.5)
            
    # Ajout des deux nouvelles courbes d'ortho et para H2
    ax1.plot(df_zone["AV"], sum_para, label=r"$\sum \mathrm{H}_2\ \mathrm{Para\ (J\ pairs)}$", color='darkcyan', linestyle=':', linewidth=2)
    ax1.plot(df_zone["AV"], sum_ortho, label=r"$\sum \mathrm{H}_2\ \mathrm{Ortho\ (J\ impairs)}$", color='indigo', linestyle='-.', linewidth=2)
    
    ax1.set_yscale('log')
    ax1.set_ylim([1e-1, ax1.get_ylim()[1] * 2])
    
    if 'plot_background_temperature' in list(globals().keys()) or 'plot_background_temperature' in dir():
        ax1_t = plot_background_temperature(ax1, df_zone)
    else:
        ax1_t = ax1.twinx()
        ax1_t.plot(df_zone["AV"], df_zone["Temperature"], color='tab:red', label=r"$T_{\mathrm{gas}}$")
        ax1_t.set_yscale('log')
        
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_t.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="lower left", ncols=2, fontsize=10)
    
    # =========================================================================
    # PANNEAUX 2, 3, 4 : EXCITATION DES FRACTIONS RELATIVES (NORMALISATION AU PIC)
    # =========================================================================
    axes_v = [ax2, ax3, ax4]
    
    for ax, (v_num, cmap_name, lbl_t) in zip(axes_v, v_configs):
        cmap = plt.get_cmap(cmap_name)
        
        valid_cols = [prefix.format(v=v_num, j=j) for j in j_range if prefix.format(v=v_num, j=j) in df_zone.columns]
        if not valid_cols:
            continue
            
        max_fraction_panel = 0.0
        dict_fractions_locales = {}
        
        # Calcul des fractions : n(v,J) / n(H2_total)
        for j_val in j_range:
            c_name = prefix.format(v=v_num, j=j_val)
            if c_name in df_zone.columns:
                y_raw = pd.to_numeric(df_zone[c_name], errors='coerce').fillna(0).values
                fraction_excitation = np.where(n_h2_tot > 0, y_raw / n_h2_tot, 0.0)
                dict_fractions_locales[j_val] = fraction_excitation
                
                if fraction_excitation.max() > max_fraction_panel:
                    max_fraction_panel = fraction_excitation.max()
                    
        # Tracé normalisé par rapport au pic du panneau
        for idx, j_val in enumerate(j_range):
            if j_val in dict_fractions_locales:
                fraction_excitation = dict_fractions_locales[j_val]
                
                if max_fraction_panel <= 0:
                    continue
                    
                y_plot = fraction_excitation / max_fraction_panel
                
                lbl = f"$J={j_val}$" if j_val in j_labels else ""
                color_frac = 0.25 + 0.70 * (idx / (len(j_range) - 1))
                
                # CORRECTION : Utilisation uniforme de color_frac ici
                ax.plot(df_zone["AV"], y_plot, color=cmap(color_frac), label=lbl)
                
                idx_max_frac = np.argmax(y_plot)
                ax.scatter(df_zone.loc[df_zone.index[idx_max_frac], "AV"], y_plot[idx_max_frac], color=cmap(color_frac), s=15, zorder=5)
                
        ax.set_ylabel(r"Norm. Fraction $\frac{n(v,J)}{n(\mathrm{H}_2)}$")
        ax.set_ylim([0, 1.25])
        ax.legend(title=lbl_t, loc="upper right", ncols=2, fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.4)

    ax4.set_xlabel(r"Visual Extinction $A_V$ (mag)", fontsize=13)
    ax4.set_xlim([df_zone["AV"].min(), df_zone["AV"].max()])
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(x=av_trans_h, color='gray', linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', direction='in', which='both', top=True)
        
    return fig, (ax1, ax2, ax3, ax4)
# =============================================================================
# CALCULS ET DEVELOPPEMENT des grandeurs secondaires (épaisseur optique...)
# =============================================================================


def compute_tau_H2_branche_S(df_zone, v_u, v_l, J_l, df_lines, df_level, delta_v=1.0e5):
    """
    Calcule rigoureusement tau_0 pour l'H2 (Branche S, J_u = J_l + 2) par intégration géométrique.
    """
    c_light = 2.99792458e10
    h_planck = 6.62607015e-27
    k_boltz = 1.380649e-16
    J_u = J_l + 2
    
    line_row = df_lines[(df_lines['vu'] == v_u) & (df_lines['Ju'] == J_u) & 
                        (df_lines['vl'] == v_l) & (df_lines['Jl'] == J_l)]
    if line_row.empty: 
        return None
        
    A_ul = float(line_row['Aij_s-1'].iloc[0])
    E_u_K = float(line_row['E_K'].iloc[0])
    
    g_u = float(df_level[(df_level['v'] == v_u) & (df_level['J'] == J_u)]['g'].iloc[0])
    g_l = float(df_level[(df_level['v'] == v_l) & (df_level['J'] == J_l)]['g'].iloc[0])
    
    g_l_row = df_level[(df_level['v'] == v_l) & (df_level['J'] == J_l)]
    E_l_K = float(g_l_row['En(Kexport)'].iloc[0] if 'En(Kexport)' in g_l_row.columns else g_l_row['En(K)'].iloc[0])
    
    nu = ((E_u_K - E_l_K) * k_boltz) / h_planck
    if nu <= 0: 
        return None
        
    n_l = pd.to_numeric(df_zone[f"n(H2v={v_l},J={J_l})"], errors='coerce').fillna(0).values
    n_u = pd.to_numeric(df_zone[f"n(H2v={v_u},J={J_u})"], errors='coerce').fillna(0).values
    dist_cm = pd.to_numeric(df_zone["Distance"], errors='coerce').values
    
    ds = np.diff(dist_cm, prepend=dist_cm[0])
    N_l = np.cumsum(n_l * ds)
    N_u = np.cumsum(n_u * ds)
    
    prefactor = (A_ul * (c_light**3)) / (8.0 * np.pi * (nu**3) * delta_v)
    return np.clip(prefactor * (N_l * (g_u / g_l) - N_u), a_min=0, a_max=None)




def compute_tau_CO_vibrational(df_zone, v_u, J_u, v_l, J_l, df_lines, df_level, delta_v=1.0e5):
    """Calcule tau_0 pour les transitions ro-vibrationnelles du CO (état fondamental X)."""
    c_light = 2.99792458e10
    line_row = df_lines[(df_lines['vu'] == int(v_u)) & (df_lines['Ju'] == int(J_u)) & 
                        (df_lines['vl'] == int(v_l)) & (df_lines['Jl'] == int(J_l))]
    if line_row.empty: return None
    
    A_ul = float(line_row['Aij_s-1'].iloc[0])
    desc = str(line_row['Description'].iloc[0])
    nu = float(desc.split('GHz')[0].strip()) * 1.0e9 if 'GHz' in desc else 1.15e11
    
    g_u = float(df_level[(df_level['v'] == int(v_u)) & (df_level['J'] == int(J_u))]['g'].iloc[0])
    g_l = float(df_level[(df_level['v'] == int(v_l)) & (df_level['J'] == int(J_l))]['g'].iloc[0])
    
    col_l, col_u = f"n(COv={v_l},J={J_l})", f"n(COv={v_u},J={J_u})"
    if col_l not in df_zone.columns or col_u not in df_zone.columns: return None
    
    n_l = pd.to_numeric(df_zone[col_l], errors='coerce').fillna(0).values
    n_u = pd.to_numeric(df_zone[col_u], errors='coerce').fillna(0).values
    
    ds = np.diff(pd.to_numeric(df_zone["Distance"]).values, prepend=pd.to_numeric(df_zone["Distance"]).iloc[0])
    N_l, N_u = np.cumsum(n_l * ds), np.cumsum(n_u * ds)
    
    prefactor = (A_ul * (c_light**3)) / (8.0 * np.pi * (nu**3) * delta_v)
    return np.clip(prefactor * (N_l * (g_u / g_l) - N_u), a_min=0, a_max=None)



def compute_tau_H2_branche_S(df_zone, v_u, v_l, J_l, df_lines, df_level, delta_v=1.0e5):
    """Calcule tau_0 pour l'H2 (Branche S, Ju = Jl + 2)."""
    c_light = 2.99792458e10
    h_planck, k_boltz = 6.62607015e-27, 1.380649e-16
    J_u = J_l + 2
    
    line_row = df_lines[(df_lines['vu'] == int(v_u)) & (df_lines['Ju'] == int(J_u)) & 
                        (df_lines['vl'] == int(v_l)) & (df_lines['Jl'] == int(J_l))]
    if line_row.empty: return None
    
    A_ul = float(line_row['Aij_s-1'].iloc[0])
    E_u_K = float(line_row['E_K'].iloc[0])
    
    g_u = float(df_level[(df_level['v'] == int(v_u)) & (df_level['J'] == int(J_u))]['g'].iloc[0])
    g_l = float(df_level[(df_level['v'] == int(v_l)) & (df_level['J'] == int(J_l))]['g'].iloc[0])
    
    g_l_row = df_level[(df_level['v'] == int(v_l)) & (df_level['J'] == int(J_l))]
    E_l_K = float(g_l_row['En(Kexport)'].iloc[0] if 'En(Kexport)' in g_l_row.columns else g_l_row['En(K)'].iloc[0])
    
    nu = ((E_u_K - E_l_K) * k_boltz) / h_planck
    if nu <= 0: return None
    
    col_l, col_u = f"n(H2v={v_l},J={J_l})", f"n(H2v={v_u},J={J_u})"
    if col_l not in df_zone.columns or col_u not in df_zone.columns: return None
    
    n_l = pd.to_numeric(df_zone[col_l], errors='coerce').fillna(0).values
    n_u = pd.to_numeric(df_zone[col_u], errors='coerce').fillna(0).values
    
    ds = np.diff(pd.to_numeric(df_zone["Distance"]).values, prepend=pd.to_numeric(df_zone["Distance"]).iloc[0])
    N_l, N_u = np.cumsum(n_l * ds), np.cumsum(n_u * ds)
    
    prefactor = (A_ul * (c_light**3)) / (8.0 * np.pi * (nu**3) * delta_v)
    return np.clip(prefactor * (N_l * (g_u / g_l) - N_u), a_min=0, a_max=None)







#Epaisseur optique complete issue de Draine 2011 propre et documentée 
def compute_tau_CO_electronic_AX_strict(df_zone, v_u, J_u, v_l, J_l, df_lines_ax, df_level_co, delta_v=1.0e5):
    """
    RÉFÉRENCE : Équation (9.8) / (9.2) - B.T. Draine (2011), "Physics of the Interstellar 
    and Intergalactic Medium", Chapitre 9 (Absorption Lines), Page 76.
    
    Calcule l'épaisseur optique au centre de la raie (tau_0) pour les transitions
    d'absorption électroniques UV (bande A-X du CO / Fourth-Positive) en unités CGS.
    On suppose l'état électronique excité dépeuplé par fluorescence (N_u ~ 0).
    
    FORMULE THÉORIQUE CALCULÉE :
        tau_0(A_V) = [ (A_ul * c^3) / (8 * pi * nu^3 * delta_v) ] * N_l(A_V) * (g_u / g_l)
    """
    # 1. CONSTANTES PHYSIQUES ET PARAMÈTRES (UNITÉS CGS)
    # c_light : Vitesse de la lumière dans le vide -> Unité : cm / s
    c_light = 2.99792458e10 
    
    # delta_v : Largeur de raie Doppler effective (thermique + turbulente) 
    # Reçue en paramètre (défaut 1.0 km/s) -> Unité : cm / s (1.0e5 cm/s)
    delta_v_cgs = float(delta_v)

    # 2. EXTRACTION DES PROPRIÉTÉS DE LA TRANSITION (CATALOGUE DE RAIES)
    # Isolement de la ligne correspondante dans le catalogue de raies électroniques UV
    line_row = df_lines_ax[
        (df_lines_ax['vu'] == int(v_u)) & (df_lines_ax['Ju'] == int(J_u)) & 
        (df_lines_ax['vl'] == int(v_l)) & (df_lines_ax['Jl'] == int(J_l))    ]
    
    if line_row.empty: 
        print(f"⚠️ Transition UV {v_u},{J_u} <- {v_l},{J_l} absente du catalogue line_co_A.")
        return None
    
    # A_ul : Taux d'émission spontanée d'Einstein -> Unité : s^-1
    # Source : Catalogue 'line_co_A.dat', colonne 'Aij_s-1'
    A_ul = float(line_row['Aij_s-1'].iloc[0])
    
    # sigma : Nombre d'onde central de la transition -> Unité : cm^-1
    # Source : Catalogue 'line_co_A.dat', colonne 'wavenumber_cm1'
    sigma = float(line_row['wavenumber_cm1'].iloc[0])
    
    # nu : Fréquence centrale de la raie d'absorption -> Unité : Hz (s^-1)
    # Calculée via la relation spectroscopique : nu = c * sigma
    nu = c_light * sigma
    
    # 3. EXTRACTION DES POIDS STATISTIQUES (DEGENERESCENCES QUANTIQUES)
    # g_u : Poids statistique du niveau électronique supérieur (A) -> Sans unité
    # Détermination théorique stricte : g_u = 2*J_u + 1
    g_u = 2 * int(J_u) + 1
    
    # g_l : Poids statistique du niveau rovibrationnel inférieur (X) -> Sans unité
    # Source : Catalogue 'level_co.dat' (état fondamental X), colonne 'g'
    g_l_match = df_level_co[(df_level_co['v'] == int(v_l)) & (df_level_co['J'] == int(J_l))]
    g_l = float(g_l_match['g'].iloc[0]) if not g_l_match.empty else float(2 * int(J_l) + 1)
    
    # 4. RECHERCHE DE LA DENSITÉ DE POPULATION LOCALE (EXTRACTION IDAT)
    # Gestion des variations syntaxiques de l'extracteur IDAT (avec ou sans espace)
    col_sans_espace = f"n(COv={v_l},J={J_l})"
    col_avec_espace = f"n(CO v={v_l},J={J_l})"
    
    if col_sans_espace in df_zone.columns:
        col_target = col_sans_espace
    elif col_avec_espace in df_zone.columns:
        col_target = col_avec_espace
    else:
        print(f"❌ Erreur : Le niveau fondamental n(CO, v={v_l}, J={J_l}) est introuvable dans votre IDAT.")
        co_cols = [c for c in df_zone.columns if "n(CO" in c]
        print(f"ℹ️ Échantillon des colonnes CO disponibles pour vérification : {co_cols[:8]}")
        return None
    
    # n_l : Densité volumique locale des molécules dans le niveau (v_l, J_l) 
    # Source : Vecteur de données issu du fichier IDAT d'Orion -> Unité : cm^-3
    n_l = pd.to_numeric(df_zone[col_target], errors='coerce').fillna(0).values
    
    # distance_cm : Position absolue des mailles physiques de la PDR 
    # Source : Colonne 'Distance' du fichier IDAT d'Orion -> Unité : cm
    distance_cm = pd.to_numeric(df_zone["Distance"], errors='coerce').values
    
    # 5. INTEGRATION NUMÉRIQUE SPATIALE (CALCUL DE DsS ET N_l)
    # ds : Épaisseur physique réelle (largeur différentielle Delta_x) de chaque maille
    # Estimée par différence discrète successive. 'prepend' évite un ds nul au bord. -> Unité : cm
    ds = np.diff(distance_cm, prepend=distance_cm[0])
    
    # N_l : Densité de colonne cumulée des absorbants le long de la ligne de visée
    # Calculée par intégration discrète (Somme cumulée : intégrale de n_l * ds) -> Unité : cm^-2
    N_l = np.cumsum(n_l * ds)
    
    # 6. APPLICATION DU FORMALISME DE TRANSFERT DE DRAINE (2011)
    # prefactor : Facteur de conversion atomique de la raie -> Unité : cm^2
    prefactor = (A_ul * (c_light**3)) / (8.0 * np.pi * (nu**3) * delta_v_cgs)
    
    # tau_0 : Épaisseur optique finale cumulée au centre de la raie -> Sans unité
    # np.clip bloque mathématiquement les résidus numériques sous 0
    tau_0 = prefactor * N_l * (g_u / g_l)
    
    return np.clip(tau_0, a_min=0, a_max=None)






# =============================================================================
# À AJOUTER DANS VOTRE BIBLIOTHÈQUE pdr_profiles_lib.py POUR LE MULTI-MODÈLE
# =============================================================================
'''
def load_multiple_models(path_data, model_names, col_names):
    """
    Charge N modèles PDR simultanément dans un dictionnaire de DataFrames.
    Modèles indexés par leur nom de fichier pour une traçabilité parfaite.
    """
    dict_models = {}
    for name in model_names:
        full_path = os.path.join(path_data, name)
        if os.path.exists(full_path):
            # Lecture du fichier IDAT en sautant les en-têtes Meudon PDR
            df = pd.read_csv(full_path, sep=r'\s+', comment='#', names=col_names, skiprows=6)
            # Nettoyage uniforme des espaces cachés dans les colonnes
            df.columns = df.columns.str.replace(' ', '')
            dict_models[name] = df
            print(f"   --> Modèle [{name}] chargé avec succès ({len(df)} mailles).")
        else:
            print("Error Can't open the models")
    return dict_models
'''


def load_multiple_models(path_data, model_names, col_names):
    """
    Charge N modèles PDR simultanément dans un dictionnaire de DataFrames.
    Sécurisé contre les tuples accidentels et les écarts de colonnes.
    """
    # Sécurité anti-tuple : extrait le 1er élément si reçu sous forme de tuple
    if isinstance(path_data, (tuple, list)):
        path_data = path_data[0]

    dict_models = {}
    for item in model_names:
        name = item[0] if isinstance(item, (tuple, list)) else item
        full_path = os.path.join(str(path_data), str(name))

        if os.path.exists(full_path):
            try:
                # Lecture brute pour adapter dynamiquement le nombre de colonnes
                df = pd.read_csv(
                    full_path, 
                    sep=r'\s+', 
                    comment='#', 
                    skiprows=6, 
                    header=None
                )
                
                # Ajustement souple des noms de colonnes
                nb_cols_data = df.shape[1]
                if len(col_names) >= nb_cols_data:
                    df.columns = col_names[:nb_cols_data]
                else:
                    # Si le fichier a plus de colonnes que l'extracteur
                    extras = [f"Col_{i}" for i in range(len(col_names), nb_cols_data)]
                    df.columns = list(col_names) + extras

                df.columns = df.columns.str.replace(' ', '')
                dict_models[name] = df
                print(f"   --> Modèle [{name}] chargé avec succès ({len(df)} mailles, {df.shape[1]} colonnes).")
            except Exception as e:
                print(f"❌ Erreur lors de la lecture de [{name}] : {e}")
        else:
            print(f"❌ Fichier introuvable : {full_path}")

    return dict_models


def plot_specific_model_quad(dict_models, selected_model_key, species="CO", normalize=True, av_max=7.0):
    """
    Génère le Quad-Panel complet d'un modèle spécifique extrait du dictionnaire global.
    Rejoint l'ensemble des fonctionnalités graphiques validées précédemment.
    """
    if selected_model_key not in dict_models:
        raise KeyError(f"Le modèle {selected_model_key} n'est pas présent dans le dictionnaire.")
        
    df_selected = dict_models[selected_model_key]
    # On appelle la fonction de tracé flexible existante sur ce DataFrame isolé
    fig, axes = plot_flexible_quad_panel(df_selected, species=species, normalize=normalize, av_max=av_max)
    axes[0].set_title(f"Quad-Panel excitation — Model: {selected_model_key}", y=1.15, fontsize=14, fontweight='bold')
    return fig, axes



