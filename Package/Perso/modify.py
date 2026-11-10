#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:27:56 2025

@author: aurelien
"""
#% Definition des fonctions 

#% Balayage du df pour voir si la matrice est complete 
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize, LogNorm
import pandas as pd
from itertools import product
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import pandas as pd
import plotly.express as px
from plotly.offline import plot
import numpy as np

import pandas as pd
from itertools import product
import numpy as np

import re 

import pandas as pd
import plotly.express as px


from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.signal import find_peaks


def merge_levels_simple(df_coll: pd.DataFrame, df_level_co: pd.DataFrame, cols=["nu","nl"]) -> pd.DataFrame:
    """
    Merges collision DataFrame with level data to add upper and lower state quantum numbers (v, J).

    Args:
        df_coll: DataFrame with collision data ('nu', 'nl').
        df_level_co: DataFrame with level data ('n', 'v', 'J').

    Returns:
        A new DataFrame with 'vu', 'Ju', 'vl', 'Jl' columns added.
    """
    df_merged = df_coll.merge(
        df_level_co[['n', 'v', 'J']], 
        left_on=cols[0], 
        right_on='n', 
        how='left', 
        suffixes=('', '_u_temp')
    ).rename(columns={'v': 'vu_temp', 'J': 'Ju_temp'}) 
    
    df_merged.drop(columns=['n'], inplace=True)

    df_merged = df_merged.merge(
        df_level_co[['n', 'v', 'J']], 
        left_on=cols[1], 
        right_on='n', 
        how='left', 
        suffixes=('', '_l_temp')
    ).rename(columns={'v': 'vl_temp', 'J': 'Jl_temp'})

    df_merged.drop(columns=['n'], inplace=True)
    
    df_merged.rename(columns={
        'vu_temp': 'vu',
        'Ju_temp': 'Ju',
        'vl_temp': 'vl',
        'Jl_temp': 'Jl'
    }, inplace=True)
    
    for col in ['vu', 'Ju', 'vl', 'Jl']:
        df_merged[col] = df_merged[col].astype('int64') 

    return df_merged


def sort_and_create_effective_values(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the DataFrame and creates effective quantum number columns (V_eff_u, V_eff_l).

    The effective value is calculated as V_eff = v + J / J_max(v).

    Args:
        df_merged: The DataFrame resulting from the merge operation, containing 
                   'vu', 'Ju', 'vl', and 'Jl'.

    Returns:
        The sorted DataFrame with 'V_eff_u' and 'V_eff_l' columns added.
    """
    # 1. Sorting the DataFrame
    # Sort by vu, Ju, vl, and Jl in ascending order
    df_sorted = df_merged.sort_values(by=['vu', 'Ju', 'vl', 'Jl'], ascending=True)
    
    # 2. Preparation for Effective Value Calculation (J_max)
    
    # Calculate J_max for each unique 'vu' value.
    # Group by 'vu' and find the maximum 'Ju' in each group.
    j_max_u = df_sorted.groupby('vu')['Ju'].max().rename('Ju_max')
    
    # Calculate J_max for each unique 'vl' value.
    # Group by 'vl' and find the maximum 'Jl' in each group.
    j_max_l = df_sorted.groupby('vl')['Jl'].max().rename('Jl_max')
    
    # Join J_max values back to the main DataFrame for vectorized calculation
    # Join on 'vu' for upper state maximum J
    df_sorted = df_sorted.merge(j_max_u, left_on='vu', right_index=True, how='left')
    
    # Join on 'vl' for lower state maximum J
    df_sorted = df_sorted.merge(j_max_l, left_on='vl', right_index=True, how='left')

    # 3. Creating Effective Quantum Number Columns
    
    # Calculate V_eff_u = vu + Ju / Ju_max
    df_sorted['V_eff_u'] = df_sorted['vu'] + df_sorted['Ju'] / df_sorted['Ju_max']
    
    # Calculate V_eff_l = vl + Jl / Jl_max
    df_sorted['V_eff_l'] = df_sorted['vl'] + df_sorted['Jl'] / df_sorted['Jl_max']

    # 4. Cleanup and Return
    # Drop the temporary J_max columns
    df_result = df_sorted.drop(columns=['Ju_max', 'Jl_max'])
    
    return df_result

def plot_static_colormap(df: pd.DataFrame, 
                         x_col: str = 'V_eff_l', 
                         y_col: str = 'V_eff_u', 
                         c_col: str = '100.0', 
                         cmap_name: str = 'viridis',
                         log_color_scale: bool = False,
                         title="Titre") -> None: # Added new argument
    """
    Generates a static scatter plot of V_eff_u vs V_eff_l with a colormap 
    based on a specified column (normalized, optionally logarithmic).

    Args:
        df: The input DataFrame.
        x_col: Name of the column for the x-axis (default: 'V_eff_l').
        y_col: Name of the column for the y-axis (default: 'V_eff_u').
        c_col: Name of the column for the colormap data (default: '100.0').
        cmap_name: Name of the matplotlib colormap to use (default: 'viridis').
        log_color_scale: If True, uses a logarithmic scale for the colormap (default: False).
    """
    # 1. Normalize the color data (c_col)
    c_data = df[c_col].values
    
    # Determine the normalization based on the log_color_scale argument
    if log_color_scale:
        # Check that all values are positive for the log scale
        if np.min(c_data) <= 0:
            print("Warning: Cannot use logarithmic color scale. Data must be strictly positive.")
            # Fallback to linear scale if log scale is requested but data isn't suitable
            norm = Normalize(vmin=np.min(c_data), vmax=np.max(c_data))
        else:
            norm = LogNorm(vmin=np.min(c_data), vmax=np.max(c_data))
    else:
        norm = Normalize(vmin=np.min(c_data), vmax=np.max(c_data))
        
    # 2. Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot with color mapping
    scatter = ax.scatter(
        df[x_col], 
        df[y_col], 
        c=c_data, 
        cmap=cmap_name, 
        norm=norm, # Use the chosen normalization
        s=20, # Marker size
        alpha=0.8 # Transparency
    )
    
    # 3. Set labels and title
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    
    # 4. Add color bar
    cbar = fig.colorbar(scatter, ax=ax)
    
    # Update color bar label based on normalization
    label_suffix = " (log scale)" if log_color_scale and isinstance(norm, LogNorm) else " (normalized)"
    cbar.set_label(f'Collision rate at {c_col}K{label_suffix}')
    
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    


def get_missing_levels_df(df_merged: pd.DataFrame, c_col: str = '100.0') -> pd.DataFrame:
    """
    Generates a DataFrame of missing (vu, Ju, vl, Jl) combinations based on the observed ranges.

    Args:
        df_merged: The DataFrame containing all existing data, including the color column (c_col).
        c_col: The column used to identify existing data (e.g., '100.0').

    Returns:
        A sorted DataFrame containing only the (vu, Ju, vl, Jl) combinations that are missing.
    """
    
    # 1. Identify the Universe of Possible Points (Combinations)
    unique_vu = df_merged['vu'].unique()
    unique_Ju = df_merged['Ju'].unique()
    unique_vl = df_merged['vl'].unique()
    unique_Jl = df_merged['Jl'].unique()
    
    # Create a DataFrame of ALL possible combinations
    all_comb_tuples = list(product(unique_vu, unique_Ju, unique_vl, unique_Jl))
    df_universe = pd.DataFrame(all_comb_tuples, columns=['vu', 'Ju', 'vl', 'Jl'])
    
    # 2. Merge the existing data onto the universe to flag existing rows
    # Select only the key columns for merging
    df_combined = df_universe.merge(
        df_merged[['vu', 'Ju', 'vl', 'Jl', c_col]].drop_duplicates(), 
        on=['vu', 'Ju', 'vl', 'Jl'], 
        how='left'
    )
    
    # 3. Identify Missing Points
    # Missing points are those where the color column (c_col) is NaN
    df_missing = df_combined[df_combined[c_col].isna()].copy()
    
    # 4. Cleanup and Return
    # Keep only the quantum number columns
    df_missing_levels = df_missing[['vu', 'Ju', 'vl', 'Jl']]
    
    # Sort the result for readability
    df_missing_levels.sort_values(by=['vu', 'Ju', 'vl', 'Jl'], inplace=True)
    
    return df_missing_levels



# --- Helper Function for V_eff calculation (Reused) ---
def calculate_v_eff(df: pd.DataFrame, prefix: str, j_max_series: pd.Series) -> pd.DataFrame:
    """Calculates V_eff = v + J / J_max for the given prefix ('u' or 'l')."""
    v_col = f'v{prefix}'
    j_col = f'J{prefix}'
    j_max_col = f'J_max_{prefix}' # Unique temporary column name

    # 1. Merge J_max values
    df = df.merge(
        j_max_series.rename(j_max_col), 
        left_on=v_col, 
        right_index=True, 
        how='left'
    )
    
    # 2. Calculate V_eff
    df[f'V_eff_{prefix}'] = df[v_col] + df[j_col] / df[j_max_col]
    
    # 3. Clean up
    df.drop(columns=[j_max_col], inplace=True)
    return df

def get_missing_levels_df_2(df_merged: pd.DataFrame, c_col: str = '100.0') -> pd.DataFrame:
    """
    Generates a DataFrame of missing (vu, Ju, vl, Jl) combinations, 
    including all other columns from df_merged, applying V_eff_u > V_eff_l constraint.
    
    Args:
        df_merged: The DataFrame containing all existing data and V_eff columns.
        c_col: The column used to identify existing data (e.g., '100.0').

    Returns:
        A sorted DataFrame containing the missing combinations with all columns set to NaN 
        for the collision data.
    """
    # 1. Determine J_max for V_eff calculation
    j_max_u = df_merged.groupby('vu')['Ju'].max()
    j_max_l = df_merged.groupby('vl')['Jl'].max()

    # 2. Identify the Universe of all OBSERVED quantum numbers (v, J)
    levels_u = df_merged[['vu', 'Ju']].drop_duplicates()
    levels_l = df_merged[['vl', 'Jl']].drop_duplicates()
    
    levels_u.columns = ['v', 'J']
    levels_l.columns = ['v', 'J']
    
    # Create the complete universe by crossing all observed upper and lower pairs
    universe_tuples = list(product(levels_u.itertuples(index=False), levels_l.itertuples(index=False)))
    df_universe = pd.DataFrame([(*u, *l) for u, l in universe_tuples], columns=['vu', 'Ju', 'vl', 'Jl'])
    df_universe.drop_duplicates(inplace=True)

    # 3. Calculate V_eff for the entire universe
    df_universe = calculate_v_eff(df_universe, 'u', j_max_u)
    df_universe = calculate_v_eff(df_universe, 'l', j_max_l)
    
    # 4. Apply Physical Constraint: V_eff_u > V_eff_l
    df_universe_physical = df_universe[df_universe['V_eff_u'] > df_universe['V_eff_l']].copy()
    
    # 5. Identify Missing Points
    key_cols = ['vu', 'Ju', 'vl', 'Jl']
    
    # Columns to include from df_merged (all of them)
    all_cols = df_merged.columns.tolist()
    
    # Tag existing rows: we include ALL columns from df_merged
    df_existing_key = df_merged[all_cols].drop_duplicates(subset=key_cols)
    
    # Merge the universe with the existing data
    df_combined = df_universe_physical.merge(
        df_existing_key, 
        on=key_cols, 
        how='left',
        suffixes=('_uni', '_orig')
    )
    
    # Harmonize V_eff columns: use the universe's V_eff (correctly calculated) 
    # and drop the V_eff from the original data (which might be duplicated/conflicting post-merge)
    v_eff_cols = ['V_eff_u', 'V_eff_l']
    for col in v_eff_cols:
        df_combined.drop(columns=[f'{col}_orig'], inplace=True, errors='ignore')
        df_combined.rename(columns={f'{col}_uni': col}, inplace=True)


    # Filter for missing points (where the data column c_col is NaN)
    df_missing = df_combined[df_combined[c_col].isna()].copy()
    
    # 6. Final Cleanup and Sort
    
    # The columns that were missing in the merge (e.g., '100.0_orig') are already NaN, 
    # but we need to rename them to the original name and keep only necessary columns.
    
    # Identify columns that came from the original data and drop the suffix '_orig'
    # The columns that have the suffix '_orig' are the ones that carry the actual collision data.
    # When merged left, these will be NaN for missing points.
    cols_to_keep = [col for col in df_combined.columns if not col.endswith('_uni')]
    df_missing = df_missing[cols_to_keep].copy()
    
    # Ensure the V_eff columns are properly named
    for col in v_eff_cols:
        if f'{col}_orig' in df_missing.columns:
             df_missing.rename(columns={f'{col}_orig': col}, inplace=True)
             
    df_missing.sort_values(by=key_cols, inplace=True)
    
    return df_missing



def plot_interactive_with_missing(df_merged: pd.DataFrame, 
                                  x_col: str = 'V_eff_l', 
                                  y_col: str = 'V_eff_u', 
                                  c_col: str = '100.0') -> None:
    """
    Generates an interactive Plotly scatter plot, highlighting the missing possible points 
    and maintaining hover functionality for both existing and missing data.

    Args:
        df_merged: The DataFrame containing all existing data.
        x_col: Name of the column for the x-axis (default: 'V_eff_l').
        y_col: Name of the column for the y-axis (default: 'V_eff_u').
        c_col: Name of the column for the colormap data (default: '100.0').
    """
    
    # 1. Prepare Missing Data (using the helper function)
    df_missing_levels = get_missing_levels_df(df_merged, c_col)
    
    # Recalculate V_eff for the missing levels (requires J_max values)
    # Replicate the J_max calculation logic from the 'sort_and_create_effective_values' function
    j_max_u = df_merged.groupby('vu')['Ju'].max()
    j_max_l = df_merged.groupby('vl')['Jl'].max()
    
    # Merge J_max values back to the missing DataFrame
    df_missing = df_missing_levels.merge(
        j_max_u.rename('Ju_max'), left_on='vu', right_index=True, how='left'
    )
    df_missing = df_missing.merge(
        j_max_l.rename('Jl_max'), left_on='vl', right_index=True, how='left'
    )
    
    # Calculate V_eff for missing points
    df_missing[y_col] = df_missing['vu'] + df_missing['Ju'] / df_missing['Ju_max']
    df_missing[x_col] = df_missing['vl'] + df_missing['Jl'] / df_missing['Jl_max']
    
    # Columns to show in the hover tooltip
    hover_cols = ['vu', 'Ju', 'vl', 'Jl','UP','LOW']

    # 2. Create Plotly figure for Missing Points (Background)
    fig_missing = px.scatter(
        df_missing,
        x=x_col,
        y=y_col,
        hover_data=hover_cols,
        color_discrete_sequence=['lightgray'], # Uniform color
        title=f'Interactive Plot: {y_col} vs {x_col} (Missing and Existing Data)'
    )
    
    # Customize missing points appearance (transparent)
    fig_missing.update_traces(
        marker=dict(size=8, opacity=0.15, symbol='circle'), 
        name='Missing Data',
        showlegend=True
    )

    # 3. Create Plotly figure for Existing Data (Foreground)
    fig_existing = px.scatter(
        df_merged, 
        x=x_col, 
        y=y_col, 
        color=c_col, 
        hover_data=hover_cols,
    )
    
    # Customize existing points appearance
    fig_existing.update_traces(
        marker=dict(size=8, opacity=0.8), 
        name='Existing Data',
        showlegend=True
    )
    
    # 4. Combine the two figures
    # Add the existing data trace to the missing data figure
    for trace in fig_existing.data:
        fig_missing.add_trace(trace)

    # Show the combined plot
    fig_missing.show()


def plot_interactive_plotly(df: pd.DataFrame, 
                            x_col: str = 'V_eff_l', 
                            y_col: str = 'V_eff_u', 
                            c_col: str = '100.0',
                            log_color_scale: bool = False,
                            hover_cols = ['vu', 'Ju', 'vl', 'Jl'],
                            title="Title"
) -> None:
    """
    Generates an interactive scatter plot of V_eff_u vs V_eff_l using Plotly.

    Args:
        df: The input DataFrame.
        x_col: Name of the column for the x-axis (default: 'V_eff_l').
        y_col: Name of the column for the y-axis (default: 'V_eff_u').
        c_col: Name of the column for the colormap data (default: '100.0').
        log_color_scale: If True, uses a logarithmic scale for the color axis (default: False).
    """
    #hover_cols = ['vu', 'Ju', 'vl', 'Jl']
    
    c_data = df[c_col].values
    
    # --- 1. Gestion de l'échelle logarithmique ---
    if log_color_scale:
        min_c_data = c_data.min()
        if min_c_data <= 0:
            print(f"Warning: Cannot use logarithmic color scale for '{c_col}' because minimum value ({min_c_data}) is zero or negative. Falling back to linear scale.")
            log_color_scale = False
        else:
            # Créer des bornes logarithmiques pour la barre de couleur Plotly
            # Plotly utilise des bornes linéaires par défaut, nous devons donc fournir
            # les bornes de couleur VRAIES (log) et les Ticks (non-log)
            cmin = np.log10(c_data.min())
            cmax = np.log10(c_data.max())
            
            # Utiliser log10(data) pour la coloration
            df['c_data_log'] = np.log10(c_data)
            color_col_name = 'c_data_log'
            
            # Définir les ticks pour la barre de couleur (ex: 10^1, 10^2, ...)
            log_ticks = np.arange(np.floor(cmin), np.ceil(cmax) + 1)
            tick_vals = [10**i for i in log_ticks]
            tick_text = [f"10^{i}" for i in log_ticks]
            
    else:
        color_col_name = c_col # Utiliser les données originales

    # --- 2. Création du graphique Plotly Express ---
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col, 
        color=color_col_name, # Utilise c_data_log ou c_col
        hover_data=hover_cols, 
        title=title,
        labels={color_col_name: f'Value of Column "{c_col}"' + (' (log scale)' if log_color_scale else '')} 
        # Plotly gère color_continuous_scale pour la colormap
    )
    
    # --- 3. Personnalisation de la barre de couleur (si log) ---
    if log_color_scale:
        # Mettre à jour la barre de couleur pour utiliser les étiquettes logarithmiques
        fig.update_traces(
            marker=dict(
                colorbar=dict(
                    title=f'Value of Column "{c_col}" (log scale)',
                    tickvals=log_ticks, # Position des ticks sur l'axe log
                    ticktext=tick_text  # Texte à afficher (10^X)
                )
            )
        )
    
    # Personnalisation
    fig.update_traces(marker=dict(size=8, opacity=0.8))

    fig.show()
    
    
def plot_interactive_combined(df_merged: pd.DataFrame, 
                              df_missing: pd.DataFrame,
                              x_col: str = 'V_eff_l', 
                              y_col: str = 'V_eff_u', 
                              c_col: str = '100.0') -> None:
    """
    Génère un graphique Plotly interactif combinant les données existantes 
    (avec colormap) et les niveaux manquants (en noir transparent).

    Args:
        df_merged: Le DataFrame contenant les données existantes (pour la couleur).
        df_missing: Le DataFrame des niveaux manquants (sans données de collision, en noir).
        x_col: Colonne pour l'axe x (défaut: 'V_eff_l').
        y_col: Colonne pour l'axe y (défaut: 'V_eff_u').
        c_col: Colonne de la colormap pour les points existants (défaut: '100.0').
    """
    
    # Séparer les données existantes pour le tracé (en appliquant la contrainte physique)
    df_existing = df_merged[df_merged[y_col] > df_merged[x_col]].copy()
    
    hover_cols = ['vu', 'Ju', 'vl', 'Jl']

    # 1. Tracé des points manquants (Noir, Transparent, Arrière-plan)
    # Nous commençons avec un go.Figure pour plus de contrôle sur les traces
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_missing[x_col],
            y=df_missing[y_col],
            mode='markers',
            name='Niveaux Manquants',
            hoverinfo='text',
            hovertext=[f"vu: {r.vu}<br>Ju: {r.Ju}<br>vl: {r.vl}<br>Jl: {r.Jl}<br>Statut: Manquant" 
                       for r in df_missing.itertuples()],
            marker=dict(
                size=12, 
                opacity=0.10, # Très transparent
                color='black', 
                symbol='circle'
            )
        )
    )

    # 2. Tracé des points existants (Colormap, Avant-plan)
    
    # Définir l'échelle de couleur (Plotly gère la normalisation automatiquement)
    color_scale = px.colors.sequential.Viridis
    
    fig.add_trace(
        go.Scatter(
            x=df_existing[x_col],
            y=df_existing[y_col],
            mode='markers',
            name='Données Existantes',
            hoverinfo='all',
            # Colormap basée sur la colonne c_col
            marker=dict(
                size=8,
                opacity=0.8,
                color=df_existing[c_col], 
                colorscale=color_scale,
                colorbar=dict(title=f'Valeur de {c_col}'),
                showscale=True
            )
        )
    )

    # 3. Mise en page finale
    fig.update_layout(
        title=f'Tracé Interactif Combiné: {y_col} vs {x_col}',
        xaxis_title=x_col,
        yaxis_title=y_col,
        # Ajuster pour un meilleur rendu visuel
        hovermode='closest' 
    )
    
    fig.show()
    
    

def get_lines_intensity(file_path, molecule):
    """
    Extrait les intensités inta_00 et les nombres quantiques (v, J) 
    pour une molécule donnée depuis un fichier .stat.
    """
    data = []
    
    # Préfixe à rechercher (ex: "inta_00_co_")
    prefix = f"inta_00_{molecule}_"
    
    # Regex pour capturer les quantiques: v(upper)_j(upper) ... v(lower)_j(lower)
    # (\d+) capture une suite de chiffres
    # .*? ignore ce qu'il y a au milieu (le séparateur, souvent "__")
    pattern_quantum = re.compile(r'v(\d+)_j(\d+).*?v(\d+)_j(\d+)')

    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith('value'):
                    continue
                
                parts = [p.strip() for p in line.split('#')]
                
                if len(parts) < 3:
                    continue
                    
                label = parts[1]
                
                # Si la ligne correspond à notre molécule
                if label.startswith(prefix):
                    # On récupère juste la partie transition (ex: v1_j15__v0_j14)
                    transition_str = label[len(prefix):]
                    
                    try:
                        intensity = float(parts[2])
                        unit = parts[3] if len(parts) > 3 else "unknown"
                        
                        # --- Extraction des Nombres Quantiques ---
                        match = pattern_quantum.search(transition_str)
                        if match:
                            # On convertit les groupes capturés en entiers
                            vu, ju, vl, jl = map(int, match.groups())
                        else:
                            # Si le format n'est pas respecté (ex: raie atomique sans v/j)
                            vu, ju, vl, jl = None, None, None, None

                        data.append({
                            'molecule': molecule,
                            'transition': transition_str,
                            'intensity': intensity,
                            'unit': unit,
                            'vu': vu, # Vibration Supérieure
                            'Ju': ju, # Rotation Supérieure
                            'vl': vl, # Vibration Inférieure
                            'Jl': jl  # Rotation Inférieure
                        })
                        
                    except ValueError:
                        continue

    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} est introuvable.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    
    # Optionnel : Convertir les colonnes quantiques en entiers (Nullable Int64)
    # Cela permet de garder des entiers même s'il y a des NaN/None ailleurs
    cols_quantiques = ['vu', 'Ju', 'vl', 'Jl']
    if not df.empty:
        for col in cols_quantiques:
            df[col] = df[col].astype('Int64')

    return df




def parse_spectro_label(label):
    """
    Parse une chaîne au format 'vu-vl_Branche( J )' (ex: '1-0_R( 6)')
    et retourne vu, Ju, vl, Jl.
    """
    # Regex expliquée :
    # (\d+)     -> Capture vu
    # -         -> Séparateur littéral
    # (\d+)     -> Capture vl
    # _         -> Séparateur littéral
    # ([A-Z])   -> Capture la lettre de la branche (P, Q, R, S, O)
    # \(\s* -> Parenthèse ouvrante et espaces optionnels
    # (\d+)     -> Capture le J de référence (généralement J_lower)
    # \s*\)     -> Espaces optionnels et parenthèse fermante
    pattern = re.compile(r"(\d+)-(\d+)_([A-Z])\(\s*(\d+)\s*\)")
    
    match = pattern.match(label)
    if match:
        vu_raw, vl_raw, branch, j_ref_raw = match.groups()
        
        vu = int(vu_raw)
        vl = int(vl_raw)
        j_ref = int(j_ref_raw)
        
        # En notation standard P,Q,R, le nombre entre parenthèses est J_lower (Jl)
        jl = j_ref
        ju = None
        
        if branch == 'R':   # Ju - Jl = +1
            ju = jl + 1
        elif branch == 'P': # Ju - Jl = -1
            ju = jl - 1
        elif branch == 'Q': # Ju - Jl = 0
            ju = jl
        elif branch == 'S': # Ju - Jl = +2 (ex: H2)
            ju = jl + 2
        elif branch == 'O': # Ju - Jl = -2 (ex: H2)
            ju = jl - 2
            
        return pd.Series([vu, ju, vl, jl], index=['vu', 'Ju', 'vl', 'Jl'])
    else:
        return pd.Series([None, None, None, None], index=['vu', 'Ju', 'vl', 'Jl'])



def add_freq_and_lambda(df):
    # Extraction de la fréquence en GHz via regex
    def extract_ghz(desc):
        match = re.search(r'([\d\.]+)\s*GHz', str(desc))
        return float(match.group(1)) if match else None
    
    df['freq_GHz'] = df['Description'].apply(extract_ghz)
    
    # Calcul de la longueur d'onde en mm : lambda (mm) = c (mm/s) / f (Hz)
    c = 299792458  # m/s vitesse de la lumière
    def freq_to_lambda(freq_ghz):
        return c / (freq_ghz * 1e9)  if freq_ghz else None  # en m 
    
    df['lambda_m'] = df['freq_GHz'].apply(freq_to_lambda)
    return df


def put_annotation (ax,df,color="darkred",Wave_col ="wavelength", Line_col="Line") : 
    column_name = Wave_col
    #Add the identification lines from the Meudon PDR code 
    for x_val, label in zip(df[column_name], df[Line_col]):
        ax.axvline(
            x=x_val,
            color=color,          
            linestyle='--',
            linewidth=1,
            alpha=0.5,
            label='Lignes du DF' if x_val == df[column_name].iloc[0] else None)
        
        ax.text(
            x=x_val,
            y=ax.get_ylim()[1] * 0.95, 
            s=f'{label}',              
            rotation=90,
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=10,
            color=color)
        
    
def get_j_upper(row):
    """Calcule J_upper basé sur le nom de la ligne (Notation Herzberg)"""
    try:
        name = row['Line']
        # Format attendu : "1-0_P( 4)" ou "2-1_R( 10)"
        parts = name.split('_')
        band = parts[0]     # "1-0"
        branch_part = parts[1] # "P( 4)"
        
        branch = branch_part[0] # "P" ou "R"
        # Extraction du nombre entre parenthèses
        j_lower = int(branch_part.split('(')[1].split(')')[0])
        
        # Règle spectroscopique pour trouver J_upper
        if branch == 'P':
            j_upper = j_lower - 1
        elif branch == 'R':
            j_upper = j_lower + 1
        else:
            return None, None, None
            
        return band, j_upper, branch
    except:
        return None, None, None
    
    

def gaussian_with_offset(x, amplitude, center, sigma, offset):
    """
    Modèle Gaussien simple + offset constant.
    Amplitude représente la hauteur du pic au-dessus de l'offset.
    """
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2)) + offset

def voigt_with_offset(x, amplitude, center, sigma, gamma, offset):
    """
    Modèle de Voigt + offset constant.
    Amplitude est un facteur d'échelle (relié à l'aire).
    """
    return amplitude * voigt_profile(x - center, sigma, gamma) + offset

# --- B. Routine de Fitting Générique ---

def fit_spectral_line(
    df_spectrum, 
    center_wavelength, 
    window_width=0.025, 
    model_type='gaussian',
    col_wave='Wavelength',
    col_int='Int',
    center_tol=0.005,         # Tolérance élargie (votre pic est un peu décalé)
    offset_tol=0.50,          # On autorise le fond à varier beaucoup
    sigma_bounds=(1e-4, 0.01),
    detection_sigma=0.3,      # Seuil de détection très bas
    edge_fraction=0.25
):
    
    # 1. Extraction locale
    mask = (df_spectrum[col_wave] >= center_wavelength - window_width) & \
           (df_spectrum[col_wave] <= center_wavelength + window_width)
    local_df = df_spectrum[mask]
    
    if len(local_df) < 5: return None
    
    x = local_df[col_wave].values
    y = local_df[col_int].values
    
    # --- CHANGEMENT MAJEUR : Stratégie "Low Percentile" ---
    # Au lieu de regarder les bords (qui sont hauts sur votre image),
    # on cherche le "sol" de la fenêtre courante.
    # On prend le 10ème percentile (pour éviter un pixel mort unique qui serait min(y))
    guess_offset = np.percentile(y, 10) 
    
    # Estimation du bruit (écart type des résidus par rapport à une moyenne glissante ou simple std)
    noise_est = np.std(y - np.poly1d(np.polyfit(x, y, 1))(x)) # Bruit autour d'une pente linéaire
    if noise_est == 0: noise_est = 1.0

    # 3. Détection du Pic
    # Le seuil est bas : juste un peu au-dessus du sol local
    threshold = guess_offset + (detection_sigma * noise_est)
    
    # Prominence : Le pic doit juste ressortir localement
    peaks_idx, _ = find_peaks(y, height=threshold, prominence=noise_est*0.5)
    
    guess_center = center_wavelength
    guess_amp = np.max(y) - guess_offset

    if len(peaks_idx) > 0:
        dists = np.abs(x[peaks_idx] - center_wavelength)
        valid_peaks_mask = dists <= center_tol
        if np.any(valid_peaks_mask):
            valid_peaks = peaks_idx[valid_peaks_mask]
            best_idx = valid_peaks[np.argmin(dists[valid_peaks_mask])]
            guess_center = x[best_idx]
            guess_amp = y[best_idx] - guess_offset

    # Valeur par défaut pour sigma
    guess_sigma = 0.001 

    try:
        if model_type == 'gaussian':
            p0 = [guess_amp, guess_center, guess_sigma, guess_offset]
            
            # --- BORNES ---
            # On autorise l'offset à descendre jusqu'au min absolu des données
            # et à monter jusqu'au max (cas d'absorption ou bruit bizarre)
            min_y_window = np.min(y)
            max_y_window = np.max(y)
            
            bounds = (
                # Amp, Center, Sigma, Offset
                [0,       center_wavelength - center_tol,  sigma_bounds[0],  min_y_window * 0.5], 
                [np.inf,  center_wavelength + center_tol,  sigma_bounds[1],  max_y_window]
            )
            
            popt, pcov = curve_fit(gaussian_with_offset, x, y, p0=p0, bounds=bounds, maxfev=10000)
            
            amp_fit, center_fit, sigma_fit, offset_fit = popt
            fwhm = 2.355 * sigma_fit
            fit_func = gaussian_with_offset

        # Calcul SNR basé sur le bruit résiduel
        residuals = y - fit_func(x, *popt)
        noise_std = np.std(residuals)
        
        # Si le fit est parfait (noise_std=0), SNR infini
        snr = amp_fit / noise_std if noise_std > 1e-9 else 100
        
        return {
            'success': True,
            'popt': popt,
            'center': center_fit,
            'shift': center_fit - center_wavelength,
            'fwhm': fwhm,
            'snr': snr,
            'amplitude': amp_fit,
            'offset': offset_fit,
            'model_type': model_type,
            'x_local': x,
            'y_local': y,
            'fit_func': fit_func
        }

    except Exception:
        return None
    
    
def plot_spectrum_with_fits(
    df_spectrum, 
    df_lines, 
    fit_results_list,
    figsize=(25, 8),
    xlim=None,
    ylim=None,
    title="Spectral Fit Analysis"
):
    """
    Trace le spectre complet et superpose les fits réussis avec le nom de la raie.
    """
    # Assurez-vous d'importer votre module viz contenant put_annotation
    # import viz 

    fig, ax = plt.subplots(figsize=figsize)
    
    # 1. Fond du spectre (données brutes)
    ax.plot(df_spectrum["Wavelength"], df_spectrum["Int"], "k-", alpha=0.3, lw=1, label='Data')
    
    # 2. Gestion des limites
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    
    # 3. Boucle sur les résultats de fit
    colors = ['crimson', 'royalblue', 'forestgreen', 'darkorange', 'purple']
    
    print(f"{'Line Name':<20} | {'Center':<10} | {'Shift (nm)':<10} | {'FWHM (nm)':<10} | {'SNR':<5}")
    print("-" * 65)

    for i, res in enumerate(fit_results_list):
        if res is None or res['snr'] < 3: continue 
        
        c = colors[i % len(colors)]
        
        # Recréation de la courbe fitée
        x_plot = np.linspace(res['x_local'].min(), res['x_local'].max(), 200)
        y_plot = res['fit_func'](x_plot, *res['popt'])
        
        # Plot du fit
        ax.plot(x_plot, y_plot, color=c, lw=2, linestyle='-')
        ax.fill_between(x_plot, res['offset'], y_plot, color=c, alpha=0.1)
        
        # --- MODIFICATION ICI : Ajout du nom ---
        line_name = res.get('name', f"Line_{i}")
        label_text = f"{line_name}\n$\Delta${res['shift']*1000:.1f}nm"
        
        ax.text(
            res['center'], 
            res['offset'] + res['amplitude'] + (res['amplitude']*0.05),
            label_text, 
            color=c, fontsize=8, ha='center', va='bottom', rotation=90
        )
        
        # Print tableau console
        print(f"{line_name:<20} | {res['center']:.5f}     | {res['shift']*1000:+.3f}      | {res['fwhm']*1000:.3f}      | {res['snr']:.1f}")

    # 4. Annotations théoriques
    # viz.put_annotation(ax, df_lines, color="darkblue", col_wave='wavelength', col_name="Line")

    # Cosmétique finale
    ax.set_xlabel("Wavelength [$\mu m$]", fontsize=14)
    ax.set_ylabel("Surface Brightness [MJy/sr]", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def analyze_velocity_distribution(fit_results_list, expected_v=None, tolerance=20):
    """
    Analyse la cohérence cinématique des raies détectées.
    
    Args:
        fit_results_list (list): La liste des dictionnaires retournés par fit_spectral_line.
        expected_v (float): Vitesse attendue du nuage (si connue).
        tolerance (float): Tolérance autour de la moyenne pour considérer un fit comme "bon".
    """
    c_light = 299792.458 # Vitesse lumière en km/s
    
    velocities = []
    names = []
    snrs = []
    
    # 1. Extraction des données
    for res in fit_results_list:
        if res is None: continue
        
        # Calcul Vitesse : v = c * (delta_lambda / lambda_theo)
        # res['shift'] est déjà (obs - theo)
        lambda_theo = res['center'] - res['shift'] 
        v = c_light * (res['shift'] / lambda_theo)
        
        velocities.append(v)
        names.append(res.get('name', 'Unknown'))
        snrs.append(res['snr'])
        
    if not velocities:
        print("Aucune raie détectée pour l'analyse de vitesse.")
        return

    # Convertir en array pour faciliter les calculs
    velocities = np.array(velocities)
    snrs = np.array(snrs)
    
    # 2. Statistiques Robustes (Médiane pour ignorer les outliers fous)
    median_v = np.median(velocities)
    std_v = np.std(velocities)
    
    # Identification des Outliers (vitesse trop loin de la médiane)
    # On considère outlier ce qui est à plus de 2 écarts-types ou d'une tolérance fixe
    is_outlier = np.abs(velocities - median_v) > max(2*std_v, tolerance)
    
    # 3. Affichage Graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Histogramme des vitesses ---
    ax1.hist(velocities, bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='Toutes les raies')
    ax1.axvline(median_v, color='crimson', linestyle='--', linewidth=2, label=f'Médiane: {median_v:.1f} km/s')
    
    if expected_v is not None:
        ax1.axvline(expected_v, color='green', linestyle='-', label=f'Attendu: {expected_v} km/s')

    ax1.set_xlabel('Vitesse Radiale (km/s)', fontsize=14)
    ax1.set_ylabel('Nombre de raies', fontsize=14)
    ax1.set_title('Distribution des Vitesses', fontsize=16)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # --- Scatter Plot : SNR vs Vitesse ---
    # Permet de voir si les raies fortes (haut SNR) sont bien groupées
    ax2.scatter(velocities[~is_outlier], snrs[~is_outlier], c='forestgreen', label='Cohérent')
    ax2.scatter(velocities[is_outlier], snrs[is_outlier], c='crimson', marker='x', label='Outlier (Vitesse suspecte)')
    
    ax2.axvline(median_v, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Vitesse Radiale (km/s)', fontsize=14)
    ax2.set_ylabel('SNR', fontsize=14)
    ax2.set_title('Qualité (SNR) vs Cinématique', fontsize=16)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 4. Rapport Texte
    print(f"\n=== ANALYSE CINÉMATIQUE ===")
    print(f"Nombre de raies étudiées : {len(velocities)}")
    print(f"Vitesse Médiane du nuage : {median_v:.2f} km/s")
    print(f"Dispersion (std)         : {std_v:.2f} km/s")
    print("-" * 30)
    print("Raies probablement fausses (Outliers de vitesse) :")
    
    bad_indices = np.where(is_outlier)[0]
    if len(bad_indices) == 0:
        print("Aucune ! Vos fits semblent très cohérents.")
    else:
        for idx in bad_indices:
            print(f"  - {names[idx]:<15} : {velocities[idx]:.1f} km/s (SNR={snrs[idx]:.1f})")
            
    return median_v, std_v