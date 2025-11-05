#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 17:03:40 2025

@author: aurelien
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# --------------- Library-style utility functions, explicit version ------------
################################################################################

def read_collision_to_df(filepath):
    """
    Reads a .dat collision rate file and returns a DataFrame.
    Assumes columns: NUM, UP, LOW, then temperature columns (e.g., 2KPDR).
    """
    dfdata = []
    collision_temperatures = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('!COLLISION TEMPERATURES '):
            collision_temperatures = [float(x) for x in lines[lines.index(line)+1].split()]
    rate_columns = [f"{int(t)}Ke" for t in collision_temperatures]
    all_columns = ['NUM', 'UP', 'LOW'] + rate_columns
    for line in lines:
        parts = line.split()
        if len(parts) == len(all_columns):
            nums = list(map(int, parts[:3]))
            rates = list(map(float, parts[3:]))
            dfdata.append(nums + rates)
    df = pd.DataFrame(dfdata, columns=all_columns)
    return df

def read_level_file(filepath, startline, columns, endline=None):
    """
    Reads a level file with columns [n,g,EnK,o,v,J,comment], returns a DataFrame.
    """
    lines = []
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= startline and (endline is None or idx < endline):
                parts = line.strip().split()
                if len(parts) >= len(columns):
                    lines.append(parts[:len(columns)])
    df = pd.DataFrame(lines, columns=columns)
    # Type conversion for int/float columns
    for col in ['n', 'g', 'v', 'J']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def merge_level_data(dftransitions, dflevel, Coltransi, Leveltransi):
    """
    Merge a transitions DF with a levels DF to add v,J quantum numbers.
    """
    # merge for Nup
    dflevelforNup = dflevel[['n', 'v', 'J']].rename(columns={'n': Coltransi[0], 'v': 'vu', 'J': 'Ju'})
    dfmerged = pd.merge(dftransitions, dflevelforNup, on=Coltransi[0], how='left')
    # merge for Nlow
    dflevelforNlow = dflevel[['n', 'v', 'J']].rename(columns={'n': Coltransi[1], 'v': 'vl', 'J': 'Jl'})
    dffinal = pd.merge(dfmerged, dflevelforNlow, on=Coltransi[1], how='left')
    return dffinal

def rename_meudon_columns(df):
    """Rename Meudon columns to Stancil format."""
    mapping = {'NUM': 'ID', 'UP': 'nu', 'LOW': 'nl'}
    for col in df.columns:
        if col.endswith('PDR'):
            mapping[col] = col.replace('KPDR', 'Ke')
    return df.rename(columns=mapping)

def update_nu_nl_from_level(df, df_levelco):
    """Update nu and nl columns based on v, J from df_levelco."""
    def find_n(v, J):
        match = df_levelco[(df_levelco['v'] == v) & (df_levelco['J'] == J)]
        if not match.empty:
            return match['n'].iloc[0]
        return None
    for idx, row in df.iterrows():
        nu = find_n(row['vu'], row['Ju'])
        nl = find_n(row['vl'], row['Jl'])
        if nu is not None:
            df.at[idx, 'nu'] = nu
        if nl is not None:
            df.at[idx, 'nl'] = nl
    return df

def merge_out_pdr(df1, df2):
    """Merge df1 and unique lines from df2 based on ['vu','Ju','vl','Jl'].""" 
    keys = ['vu', 'Ju', 'vl', 'Jl']
    temp = pd.merge(df1, df2, on=keys, how='outer', indicator=True, suffixes=('_df1', '_df2'))
    unique_df2 = temp[temp['_merge']=='right_only'].copy()
    rename_map = {col: col.replace('_df2', '') for col in unique_df2.columns if '_df2' in col}
    unique_df2.rename(columns=rename_map, inplace=True)
    merged = pd.concat([df1, unique_df2[list(df1.columns)]], ignore_index=True)
    return merged

def compare_dfs(df1, df2, keys, cols, epsilon=1):
    """Compare columns between two DFs, return mismatches above epsilon."""
    mdf = pd.merge(df1, df2, on=keys, suffixes=('_df1', '_df2'))
    errors = {}
    for col in cols:
        err = np.abs(mdf[f"{col}_df1"] - mdf[f"{col}_df2"])
        errors[f"{col}_error"] = err
    mask = np.column_stack([v > epsilon for v in errors.values()]).any(axis=1)
    result = mdf.loc[mask, [*keys] + sum(([f"{col}_df1", f"{col}_df2", f"{col}_error"] for col in cols),[])]
    return result

################################################################################
# --------------- Plotting functions ------------------------------------------
################################################################################

def plot_transitions(df, Tvalues, idxs, label):
    """Plot several transitions from the given DataFrame."""
    for i in idxs:
        row = df.iloc[i]
        plt.plot(Tvalues, [row[f"{int(T)}Ke"] for T in Tvalues], '--', label=f"{row['vu']}-{row['vl']} {row['Ju']}-{row['Jl']}")
    plt.grid(alpha=0.5)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Collision rate (cm3 s-1)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.title(label)
    plt.show()

def plot_matrix_comparison(df_error, cols):
    """Visualize error matrix."""
    data = df_error[[f"{col}_error" for col in cols]].to_numpy()
    plt.imshow(data, cmap='viridis', aspect='auto', vmin=0, vmax=10)
    plt.colorbar(label="Absolute relative error")
    plt.xlabel("Temperatures K")
    plt.ylabel("Lines Index")
    plt.title("Comparison error")
    plt.tight_layout()
    plt.show()

################################################################################
# --------------- Main study pipeline -----------------------------------------
################################################################################

def full_study(label, collision_file, level_file, stancil_settings, temp_cols):
    print(f"\n##### {label} #####")

    # Load Meudon DF
    df_meudon = read_collision_to_df(collision_file)
    df_levelco = read_level_file(level_file, stancil_settings['level_start'], stancil_settings['level_columns'], stancil_settings['level_end'])
    df_meudon_full = merge_level_data(df_meudon, df_levelco, Coltransi=['UP','LOW'], Leveltransi=['v','J'])
    df_meudon_renamed = rename_meudon_columns(df_meudon_full)

    # Load Stancil
    df_stancil = read_collision_to_df(stancil_settings['stancil_file'])
    df_stancil_full = merge_level_data(df_stancil, df_levelco, Coltransi=['nu','nl'], Leveltransi=['v','J'])
    df_stancil_full = update_nu_nl_from_level(df_stancil_full, df_levelco)

    # Merge PDR and Stancil
    df_fusion = merge_out_pdr(df_stancil_full, df_meudon_renamed)
    df_fusion.sort_values(by=['vu','Ju','vl','Jl'], inplace=True)
    df_fusion.reset_index(drop=True, inplace=True)
    df_fusion['ID'] = df_fusion.index + 1

    # Graphics
    plot_transitions(df_fusion, temp_cols, idxs=[0,1,2], label=f"{label}: New merged transitions")

    # Comparisons
    keys = ['vu','Ju','vl','Jl']
    comparison = compare_dfs(df_fusion, df_meudon_renamed, keys=keys, cols=temp_cols, epsilon=1)
    plot_matrix_comparison(comparison, temp_cols)

    return df_fusion, comparison

################################################################################
# --------------- Example usage for H, oH2, pH2 -------------------------------
################################################################################

Tcols = ['2Ke', '3Ke', '5Ke', '7Ke', '10Ke', '15Ke', '20Ke', '30Ke', '50Ke', '70Ke', '100Ke', '150Ke', '200Ke', '300Ke', '500Ke', '700Ke', '1000Ke', '1500Ke', '2000Ke', '3000Ke', '5000Ke', '7000Ke', '10000Ke', '15000Ke', '20000Ke']

# Example settings, you must replace with your actual file paths and line/col info
studies = {
    'H': {
        'collision_file': 'coll_co_h.dat',
        'level_file': 'level_co.dat',
        'stancil_file': 'co_Stancil2024.dat',
        'level_start': 7,
        'level_end': 253,
        'level_columns': ['n', 'g', 'EnK', 'o', 'v', 'J', 'comment'], },
    'oH2': {
        'collision_file': 'coll_co_oh2.dat',
        'level_file': 'level_co.dat',
        'stancil_file': 'co_Stancil2024.dat',
        'level_start': 7,
        'level_end': 253,
        'level_columns': ['n', 'g', 'EnK', 'o', 'v', 'J', 'comment'],  },
    'pH2': {
        'collision_file': 'coll_co_pH2.dat',
        'level_file': 'level_co.dat',
        'stancil_file': 'co_Stancil2024.dat',
        'level_start': 7,
        'level_end': 253,
        'level_columns': ['n', 'g', 'EnK', 'o', 'v', 'J', 'comment'], },
}


for label, params in studies.items():
    stancil_params = {
        'stancil_file': params['stancil_file'],
        'level_start': params['level_start'],
        'level_end': params['level_end'],
        'level_columns': params['level_columns'],
    }
    df_final, df_compare = full_study(
        label,
        params['collision_file'],
        params['level_file'],
        stancil_params,
        Tcols    )