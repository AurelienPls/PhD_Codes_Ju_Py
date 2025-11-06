#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 11:16:26 2025

@author: aurelien PILUSO
If you find any issue with this code, please report it to: aurelien.piluso@gmail.com
This module contains utility functions for data processing.
"""


import matplotlib.pyplot as plt
import re
import pandas as pd    
import numpy as np
import os
import plotly.express as px

########READING HELPERS########


def dat_to_df(
    path,            # Directory or file path
    name,            # Filename
    start_row=0,     # First line to read (index)
    end_row=None,    # Last line to read (index, not inclusive)
    col_names=None,  # List of column names
    to_float=True,   # Convert columns to float if possible
    delimiter=None,  # Delimiter, default: any whitespace
    comment_char=None # Character to skip comment lines
):
    """Read a .dat file section and convert to a DataFrame."""
    with open(os.path.join(path, name), 'r') as f:
        lines = f.read().splitlines()
    if comment_char: 
        lines = [line for line in lines if not line.strip().startswith(comment_char)]
    section = lines[start_row:end_row] if end_row is not None else lines[start_row:]
    splitter = (lambda row: row.split(delimiter)) if delimiter else (lambda row: row.split())
    data = [splitter(row) for row in section if row.strip()]
    df = pd.DataFrame(data, columns=col_names if col_names is not None else None)
    # Try converting columns to float
    if to_float and col_names is not None:
        for col in col_names:
            try: df[col] = df[col].astype(float)
            except ValueError: pass
    return df




def read_energy_levels(path, element="H2", start=8, stop=-5):
    col_names = ['n', 'g', 'En(K)', 'o', 'v', 'J', 'comment']
    df = dat_to_df(
        path=path,
        name=f"level_{element}.dat",
        start_row=start,
        end_row=stop,
        col_names=col_names,
        to_float=False
    )
    df[['n', 'v', 'J']] = df[['n', 'v', 'J']].astype(int)
    return df




def read_lines_df(data_path, molecule="h2"):
    """Read radiative transition data for a given molecule and return as DataFrame."""
    file = os.path.join(data_path, f"line_{molecule}.dat")
    data = []

    with open(file, 'r') as f:
        for _ in range(4):
            next(f)  # skip header lines
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                n_trans, nu, nl = map(int, parts[0:3])
                E_K, Aij = map(float, parts[3:5])
                quant_idx = parts.index('quant:')
                info_idx = parts.index('info:')

                vu = int(parts[quant_idx+1].rstrip(';'))
                Ju = int(parts[quant_idx+2].rstrip(';'))
                vl = int(parts[quant_idx+3].rstrip(';'))
                Jl = int(parts[quant_idx+4].rstrip(';'))

                desc = parts[info_idx+1:]

                if molecule == "h2":
                    lambda_val = np.nan
                    desc_text = ' '.join(desc)
                    if ';' in desc_text:
                        parts_desc = desc_text.split(';', 1)
                        try:
                            lambda_val = float(parts_desc[0].replace('micrometres', '').strip())
                        except ValueError:
                            lambda_val = np.nan
                        desc_text = parts_desc[1].strip() if len(parts_desc) > 1 else desc_text
                    data.append([n_trans, nu, nl, E_K, Aij, vu, Ju, vl, Jl, desc_text, lambda_val])
                else:
                    frequency_ghz = np.nan
                    desc_text = ' '.join(desc).strip()
                    if desc and "GHz" in desc[-1]:
                        try:
                            frequency_ghz = float(desc[-1].replace("GHz", ""))
                            desc_text = ' '.join(desc[:-1]).strip()
                        except ValueError:
                            pass
                    data.append([n_trans, nu, nl, E_K, Aij, vu, Ju, vl, Jl, desc_text, frequency_ghz])
            except (ValueError, IndexError):
                continue
    cols = ['n_trans', 'nu', 'nl', 'E_K', 'Aij_s-1', 'vu', 'Ju', 'vl', 'Jl']
    if molecule == "h2":
        cols += ['Description_text', 'Lambda_micrometres']
    else:
        cols += ['Description_text', 'Frequency_GHz']

    df = pd.DataFrame(data, columns=cols)
    int_cols = ['n_trans', 'nu', 'nl', 'vu', 'Ju', 'vl', 'Jl']
    for c in int_cols:
        df[c] = df[c].astype(int)

    df['E_K'] = df['E_K'].astype(float)
    df['Aij_s-1'] = df['Aij_s-1'].astype(float)

    if molecule == "h2":
        df['Lambda_micrometres'] = pd.to_numeric(df['Lambda_micrometres'], errors='coerce')
    else:
        df['Frequency_GHz'] = pd.to_numeric(df['Frequency_GHz'], errors='coerce')
    return df


def read_collision_to_df( path, name: str) -> pd.DataFrame:
    filename = f"coll_{name}.dat"
    file = path + filename

    collision_temperatures = []
    data_lines = []
    start_data_reading = False

    collision_rate_columns = [f"{int(temp) if temp == int(temp) else temp}K_PDR" for temp in collision_temperatures]
    initial_columns = ['NUM', 'UP', 'LOW']
    all_columns = initial_columns + collision_rate_columns
    df_data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= len(all_columns): 
            num_up_low = [int(p) for p in parts[:3]]
            rates = [float(p) for p in parts[3:3+len(collision_rate_columns)]]
            df_data.append(num_up_low + rates)
    df = pd.DataFrame(df_data, columns=all_columns)
    return df




def Read_Stancil_transitions(root,line_start,endline,Col,desired_order,
                             df_level_u,df_level_l):
    df1 = dat_to_df(root,"co_Stancil2024.dat" , start_row= line_start , end_row=endline,col_names=Col)
    df1[Col] = df1[Col].astype(float)
    df1[["ID","nu","nl"]] = df1[["ID","nu","nl"]].astype(int)
    df2 = pd.merge(df1, df_level_u, on='nu', how='left')
    df3 = pd.merge(df2, df_level_l, on='nl', how='left')
    df3 = df3[desired_order]
    return df3









def merge_level_data(df_transitions: pd.DataFrame, 
                     df_level: pd.DataFrame, 
                     Col_transi,
                     Level_transi) -> pd.DataFrame:
    """
    Fusionne un DataFrame de transitions avec un DataFrame de niveaux pour ajouter
    les informations de v et J pour les niveaux supérieur (Nup) et inférieur (Nlow).

    Args:
        df_transitions (pd.DataFrame): Le DataFrame principal contenant les transitions
                                       (doit inclure 'Nup' et 'Nlow').
        df_level (pd.DataFrame): Le DataFrame des niveaux (doit inclure 'n', 'v', 'J').

    Returns:
        pd.DataFrame: Un nouveau DataFrame fusionné avec les colonnes 'vu', 'Ju', 'vl', 'Jl' ajoutées.
                      Retourne un DataFrame vide si les colonnes nécessaires sont manquantes.
    """
    df_level_for_Nup = df_level[Level_transi].rename(
        columns={'n': Col_transi[0], 'v': 'vu', 'J': 'Ju'}  )
    
    df_merged = pd.merge(
        df_transitions,
        df_level_for_Nup,
        on= Col_transi[0],
        how='left' )
    
    df_level_for_Nlow = df_level[Level_transi].rename(
        columns={'n': Col_transi[1], 'v': 'vl', 'J': 'Jl'} )
    
    df_final = pd.merge(
        df_merged,
        df_level_for_Nlow,
        on= Col_transi[1],
        how='left' )
    return df_final  


def merge_levels(trans_df, level_df, trans_cols, level_cols):
    """Merge transition DataFrame with level DataFrame to add level info."""    
    upper = level_df[level_cols].rename(
        columns={'n': trans_cols[0], 'v': 'vu', 'J': 'Ju'})
    merged = pd.merge(trans_df, upper, on=trans_cols[0], how='left')
    lower = level_df[level_cols].rename(
        columns={'n': trans_cols[1], 'v': 'vl', 'J': 'Jl'})
    result = pd.merge(merged, lower, on=trans_cols[1], how='left')
    return result






def extract_v_j(text):
    """Return v,J from 'ElecStateLabel=X v=.. J=..'."""
    m = re.match(r"ElecStateLabel=X v=(\d+) J=(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def rename_columns(df):
    mapping = {'NUM': 'ID', 'UP': 'nu', 'LOW': 'nl'}
    return df.rename(columns=mapping)

def merge_unique_rows(df_main: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    keys = ['vu', 'Ju', 'vl', 'Jl']
    merged = pd.merge(df_main, df_new, on=keys, how='outer', indicator=True, suffixes=('_old', '_new'))
    new_only = merged[merged['_merge'] == 'right_only'].rename(
        columns={c: c.replace('_new', '') for c in merged.columns if '_new' in c})[df_new.columns]
    return pd.concat([df_main, new_only], ignore_index=True)




##########WRITING HELPERS##########

def write_radiative_coefficient_file(df, filepath):
    n_trans = len(df)
    header = [
        f"#{n_trans:d}         # Number of transitions",
        "#K          # Unit energy transitions",
        "#   n     nu     nl           E(K)           Aij(s-1)               quant:    vu      Ju      vl      Jl  info:       Description",
        "#" + "-" * 118
    ]
    with open(filepath, 'w') as f:
        for line in header:
            f.write(line + '\n')
        for _, row in df.iterrows():
            desc = str(row['Description_text'])
            if pd.notna(row.get('Frequency_GHz')):
                desc += f" {row['Frequency_GHz']:.3f} GHz"

            line = (
                f"{int(row['n_trans']):4d} "
                f"{int(row['nu']):6d} "
                f"{int(row['nl']):6d} "
                f"{row['E_K']:18.8f} "
                f"{row['Aij_s-1']:20.7E} "
                "quant: "
                f"{int(row['vu']):5d}; "
                f"{int(row['Ju']):5d}; "
                f"{int(row['vl']):5d}; "
                f"{int(row['Jl']):5d}; "
                f"info:   {desc}"
            )
            f.write(line + '\n')

    print(f"\nFile '{filepath}' created.")


def write_collisions(df, n_levels, file_out="Coll_co_oh2.txt"):
    temp = [2,3,5,7,10,15,20,30,50,70,100,150,200,300,500,700,1000,1500,2000,3000,5000,7000,10000,15000,20000]
    cols = ['NUM','nu','nl'] + [f'{int(t)}Ke' for t in temp]
    df = df[cols]
    with open(file_out, 'w') as f:
        f.write(f"!NUMBER OF ENERGY LEVELS\n{n_levels}\n")
        f.write(f"!NUMBER OF COLLISIONAL TRANSITIONS\n{len(df)}\n")
        f.write(f"!NUMBER OF COLLISION TEMPERATURES\n{len(temp)}\n")
        f.write("!COLLISION TEMPERATURES\n" + " "*25 + ''.join(f"{t:10.1f}" for t in temp) + "\n")
        f.write("!NUM   UP   LOW   DOWN_COLL_RATES [cm^3 s^-1]\n")
        for _, r in df.iterrows():
            vals = [f"{int(r['NUM']):<5}",f"{int(r['nu']):<5}",f"{int(r['nl']):<5}"]
            vals += [f"{r[f'{int(t)}Ke']:.4e}" for t in temp]
            f.write(' '.join(vals)+"\n")
    print(f"Collision data written to {file_out}")







###########CALCULATION HELPERS############
def abs_rel_error(df1, df2, key_cols, value_cols):
    merged = pd.merge(df1, df2, on=key_cols, how='inner', suffixes=('_1', '_2'))
    out = merged[key_cols].copy()
    for col in value_cols:
        a, b = merged[f'{col}Ke_1'], merged[f'{col}Ke_2']
        err = np.where(a != 0, np.abs(a-b)/np.abs(a), 0)
        out[f'{col}_err'] = np.nan_to_num(err, nan=0, posinf=0, neginf=0)
    return out

def add_veff(df):
    """Add effective quantum numbers V_eff_u and V_eff_l to DataFrame."""
    df_sorted = df.sort_values(['vu', 'Ju', 'vl', 'Jl'])
    # Get max J for each v upper/lower
    ju_max = df_sorted.groupby('vu')['Ju'].transform('max')
    jl_max = df_sorted.groupby('vl')['Jl'].transform('max')
    # Compute effective quantum numbers
    df_sorted['V_eff_u'] = df_sorted['vu'] + df_sorted['Ju']/ju_max
    df_sorted['V_eff_l'] = df_sorted['vl'] + df_sorted['Jl']/jl_max
    return df_sorted


##########PLOTTING HELPERS############
def plot_comparison(df1, df2, col, eps=1e-9, label1='df_co', label2='df_combined',mode="log",lim=5000):
    """Plot comparison of a column between two DataFrames with relative error."""
    # Compute relative error (%)
    rel_err = np.abs(df2[col] - df1[col]) / (df1[col] + eps) * 100
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[3, 1])
    # Top: compare values
    ax.scatter(df1.index, df1[col], c='r', s=20, label=label1)
    ax.scatter(df2.index, df2[col], c='b', s=20, alpha=0.5, label=label2)
    ax.set_yscale('log')
    ax.set_ylabel(col)
    ax.set_xlim([0, lim])
    ax.set_title(f'Comparison: {col}')
    ax.grid(True, ls='--', alpha=0.7)
    ax.legend()
    # Bottom: error
    ax2.plot(df1.index, rel_err, 'go:', label=r'Rel. Error ($\%$)')
    ax2.axhline(0, c='grey', ls=':', lw=0.8)
    ax2.set_ylim([-2,100])
    ax2.set_xlabel('Index')
    ax2.set_ylabel(r'Rel. Error [$\%$]')
    ax2.grid(True, ls='-', alpha=0.6)
    ax2.legend()
    if mode == "log":
        ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

def plot_interactive(df, x_col='V_eff_l', y_col='V_eff_u', c_col='100.0', log_color=False, hover_cols=['vu', 'Ju', 'vl', 'Jl'], title="Title"):
    """Create an interactive scatter plot using Plotly."""
    color_data = df[c_col].copy()
    if log_color:
        min_val = color_data.min()
        if min_val > 0:
            color_data = np.log10(color_data)
            c_label = f"{c_col} (log10)"
        else:
            color_data = color_data # fallback to linear if zeros/negatives
            c_label = c_col
    else:
        c_label = c_col
    fig = px.scatter(df, x=x_col, y=y_col, color=color_data, hover_data=hover_cols, title=title, labels={"color": c_label})
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.show()

def prolong_curve(x, y, x_new, method='flat', x_fit=1500):
    """Extrapolate y for new x, using last region's trend."""
    x = np.asarray(x)
    y = np.asarray(y)
    x_new = np.asarray(x_new)
    mask = x >= x_fit
    x_fit_data = x[mask]
    y_fit_data = y[mask]
    if method == 'flat':
        y_out = np.full_like(x_new, y_fit_data[-1], dtype=float)
    elif method == 'linear':
        a, b = np.polyfit(x_fit_data, y_fit_data, 1)
        y_out = a * x_new + b
    elif method == 'log':
        if np.any(x_fit_data <= 0) or np.any(x_new <= 0):
            raise ValueError('x for log method must be > 0')
        a, b = np.polyfit(np.log(x_fit_data), y_fit_data, 1)
        y_out = a * np.log(x_new) + b
    else:
        raise ValueError('method must be flat, linear, or log')
    return y_out