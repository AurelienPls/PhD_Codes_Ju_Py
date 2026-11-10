#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 11:24:21 2025

@author: aurelien
"""

import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import seaborn as sns # Pour kdeplot, installez-le avec pip install seaborn
from mpl_toolkits.mplot3d import Axes3D # Nécessaire pour les plots 3D

# --- Configuration du snap à tracer ---
SNAP_FILE_PATH = '/Users/aurelien/Desktop/These/Formations/Summer_School_Banyuls/RAMSES/RAMSES_AREPO/snap_200.hdf5'

# --- 1. Charger les données du snap ---
x_coords, y_coords, z_coords, c_values, snap_number = None, None, None, None, None
try:
    with h5py.File(SNAP_FILE_PATH, 'r') as f:
        snap_number = int(os.path.basename(SNAP_FILE_PATH).split('_')[-1].split('.')[0])
        part0_data = f['PartType0']
        # Assurez-vous que CenterOfMass a au moins 3 dimensions
        if 'CenterOfMass' in part0_data and part0_data['CenterOfMass'].shape[0] >= 3:
            com_data = part0_data['CenterOfMass'][:]
            x_coords = com_data[0, :]
            y_coords = com_data[1, :]
            z_coords = com_data[2, :]
            c_values = part0_data['Masses'][:]
        else:
            print(f"Erreur: Les données 3D ('CenterOfMass' ou dimensions insuffisantes) ne sont pas complètes pour PartType0 dans {SNAP_FILE_PATH}.")
            exit()

except FileNotFoundError:
    print(f"Erreur : Le fichier '{SNAP_FILE_PATH}' n'a pas été trouvé. Vérifiez le chemin.")
    exit()
except KeyError as e:
    print(f"Erreur : Clé manquante dans le fichier HDF5 : {e}. Assurez-vous que 'PartType0', 'CenterOfMass' et 'Masses' existent.")
    exit()
except Exception as e:
    print(f"Une erreur inattendue s'est produite lors du chargement : {e}")
    exit()

# --- Calculer les limites dynamiques pour les plots ---
# Ajoutez une petite marge pour éviter que les points ne touchent les bords
x_min, x_max = np.min(x_coords), np.max(x_coords)
y_min, y_max = np.min(y_coords), np.max(y_coords)
z_min, z_max = np.min(z_coords), np.max(z_coords)

x_margin = (x_max - x_min) * 0.02
y_margin = (y_max - y_min) * 0.02
z_margin = (z_max - z_min) * 0.02

# --- 2. Plotting Alternatives (X-Y Plane Examples) ---
print("\n--- Visualisations alternatives pour le plan X-Y ---")

# Figure 1: hist2d (Histogramme 2D rectangulaire)
plt.figure(figsize=(10, 8))
plt.hist2d(x_coords, y_coords, bins=256, cmap='magma', cmin=1) # cmin=1 pour ignorer les bacs vides
plt.colorbar(label='Nombre de Particules') # Ou 'Masses' si vous utilisez un weight pour les bacs
plt.title(f"Snap {snap_number} (X-Y) - hist2d")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(x_min - x_margin, x_max + x_margin)
plt.ylim(y_min - y_margin, y_max + y_margin)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()

# Figure 2: seaborn.kdeplot (Estimation de la Densité de Noyau)
# kdeplot peut prendre du temps sur de très grands datasets.
# Pour de grandes données, vous pouvez échantillonner ou augmenter 'n_levels' et 'fill'.
plt.figure(figsize=(10, 8))
sns.kdeplot(x=x_coords, y=y_coords, cmap="magma", fill=True, levels=50, cbar=True, cbar_kws={'label': 'Densité Estimée'})
plt.title(f"Snap {snap_number} (X-Y) - kdeplot")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(x_min - x_margin, x_max + x_margin)
plt.ylim(y_min - y_margin, y_max + y_margin)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()

#%% Vue 3D 
# --- 3. Sortie d'un cube 3D (X, Y, Z) ---
print("\n--- Visualisation 3D du cube ---")

fig_3d = plt.figure(figsize=(12, 12))
ax_3d = fig_3d.add_subplot(111, projection='3d')

scatter_3d = ax_3d.scatter(x_coords, y_coords, z_coords,
                           s=0.5,
                           alpha=0.5,
                           c=c_values,
                           cmap='magma',
                           vmin=0, vmax=2000)

ax_3d.set_xlabel('X Coordinate')
ax_3d.set_ylabel('Y Coordinate')
ax_3d.set_zlabel('Z Coordinate')
ax_3d.set_title(f"Snap {snap_number} (Vue 3D du Centre de Masse)")

ax_3d.set_xlim(600,1000)#x_min - x_margin, x_max + x_margin)
ax_3d.set_ylim(600,1000)#y_min - y_margin, y_max + y_margin)
ax_3d.set_zlim(z_min - z_margin, z_max + z_margin)

# Tentative de forcer l'aspect ratio du cube (nécessite Matplotlib 3.3+)
ax_3d.set_box_aspect([1, (y_max-y_min)/(x_max-x_min), (z_max-z_min)/(x_max-x_min)])

cbar_3d = fig_3d.colorbar(scatter_3d, ax=ax_3d, pad=0.1)
cbar_3d.set_label('Masses')

plt.tight_layout()
plt.show()

print("\nVisualisations terminées.")
#%%
# Obtenez les min/max réels de c_values
mass_min_actual = np.min(c_values)
mass_max_actual = np.max(c_values)
print(f"Masses réelles min: {mass_min_actual}, max: {mass_max_actual}")

# Ajustez vmin/vmax si nécessaire
scatter_3d = ax_3d.scatter(x_coords, y_coords, z_coords,
                           s=0.5,
                           alpha=0.5,
                           c=c_values,
                           cmap='magma',
                           vmin=mass_min_actual, # Ou une valeur fixe si vous le souhaitez
                           vmax=mass_max_actual) # Ou une valeur fixe

#%%
# --- 1. Charger les données du snap ---
x_coords, y_coords, z_coords, c_values, snap_number = None, None, None, None, None
try:
    with h5py.File(SNAP_FILE_PATH, 'r') as f:
        snap_number = int(os.path.basename(SNAP_FILE_PATH).split('_')[-1].split('.')[0])

        if 'PartType0' in f:
            part0_data = f['PartType0']

            if 'CenterOfMass' in part0_data and 'Masses' in part0_data:
                com_raw_data = part0_data['CenterOfMass'][:]
                mass_raw_data = part0_data['Masses'][:]

                print(f"--- Diagnostic des dimensions pour '{SNAP_FILE_PATH}' ---")
                print(f"Shape de CenterOfMass (brut): {com_raw_data.shape}") # (12861790, 3)
                print(f"Shape de Masses (brut): {mass_raw_data.shape}") # (12861790,)
                print("-------------------------------------------------")

                # --- VÉRIFICATION ET INDEXATION CORRIGÉE ---
                # Si CenterOfMass a la forme (N_particles, Dimensions)
                if com_raw_data.ndim == 2 and com_raw_data.shape[1] >= 3:
                    # Ici, les colonnes sont les dimensions (X, Y, Z)
                    # et les lignes sont les particules.
                    x_coords = com_raw_data[:, 0] # Toutes les lignes, 1ère colonne (X)
                    y_coords = com_raw_data[:, 1] # Toutes les lignes, 2ème colonne (Y)
                    z_coords = com_raw_data[:, 2] # Toutes les lignes, 3ème colonne (Z)

                    # Vérifier que le nombre de particules correspond
                    if x_coords.size == mass_raw_data.size:
                        c_values = mass_raw_data
                    else:
                        print(f"Erreur: Le nombre de coordonnées ({x_coords.size}) ne correspond pas au nombre de masses ({mass_raw_data.size}).")
                        exit()
                # Sinon, si CenterOfMass a la forme (Dimensions, N_particles) (moins commun)
                elif com_raw_data.ndim == 2 and com_raw_data.shape[0] >= 3:
                    # Ici, les lignes sont les dimensions (X, Y, Z)
                    # et les colonnes sont les particules.
                    x_coords = com_raw_data[0, :]
                    y_coords = com_raw_data[1, :]
                    z_coords = com_raw_data[2, :]

                    if x_coords.size == mass_raw_data.size:
                        c_values = mass_raw_data
                    else:
                        print(f"Erreur: Le nombre de coordonnées ({x_coords.size}) ne correspond pas au nombre de masses ({mass_raw_data.size}).")
                        exit()
                else:
                    print(f"Erreur: La structure de 'CenterOfMass' dans {SNAP_FILE_PATH} n'est pas attendue (ndim={com_raw_data.ndim}, shape={com_raw_data.shape}).")
                    exit()

            else:
                print(f"Erreur: 'CenterOfMass' ou 'Masses' manquant(s) pour PartType0 dans {SNAP_FILE_PATH}.")
                exit()
        else:
            print(f"Erreur: 'PartType0' non trouvé dans {SNAP_FILE_PATH}.")
            exit()

except FileNotFoundError:
    print(f"Erreur : Le fichier '{SNAP_FILE_PATH}' n'a pas été trouvé. Vérifiez le chemin.")
    exit()
except Exception as e:
    print(f"Une erreur inattendue s'est produite lors du chargement : {e}")
    exit()

# Vérification finale avant le plotting
if x_coords is None or y_coords is None or z_coords is None or c_values is None:
    print("Données non chargées correctement, impossible de procéder aux visualisations.")
    exit()
    
# ... (votre code de chargement) ...

# Vérification finale avant le plotting (cette partie est déjà bonne)
if x_coords is None or y_coords is None or z_coords is None or c_values is None:
    print("Données non chargées correctement, impossible de procéder aux visualisations.")
    exit()

# --- NOUVEAUX PRINTS DE DÉBOGAGE ICI ---
print(f"Taille de x_coords avant plotting: {x_coords.size}")
print(f"Taille de y_coords avant plotting: {y_coords.size}")
print(f"Taille de z_coords avant plotting: {z_coords.size}")
print(f"Taille de c_values avant plotting: {c_values.size}")
# --- FIN DES PRINTS DE DÉBOGAGE ---

# Calculer les limites dynamiques pour les plots
# ... (le reste de votre code de plotting) ...