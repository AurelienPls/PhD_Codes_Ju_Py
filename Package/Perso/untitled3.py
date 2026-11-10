#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:42:03 2026

@author: aurelien
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% 1. IMPORTATIONS ET PARAMÉTRAGES
import sys
import pandas as pd

# Ajout du chemin vers tes modules personnels
chemin_modules = "/Users/aurelien/Desktop/PhD/Codes/Package/Perso/"
if chemin_modules not in sys.path:
    sys.path.append(chemin_modules)

import ism_pd as ism_pd

# --- Définition des chemins ---
obs_path = "/Users/aurelien/Desktop/PhD/Data/0_JWST_ORION/Base/Orion_JWST/Base/"
model_path = "/Users/aurelien/Desktop/Stage_M2/Traitement/"
db_path = "/Users/aurelien/Desktop/PhD/Data/"

# --- Noms des fichiers ---
fichier_obs = obs_path + "Orion_Perso_AllLines_Correction_Mixte.dat"
fichier_db = db_path + "CO_Lines_Levels_HITRAN_Formatted.dat" # La base qu'on a formatée ensemble !

# Liste des grilles de modèles à tester
fichiers_grilles = [
    model_path + "Grid_JWST_CO_UpdateV1.dat",
    # model_path + "Grid_2_Nom.dat"  <-- Ajoute autant de grilles que tu veux ici
]

#%% 2. PROCESSING UNIFIÉ (Génération du DataFrame)

print("🚀 Lancement du traitement unifié PyISM -> Pandas...")

# Cette seule ligne charge la DB, lit l'obs, interpole les grilles, fait les fits, 
# calcule la physique (Nu/gu) et fusionne tout dans une seule table Pandas.
df_master = ism_pd.process_to_unified_df(
    obs_file=fichier_obs, 
    obs_name="JWST_Orion_Mixte", 
    grid_files=fichiers_grilles, 
    db_file=fichier_db,
    threshold=0.3
)

print("\n✅ Traitement terminé ! Aperçu du DataFrame Master :")
print(df_master.head())

# (Optionnel) Sauvegarde de tout ton fit et tes données physiques pour analyse ultérieure
# df_master.to_csv(obs_path + "Resultats_Fit_Master.csv", index=False)


#%% 3. DIAGRAMMES D'EXCITATION PAR NIVEAU V

# La fonction se charge de filtrer dynamiquement les v présents dans le DataFrame
# et trace un graphique séparé pour chacun. Fini les IndexError !
ism_pd.plot_excitation_df(
    df_unified=df_master, 
    title="Excitation CO - Modèle vs Obs"
)


#%% 4. DIAGRAMME D'INTENSITÉ GLOBAL

# Trace toutes les intensités sur un seul graphique, parfaitement alignées grâce à l'indexation Pandas.
ism_pd.plot_intensity_df(
    df_unified=df_master,
    title="Intensités CO (Mixte) - JWST vs Modèles"
)