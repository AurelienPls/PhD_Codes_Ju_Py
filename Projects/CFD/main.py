#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:37:33 2026

@author: aurelien
"""
import numpy as np 
import sys
chemin_modules = "/Users/aurelien/Desktop/PhD/Codes/Projects/CFD/"
sys.path.append(chemin_modules)
import cfd_lib as cfd

def run_simulation():
    print("=== Démarrage de la simulation CFD 1D ===")
    
    # 1. Configuration du milieu et de la géométrie
    config = {
        "length": 2.0,            # Longueur du domaine
        "nx": 201,                # Nombre de points de maillage
        "sim_time": 0.4,          # Temps total de simulation physique
        "cfl": 0.8,               # Nombre de Courant (doit être < 1 pour la stabilité)
        "u_left": 2.0,            # Vitesse du fluide à gauche (génère la pression)
        "u_right": 1.0,           # Vitesse du fluide à droite
        "split_point": 0.5        # Position initiale du front de choc
    }
    
    # --- PHASE READ/INIT ---
    print("[INIT] Création du maillage et des conditions initiales...")
    x, dx = cfd.init_geometry_1d(config["length"], config["nx"])
    u0 = cfd.init_shock_conditions(config["nx"], x, config["split_point"], config["u_left"], config["u_right"])
    
    # Calcul dynamique du temps pour garantir la stabilité mathématique
    dt = cfd.compute_dt_cfl(u0, dx, config["cfl"])
    nt = int(config["sim_time"] / dt)
    print(f"[INIT] Maillage spatial : dx = {dx:.4f}")
    print(f"[INIT] Pas de temps dynamique : dt = {dt:.4f} ({nt} itérations)")
    
    # --- PHASE PROCESS ---
    print("[PROCESS] Résolution des équations (Schéma Lax-Friedrichs)...")
    u_final, u_history = cfd.solve_burgers_lax_friedrichs(u0, dx, dt, nt)
    
    # --- PHASE WRITE ---
    print("[WRITE] Export des données...")
    cfd.save_simulation_data("shock_wave_01", x, u_final, config)
    
    # --- PHASE PLOT ---
    print("[PLOT] Génération de la figure...")
    cfd.plot_shock_propagation(x, u0, u_final)
    
    print("=== Simulation terminée avec succès ===")

if __name__ == "__main__":
    run_simulation()
    
#%% 
def run_simulation_2d():
    print("=== Démarrage de la simulation CFD 2D (Goutte) ===")
    
    # 1. Paramètres de la simulation
    config = {
        "lx": 4.0, "ly": 4.0,     # Taille du bassin physique (mètres)
        "nx": 100, "ny": 100,     # Résolution du maillage (pixels de calcul)
        "h_base": 1.0,            # Profondeur de l'eau au repos
        "drop_height": 0.8,       # Hauteur supplémentaire de la goutte
        "radius": 0.3,            # Rayon de la goutte initiale
        "g": 9.81,                # Gravité
        "sim_time": 0.5           # Temps de simulation (secondes)
    }
    
    print("[INIT] Création du bassin 2D...")
    X, Y, dx, dy = cfd.init_geometry_2d(config["lx"], config["ly"], config["nx"], config["ny"])
    h, hu, hv = cfd.init_drop_conditions_2d(X, Y, config["h_base"], config["drop_height"], config["radius"])
    
    # Calcul du pas de temps (CFL 2D)
    # La vitesse d'une onde dans l'eau peu profonde est sqrt(g*h)
    wave_speed = np.sqrt(config["g"] * (config["h_base"] + config["drop_height"]))
    dt = 0.2 * min(dx, dy) / wave_speed
    nt = int(config["sim_time"] / dt)
    
    print(f"[INIT] Maillage : {config['nx']}x{config['ny']} | Pas de temps : dt = {dt:.5f} ({nt} itérations)")
    
    # --- PHASE PROCESS ---
    print("[PROCESS] Calcul en cours (cela peut prendre quelques secondes en 2D)...")
    h_hist = cfd.solve_swe_2d(h, hu, hv, dx, dy, dt, nt, config["g"])
    
    # --- PHASE PLOT ---
    cfd.animate_drop_2d(X, Y, h_hist, filename="propagation_goutte.gif")
    
    print("=== Simulation terminée avec succès ===")

if __name__ == "__main__":
    run_simulation_2d()