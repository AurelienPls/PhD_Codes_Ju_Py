#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:37:09 2026

@author: aurelien
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

# ==========================================
# 1. READ / INIT (Configuration et Géométrie)
# ==========================================

def init_geometry_1d(length, nx):
    """Crée le maillage 1D."""
    x = np.linspace(0, length, nx)
    dx = length / (nx - 1)
    return x, dx

def init_shock_conditions(nx, x, split_point=0.5, u_left=2.0, u_right=1.0):
    """
    Crée une condition initiale en forme de marche (Riemann problem)
    qui va générer une onde de choc.
    """
    u0 = np.ones(nx) * u_right
    u0[x <= split_point] = u_left
    return u0

def compute_dt_cfl(u_max, dx, cfl_number=0.8):
    """Calcule le pas de temps dynamique basé sur la condition CFL."""
    return cfl_number * dx / np.max(np.abs(u_max))

# ==========================================
# 2. PROCESS (Solvers et Mathématiques)
# ==========================================

def flux_burgers(u):
    """Calcule le flux pour l'équation de Burgers."""
    return 0.5 * u**2

def solve_burgers_lax_friedrichs(u0, dx, dt, nt):
    """
    Résout l'équation de Burgers 1D avec le schéma de Lax-Friedrichs.
    Robuste pour capturer la propagation d'une onde de choc.
    """
    u = u0.copy()
    u_hist = [u0.copy()] # Historique pour l'animation ou les tracés multiples
    
    for n in range(nt):
        un = u.copy()
        
        # Schéma de Lax-Friedrichs vectorisé (évite les boucles for lentes)
        u[1:-1] = 0.5 * (un[2:] + un[:-2]) - (dt / (2 * dx)) * (flux_burgers(un[2:]) - flux_burgers(un[:-2]))
        
        # Conditions aux limites (Neumann : dérivée nulle aux bords)
        u[0] = un[1]
        u[-1] = un[-2]
        
        u_hist.append(u.copy())
        
    return u, u_hist

# ==========================================
# 3. WRITE (Sauvegarde des données)
# ==========================================

def save_simulation_data(filename, x, u, config):
    """Sauvegarde les résultats et la configuration."""
    if not os.path.exists('results'):
        os.makedirs('results')
        
    np.savez(f'results/{filename}.npz', x=x, u=u)
    with open(f'results/{filename}_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print(f"[WRITE] Données sauvegardées sous results/{filename}")

# ==========================================
# 4. PLOT (Visualisation)
# ==========================================

def plot_shock_propagation(x, u0, u_final, title="Propagation de l'onde de choc (Burgers)"):
    """Trace l'état initial et l'état final."""
    plt.figure(figsize=(10, 6))
    
    # État initial
    plt.plot(x, u0, label='État Initial (t=0)', color='blue', linestyle='--')
    
    # État final
    plt.plot(x, u_final, label='État Final', color='red', linewidth=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Position spatiale (x)', fontsize=12)
    plt.ylabel('Vitesse (u)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    
    

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. READ / INIT 2D
# ==========================================

def init_geometry_2d(lx, ly, nx, ny):
    """Crée le maillage 2D."""
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y)
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)
    return X, Y, dx, dy

def init_drop_conditions_2d(X, Y, h_base=1.0, drop_height=2.0, radius=0.2):
    """
    Crée un bassin calme avec une goutte (un pic d'eau) au centre.
    """
    # Hauteur de l'eau initiale
    h = np.ones_like(X) * h_base
    
    # Position du centre
    cx, cy = np.mean(X[0,:]), np.mean(Y[:,0])
    
    # Ajout de la "goutte" (distribution gaussienne)
    r2 = (X - cx)**2 + (Y - cy)**2
    h += drop_height * np.exp(-r2 / (radius**2))
    
    # Vitesses initiales nulles (hu = 0, hv = 0)
    hu = np.zeros_like(X)
    hv = np.zeros_like(X)
    
    return h, hu, hv

# ==========================================
# 2. PROCESS 2D (Shallow Water - Lax-Friedrichs)
# ==========================================

def solve_swe_2d(h, hu, hv, dx, dy, dt, nt, g=9.81):
    """
    Résout les équations de Saint-Venant 2D.
    Utilise une approche vectorisée massive pour la vitesse sous Python.
    """
    h_hist = [h.copy()]
    
    for _ in range(nt):
        h_n, hu_n, hv_n = h.copy(), hu.copy(), hv.copy()
        
        # Calcul des flux avec précautions pour éviter la division par zéro
        u = np.divide(hu_n, h_n, out=np.zeros_like(hu_n), where=h_n!=0)
        v = np.divide(hv_n, h_n, out=np.zeros_like(hv_n), where=h_n!=0)
        
        # Flux en X (F)
        F_h = hu_n
        F_hu = hu_n * u + 0.5 * g * h_n**2
        F_hv = hu_n * v
        
        # Flux en Y (G)
        G_h = hv_n
        G_hu = hv_n * u
        G_hv = hv_n * v + 0.5 * g * h_n**2
        
        # Moyennes spatiales de Lax-Friedrichs (pour stabiliser)
        h_avg = 0.25 * (h_n[2:, 1:-1] + h_n[:-2, 1:-1] + h_n[1:-1, 2:] + h_n[1:-1, :-2])
        hu_avg = 0.25 * (hu_n[2:, 1:-1] + hu_n[:-2, 1:-1] + hu_n[1:-1, 2:] + hu_n[1:-1, :-2])
        hv_avg = 0.25 * (hv_n[2:, 1:-1] + hv_n[:-2, 1:-1] + hv_n[1:-1, 2:] + hv_n[1:-1, :-2])
        
        # Mise à jour des variables au temps t+1
        h[1:-1, 1:-1] = h_avg - (dt / (2 * dx)) * (F_h[2:, 1:-1] - F_h[:-2, 1:-1]) - (dt / (2 * dy)) * (G_h[1:-1, 2:] - G_h[1:-1, :-2])
        hu[1:-1, 1:-1] = hu_avg - (dt / (2 * dx)) * (F_hu[2:, 1:-1] - F_hu[:-2, 1:-1]) - (dt / (2 * dy)) * (G_hu[1:-1, 2:] - G_hu[1:-1, :-2])
        hv[1:-1, 1:-1] = hv_avg - (dt / (2 * dx)) * (F_hv[2:, 1:-1] - F_hv[:-2, 1:-1]) - (dt / (2 * dy)) * (G_hv[1:-1, 2:] - G_hv[1:-1, :-2])
        
        # Conditions aux limites (Murs réfléchissants basiques)
        h[0,:] = h[1,:]; h[-1,:] = h[-2,:]; h[:,0] = h[:,1]; h[:,-1] = h[:,-2]
        hu[0,:] = -hu[1,:]; hu[-1,:] = -hu[-2,:]; hu[:,0] = hu[:,1]; hu[:,-1] = hu[:,-2] # Rebond en X
        hv[0,:] = hv[1,:]; hv[-1,:] = hv[-2,:]; hv[:,0] = -hv[:,1]; hv[:,-1] = -hv[:,-2] # Rebond en Y
        
        h_hist.append(h.copy())
        
    return h_hist

# ==========================================
# 3. PLOT 2D (Animation)
# ==========================================

def animate_drop_2d(X, Y, h_hist, filename="goutte_2d.gif"):
    """Crée et sauvegarde une animation de la propagation 2D."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Configuration visuelle
    cax = ax.pcolormesh(X, Y, h_hist[0], cmap='viridis', shading='auto', vmin=0.9, vmax=1.5)
    fig.colorbar(cax, label='Hauteur d\'eau (h)')
    ax.set_title("Effondrement de la goutte (t=0)")
    ax.set_aspect('equal')
    
    def update(frame):
        cax.set_array(h_hist[frame].ravel())
        ax.set_title(f"Propagation (Itération {frame})")
        return cax,
    
    # On n'anime qu'une frame sur 5 pour accélérer la création du gif
    frames_to_plot = range(0, len(h_hist), max(1, len(h_hist)//50)) 
    ani = animation.FuncAnimation(fig, update, frames=frames_to_plot, blit=False)
    
    print(f"[PLOT] Sauvegarde de l'animation en cours ({filename})...")
    ani.save(filename, fps=10, writer='pillow')
    print("[PLOT] Animation sauvegardée !")