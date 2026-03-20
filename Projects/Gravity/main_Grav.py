#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:40:59 2026

@author: aurelien
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
chemin_modules = "/Users/aurelien/Desktop/PhD/Codes/Projects/Gravity/"
sys.path.append(chemin_modules)

import nbody_lib  as nbody
# --- PARAMÈTRES ---
G = 1.0
dt = 0.015
num_stars = 800  # Plus d'étoiles = plus de chances de voir des bras nets
scale_radius = 5.0 # Contrôle la concentration du disque
central_mass = 2000.0
star_mass = 1.0

# --- INITIALISATION ---
sim = nbody.NBodySimulation(G=G, softening=0.3, dt=dt)

masses, positions, velocities = nbody.create_realistic_disk(
    num_stars=num_stars, 
    center=[0.0, 0.0], 
    scale_radius=scale_radius, 
    central_mass=central_mass, 
    star_mass=star_mass, 
    G=G
)

sim.add_particles(masses, positions, velocities)

# --- AFFICHAGE ---
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_facecolor('black')
ax.set_title("Évolution d'un disque galactique")

# Points plus petits et un peu transparents pour un effet "nuage"
scatter_stars = ax.scatter(sim.positions[1:, 0], sim.positions[1:, 1], s=1, c='cyan', alpha=0.6)
scatter_center = ax.scatter(sim.positions[0, 0], sim.positions[0, 1], s=15, c='white', marker='+')

def update(frame):
    for _ in range(3): # Accélère le temps
        sim.step()
    scatter_stars.set_offsets(sim.positions[1:])
    scatter_center.set_offsets(sim.positions[0])
    return scatter_stars, scatter_center

ani = animation.FuncAnimation(fig, update, frames=400, interval=20, blit=True)
plt.show()
#%%
# --- PARAMÈTRES ---
G = 1.0
dt = 0.015
num_stars = 1000  # On augmente pour bien voir les bras
galaxy_radius = 15.0
central_mass = 1000.0
star_mass = 1.0

# --- INITIALISATION ---
sim = nbody.NBodySimulation(G=G, softening=0.3, dt=dt)

# Utilisation de la nouvelle fonction spirale
masses, positions, velocities = nbody.create_spiral_galaxy(
    num_stars=num_stars, 
    center=[0.0, 0.0], 
    radius=galaxy_radius, 
    central_mass=central_mass, 
    star_mass=star_mass, 
    G=G,
    num_arms=4,  # Tu peux tester avec 3 ou 4 bras !
    velocity_factor =0.8
)

sim.add_particles(masses, positions, velocities)

# --- AFFICHAGE ---
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_facecolor('black')
ax.set_title("Galaxie Spirale Initiale (N-corps)")

scatter_stars = ax.scatter(sim.positions[1:, 0], sim.positions[1:, 1], s=1, c='cyan', alpha=0.6)
scatter_center = ax.scatter(sim.positions[0, 0], sim.positions[0, 1], s=15, c='white', marker='+')

def update(frame):
    for _ in range(3): 
        sim.step()
    scatter_stars.set_offsets(sim.positions[1:])
    scatter_center.set_offsets(sim.positions[0])
    return scatter_stars, scatter_center

ani = animation.FuncAnimation(fig, update, frames=400, interval=20, blit=True)
plt.show()
