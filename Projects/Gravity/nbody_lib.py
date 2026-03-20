#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:40:39 2026

@author: aurelien
"""

import numpy as np

class NBodySimulation:
    def __init__(self, G=1.0, softening=0.1, dt=0.01):
        """
        Initialise la simulation.
        G: Constante gravitationnelle (ajustable pour l'échelle de la simu)
        softening: Évite les divisions par zéro lors des collisions proches
        dt: Pas de temps de la simulation
        """
        self.G = G
        self.softening = softening
        self.dt = dt
        
        self.masses = np.array([])
        self.positions = np.empty((0, 2)) # S'adaptera en 3D dynamiquement
        self.velocities = np.empty((0, 2))

    def add_particles(self, masses, positions, velocities):
        """Ajoute un groupe de particules à la simulation."""
        self.masses = np.concatenate((self.masses, masses)) if self.masses.size else np.array(masses)
        self.positions = np.vstack((self.positions, positions)) if self.positions.size else np.array(positions)
        self.velocities = np.vstack((self.velocities, velocities)) if self.velocities.size else np.array(velocities)

    def compute_accelerations(self):
        """Calcule l'accélération de chaque particule (vectorisé pour la vitesse)."""
        N = self.positions.shape[0]
        
        # diff[i, j] contient le vecteur allant de la particule i à la particule j
        # C'est ici que la magie opère pour la 2D ou la 3D : on soustrait des matrices
        diff = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]
        
        # Distance au carré plus le softening
        dist_sq = np.sum(diff**2, axis=-1) + self.softening**2
        
        # Facteur de force : G * m_j / r^3
        # On utilise une astuce NumPy pour diviser efficacement
        f_factor = self.G * self.masses[np.newaxis, :] / (dist_sq**1.5)
        
        # On met la diagonale à zéro pour qu'une particule n'exerce pas de force sur elle-même
        np.fill_diagonal(f_factor, 0)
        
        # Accélération = somme des forces massiques
        accelerations = np.sum(diff * f_factor[:, :, np.newaxis], axis=1)
        return accelerations

    def step(self):
        """Fait avancer la simulation d'un pas de temps (Méthode Euler semi-implicite)."""
        accelerations = self.compute_accelerations()
        
        # Mise à jour des vitesses
        self.velocities += accelerations * self.dt
        
        # Mise à jour des positions avec les nouvelles vitesses (plus stable que l'Euler classique)
        self.positions += self.velocities * self.dt


def create_galaxy(num_stars, center, radius, central_mass, star_mass, G):
    """
    Génère les conditions initiales pour une galaxie en 2D stable.
    """
    center = np.array(center)
    
    # Génération aléatoire des positions polaires
    angles = np.random.uniform(0, 2 * np.pi, num_stars)
    radii = np.random.uniform(0.5, radius, num_stars) # On évite le centre absolu
    
    # Conversion en cartésien
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    positions = np.column_stack((x, y))
    
    # Les masses
    masses = np.full(num_stars, star_mass)
    
    # Pour que la galaxie tourne sans s'effondrer, il faut la bonne vitesse orbitale
    # v = sqrt(G * M / r) (simplification en supposant que la masse centrale domine)
    velocities_mag = np.sqrt(G * central_mass / radii)
    
    # Le vecteur vitesse doit être perpendiculaire au vecteur position
    vx = -velocities_mag * np.sin(angles)
    vy = velocities_mag * np.cos(angles)
    velocities = np.column_stack((vx, vy))
    
    # Ajout du trou noir supermassif au centre
    positions = np.vstack(([center[0], center[1]], positions))
    velocities = np.vstack(([0.0, 0.0], velocities))
    masses = np.insert(masses, 0, central_mass)
    
    return masses, positions, velocities

def create_realistic_disk(num_stars, center, scale_radius, central_mass, star_mass, G):
    """
    Génère un disque galactique réaliste avec une distribution exponentielle.
    """
    center = np.array(center)
    
    # 1. Distribution spatiale : plus dense au centre (distribution exponentielle)
    # On génère des rayons et on évite le centre absolu pour la stabilité
    radii = np.random.exponential(scale=scale_radius, size=num_stars)
    radii = np.clip(radii, 0.5, scale_radius * 4) # On limite la taille maximale
    
    angles = np.random.uniform(0, 2 * np.pi, num_stars)
    
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    positions = np.column_stack((x, y))
    
    masses = np.full(num_stars, star_mass)
    
    # 2. Vitesses orbitales (Rotation Différentielle)
    # L'approximation keplérienne : v = sqrt(G * M / r)
    # Pour être un peu plus réaliste, on inclut la masse des étoiles intérieures 
    # (simplifié ici en ajoutant une fraction de la masse du disque)
    enclosed_mass = central_mass + (radii / (scale_radius * 4)) * (num_stars * star_mass)
    velocities_mag = np.sqrt(G * enclosed_mass / radii)
    
    # Vecteurs vitesse (perpendiculaires au rayon)
    vx = -velocities_mag * np.sin(angles)
    vy = velocities_mag * np.cos(angles)
    
    # 3. Ajout d'une petite perturbation (dispersion des vitesses)
    # C'est ce "bruit" qui va créer des amas, qui seront ensuite étirés en bras spiraux
    vx += np.random.normal(0, velocities_mag * 0.05, num_stars)
    vy += np.random.normal(0, velocities_mag * 0.05, num_stars)
    
    velocities = np.column_stack((vx, vy))
    
    # Ajout du trou noir supermassif
    positions = np.vstack(([center[0], center[1]], positions))
    velocities = np.vstack(([0.0, 0.0], velocities))
    masses = np.insert(masses, 0, central_mass)
    
    return masses, positions, velocities


def create_spiral_galaxy(num_stars, center, radius, central_mass, star_mass, G, num_arms=2,velocity_factor=1):
    """
    Génère une galaxie avec une structure initiale en spirale logarithmique.
    """
    center = np.array(center)
    
    # 1. Génération des rayons (plus dense vers le centre)
    radii = np.random.exponential(scale=radius/2, size=num_stars)
    radii = np.clip(radii, 0.5, radius) # On garde les étoiles dans des limites raisonnables
    
    # 2. Calcul de l'angle de base pour la spirale
    b = 0.3  # Paramètre d'enroulement de la spirale (modifie-le pour voir l'effet)
    theta_spiral = (1 / b) * np.log(radii)
    
    # 3. Répartition des étoiles dans les différents bras
    arm_indices = np.random.randint(0, num_arms, size=num_stars)
    arm_offsets = arm_indices * (2 * np.pi / num_arms)
    
    # 4. Ajout d'une dispersion (bruit) pour élargir les bras
    # Sans ça, toutes les étoiles feraient une ligne parfaite
    dispersion = np.random.normal(0, 0.5, num_stars) 
    angles = theta_spiral + arm_offsets + dispersion
    
    # Conversion en cartésien
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    positions = np.column_stack((x, y))
    
    masses = np.full(num_stars, star_mass)
    
    # 5. Vitesses orbitales (toujours pour maintenir la stabilité dynamique)
    # On dirige la vitesse tangentiellement au cercle, pas le long du bras
    velocities_mag = np.sqrt(G * central_mass / radii)
    vx = -velocities_mag * np.sin(angles)
    vy = velocities_mag * np.cos(angles)
    velocities = np.column_stack((vx, vy))
    
    # Ajout du trou noir supermassif
    positions = np.vstack(([center[0], center[1]], positions))
    velocities = np.vstack(([0.0, 0.0], velocities))
    masses = np.insert(masses, 0, central_mass)
    
    
    # Facteur de vitesse : 1.0 = orbite parfaite. 1.5 = explosion. 0.5 = effondrement.
    velocity_factor = velocity_factor
    velocities *= velocity_factor
    


    return masses, positions, velocities

