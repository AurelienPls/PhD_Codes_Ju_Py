#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bibliothèque d'analyse spatiale, spectrale et cinématique pour cubes JWST/NIRSpec.
Optimisée pour l'étude de la stratification des PDR (Barre d'Orion).
"""

# =============================================================================
# 1. IMPORTATIONS ET CONFIGURATION
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.interpolate import UnivariateSpline

# Constantes physiques globales
VITESSE_C = 299792.458  # Vitesse de la lumière en km/s


# =============================================================================
# 2. FONCTIONS OUTILS INSTRUMENTALES ET GÉOMÉTRIQUES DE BASE
# =============================================================================

def get_pixel_scale(wcs):
    """Extrait l'échelle de plaque spatiale (en arcsec/pixel) de manière robuste."""
    try:
        return abs(wcs.to_header()['CDELT2']) * 3600.0
    except Exception:
        return 0.06

def dessiner_ticks_fente(ax, x_ref, y_ref, ux, uy, px_per_arcsec, shape, max_arcsec=40, step=1, color='red'):
    """
    Trace des repères perpendiculaires gradués à chaque arcseconde (comme une règle).
    Le texte n'est affiché que toutes les 5 arcsecondes pour ne pas surcharger.
    """
    h, w = shape

    # On balaie de -max_arcsec à +max_arcsec par pas de 1 arcseconde
    for d in range(-int(max_arcsec), int(max_arcsec) + 1, step):
        # Position du tick en pixels depuis l'intersection (0'')
        x_tick = x_ref + (d * px_per_arcsec * ux)
        y_tick = y_ref + (d * px_per_arcsec * uy)
        
        if 0 <= x_tick < w and 0 <= y_tick < h:
            # Vecteur normal unitaire pour l'orientation du tick
            nx, ny_v = -uy, ux
            
            # RÈGLE SPÉCIFIQUE : Les ticks principaux (tous les 5") sont plus longs que les ticks secondaires (tous les 1")
            if d % 5 == 0:
                taille_tick = 3.5  # Grand trait pour les multiples de 5
                lw_tick = 1.2
            else:
                taille_tick = 1.8  # Petit trait pour les arcsecondes intermédiaires
                lw_tick = 0.7
            
            # Dessin du trait de la règle
            ax.plot([x_tick - taille_tick*nx, x_tick + taille_tick*nx], 
                    [y_tick - taille_tick*ny_v, y_tick + taille_tick*ny_v], 
                    color=color, lw=lw_tick, zorder=20)
            
            # AFFICHAGE DES CHIFFRES UNIQUEMENT TOUTES LES 5 ARCSECONDES
            if d % 5 == 0:
                if d == 0:
                    txt = "IF (0'')"
                else:
                    txt = f"{d:+}''" if d > 0 else f"{d}''"
                    
                ax.text(x_tick + 6*nx, y_tick + 6*ny_v, txt, 
                        color=color, fontsize=8, weight='bold', 
                        ha='center', va='center', zorder=21)

def calculer_axe_av_physique(dist_arcsec, profil_reference=None, av_max_pdr=15.0):
    """
    Calcule un profil d'Av cumulé robuste et physique le long de la coupe.
    Sécurisé contre les fentes commençant pile à l'IF (dist_arcsec >= 0).
    """
    if profil_reference is not None and np.nanmax(profil_reference) > 0:
        # --- MÉTHODE EMPIRIQUE PONDÉRÉE PAR LE FLUX ---
        masque_avant = dist_arcsec < 0
        masque_apres = dist_arcsec >= 0
        
        av_profil = np.zeros_like(dist_arcsec)
        
        # Sécurité : On ne calcule le profil avant que s'il y a des points valides
        if np.any(masque_avant):
            av_profil[masque_avant] = (dist_arcsec[masque_avant] - dist_arcsec.min()) * 0.2
            valeur_depart = av_profil[masque_avant].max()
        else:
            valeur_depart = 0.0
        
        # Après l'IF, on intègre le profil cumulé du gaz (colonne de densité N(H))
        profil_apres = np.maximum(0, profil_reference[masque_apres])
        flux_cumule = np.cumsum(profil_apres)
        
        if flux_cumule[-1] > 0:
            flux_cumule_norm = flux_cumule / flux_cumule[-1]
            av_profil[masque_apres] = valeur_depart + flux_cumule_norm * av_max_pdr
        else:
            av_profil[masque_apres] = valeur_depart
            
        return av_profil
    else:
        # --- MÉTHODE PHYSIQUE PAR COMPRESSION (PROFIL MODÈLE) ---
        av_profil = np.zeros_like(dist_arcsec)
        masque_avant = dist_arcsec < 0
        
        valeur_depart = 0.0
        if np.any(masque_avant):
            valeur_depart = 0.15 * (-dist_arcsec.min())
            
        for i, d in enumerate(dist_arcsec):
            if d < 0:
                av_profil[i] = max(0.1, 0.15 * (d - dist_arcsec.min()))
            else:
                av_profil[i] = valeur_depart + 1.2 * d * (1.0 + (d / 12.0)**1.5)
        return av_profil

# =============================================================================
# 3. CATALOGUE SYNCHRONISÉ ET SEPARATION DES FAMILLES DE RAIES
# =============================================================================

catalogue_raies = {
    # Transitions purement rotationnelles du H2 (v=0-0)
    "H2_00S8":    {"val": 5.0531,    "label": r"$H_2$ 0-0 S(8)",   "type": "H2_v0"}, 
    "H2_00S9":    {"val": 4.6946,    "label": r"$H_2$ 0-0 S(9)",    "type": "H2_v0"},
    "H2_00S10":   {"val": 4.4098,    "label": r"$H_2$ 0-0 S(10)",  "type": "H2_v0"}, 
    
    # Branche ro-vibrationnelle v=1-0 du H2
    "H2_10S0":    {"val": 2.2233,    "label": r"$H_2$ 1-0 S(0)",   "type": "H2_v1"}, 
    "H2_10S1":    {"val": 2.1218,    "label": r"$H_2$ 1-0 S(1)",   "type": "H2_v1"}, 
    "H2_10S2":    {"val": 2.0338,    "label": r"$H_2$ 1-0 S(2)",   "type": "H2_v1"}, 
    "H2_10S3":    {"val": 1.9576,    "label": r"$H_2$ 1-0 S(3)",   "type": "H2_v1"}, 
    
    # Branche ro-vibrationnelle v=2-1 du H2
    "H2_21S1":    {"val": 2.2477,    "label": r"$H_2$ 2-1 S(1)",   "type": "H2_v2"}, 
    "H2_21S2":    {"val": 2.1542,    "label": r"$H_2$ 2-1 S(2)",   "type": "H2_v2"}, 
    "H2_21S3":    {"val": 2.0732,    "label": r"$H_2$ 2-1 S(3)",   "type": "H2_v2"}, 
    
    # Diagramme V1 : Transitions CO Fondamentales v=1-0 (Branche R et P)
    "CO_10_P9":   {"val": 4.7504675, "label": "CO 1-0 P(9)",       "type": "CO_10"},
    "CO_10_P10":  {"val": 4.7545019, "label": "CO 1-0 P(10)",      "type": "CO_10"}, 
    "CO_10_P11":  {"val": 4.7586561, "label": "CO 1-0 P(11)",      "type": "CO_10"},
    "CO_10_P12":  {"val": 4.7629312, "label": "CO 1-0 P(12)",      "type": "CO_10"},
    "CO_10_P13":  {"val": 4.7673291, "label": "CO 1-0 P(13)",      "type": "CO_10"},
    "CO_10_R1":   {"val": 4.6573130, "label": "CO 1-0 R(1)",       "type": "CO_10"},
    "CO_10_R3":   {"val": 4.6493011, "label": "CO 1-0 R(3)",       "type": "CO_10"},
    "CO_10_R4":   {"val": 4.6454747, "label": "CO 1-0 R(4)",       "type": "CO_10"},
    
    # Diagramme V2 : Transitions CO Excitées v=2-1 (Branche R et P)
    "CO_21_P4":   {"val": 4.7588627, "label": "CO 2-1 P(4)",       "type": "CO_21"},
    "CO_21_P5":   {"val": 4.7629471, "label": "CO 2-1 P(5)",       "type": "CO_21"},
    "CO_21_P6":   {"val": 4.7671549, "label": "CO 2-1 P(6)",       "type": "CO_21"},
    "CO_21_P8":   {"val": 4.7759840, "label": "CO 2-1 P(8)",       "type": "CO_21"},
    "CO_21_R7":   {"val": 4.6549511, "label": "CO 2-1 R(7)",       "type": "CO_21"},
    "CO_21_R9":   {"val": 4.6453912, "label": "CO 2-1 R(9)",       "type": "CO_21"},
    "CO_21_R10":  {"val": 4.6408226, "label": "CO 2-1 R(10)",      "type": "CO_21"},
    "CO_21_R11":  {"val": 4.6364239, "label": "CO 2-1 R(11)",      "type": "CO_21"},
    "CO_21_R12":  {"val": 4.6321946, "label": "CO 2-1 R(12)",      "type": "CO_21"}
}


def trier_raies_par_familles(liste_cles_brutes):
    """Analyse les clés de raies disponibles et les sépare par groupes physiques triés par J."""
    raies_h2, raies_co_v1, raies_co_v2 = [], [], []
    for raie in liste_cles_brutes:
        if "H2" in raie:
            try:
                j_val = int(''.join(filter(str.isdigit, raie.split('S')[-1])))
                raies_h2.append((raie, j_val))
            except ValueError: raies_h2.append((raie, 0))
        elif "CO_10" in raie:
            try:
                j_val = int(''.join(filter(str.isdigit, raie.split('_')[-1])))
                raies_co_v1.append((raie, j_val))
            except ValueError: raies_co_v1.append((raie, 0))
        elif "CO_21" in raie:
            try:
                j_val = int(''.join(filter(str.isdigit, raie.split('_')[-1])))
                raies_co_v2.append((raie, j_val))
            except ValueError: raies_co_v2.append((raie, 0))

    raies_h2.sort(key=lambda x: x[1])
    raies_co_v1.sort(key=lambda x: x[1])
    raies_co_v2.sort(key=lambda x: x[1])
    return raies_h2, raies_co_v1, raies_co_v2


# =============================================================================
# 4. SPECTRO-CINÉMATIQUE ET EXTRACTIONS DE LOGICIELS
# =============================================================================

def integrer_raie_cinematique(cube, lambdas_arr, lambda_0, v_min=None, v_max=None, v_lsr=0.0):
    """Isole et intègre le flux d'une raie spectrale par la méthode des trapèzes."""
    lambda_corrigee = lambda_0 * (1.0 + v_lsr / VITESSE_C)
    
    if v_min is None or v_max is None:
        idx_z = np.argmin(np.abs(lambdas_arr - lambda_corrigee))
        return cube[idx_z, :, :]
        
    vitesses = VITESSE_C * (lambdas_arr - lambda_corrigee) / lambda_corrigee
    idx_plage = np.where((vitesses >= v_min) & (vitesses <= v_max))[0]
    
    if idx_plage.size == 0:
        idx_plage = np.array([np.argmin(np.abs(lambdas_arr - lambda_corrigee))])
        
    cube_fenetre = cube[idx_plage, :, :]
    lambdas_fenetre = lambdas_arr[idx_plage]
    
    if lambdas_fenetre.size > 1:
        return np.trapz(cube_fenetre, x=lambdas_fenetre, axis=0)
    return cube_fenetre[0, :, :]


def calculer_carte_moyenne_raies(liste_cles, catalogue, cube, lambdas_arr, obs_module, v_min=None, v_max=None, v_lsr=0.0):
    """Calcule la moyenne des cartes d'intensité (brutes ou cinématiques) pour un groupe."""
    cartes_2d = []
    for cle in liste_cles:
        if cle in catalogue:
            carte_indiv = integrer_raie_cinematique(cube, lambdas_arr, catalogue[cle]["val"], v_min=v_min, v_max=v_max, v_lsr=v_lsr)
            if carte_indiv is not None and not np.all(np.isnan(carte_indiv)):
                cartes_2d.append(carte_indiv)
    if not cartes_2d:
        return None
    return np.mean(cartes_2d, axis=0)


def extraire_profils_bruts_3d(data_cube, lambdas_arr, catalogue, liste_cles, x_pts, y_pts, v_min=None, v_max=None, v_lsr=0.0):
    """Extraction à fente épaisse (boîte 3x3) sur grille native unifiée."""
    ny, nx = data_cube.shape[1], data_cube.shape[2]
    profils_extraits = {}

    print("="*65)
    print("EXTRACTION DU PROFIL SANS REDONDANCE ET COHÉRENCE GÉOMÉTRIQUE")
    print("="*65)

    for raie in liste_cles:
        if raie not in catalogue: continue
        carte_2d = integrer_raie_cinematique(data_cube, lambdas_arr, catalogue[raie]["val"], v_min=v_min, v_max=v_max, v_lsr=v_lsr)
        
        profil_courant = []
        for x, y in zip(x_pts, y_pts):
            ix, iy = int(np.round(x)), int(np.round(y))
            y_min, y_max = max(0, iy - 1), min(ny, iy + 2) #Moyenne sur une fente de 3pixels par 3 pixels le long de la coupe 
            x_min, x_max = max(0, ix - 1), min(nx, ix + 2)
            bloc = carte_2d[y_min:y_max, x_min:x_max]
            
            if bloc.size > 0 and not np.all(np.isnan(bloc)):
                profil_courant.append(np.nanmean(bloc))
            else:
                profil_courant.append(np.nan)
                
        profil_courant = np.array(profil_courant)
        indices_valides = np.where(~np.isnan(profil_courant))[0]
        if indices_valides.size > 0:
            profil_courant = np.interp(np.arange(len(profil_courant)), indices_valides, profil_courant[indices_valides])
        else:
            profil_courant = np.zeros_like(profil_courant)
        profils_extraits[raie] = profil_courant
        
    return profils_extraits


# =============================================================================
# 5. CARTOGRAPHIE ET FIGURES JUMELLES SÉCURISÉES
# =============================================================================

def calculer_trajectoire_coupe(wcs_obj, coord_start, coord_end, coord_IF_ref, max_arcsec, pixel_scale):
    """Calcule l'échantillonnage de la trajectoire linéaire projetée de l'IF vers l'arrière."""
    x_start, y_start = wcs_obj.world_to_pixel(coord_start)
    x_end, y_end = wcs_obj.world_to_pixel(coord_end)
    x_IF, y_IF = wcs_obj.world_to_pixel(coord_IF_ref)

    dx, dy = x_end - x_start, y_end - y_start
    norme = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / norme, dy / norme

    prod_scal = (x_IF - x_start) * ux + (y_IF - y_start) * uy
    x_IF_proj = x_start + prod_scal * ux
    y_IF_proj = y_start + prod_scal * uy

    pas_pixels = np.linspace(0.0, max_arcsec / pixel_scale, 450)
    return x_IF_proj + pas_pixels * ux, y_IF_proj + pas_pixels * uy, pas_pixels


def tracer_multi_contours_percentiles(ax, carte_data, niveaux_pc, couleur, label_groupe, elements_legende):
    """Génère un lot de contours progressifs et peuple la légende."""
    if carte_data is None: return
    donnees_valides = carte_data[~np.isnan(carte_data)]
    if donnees_valides.size == 0: return
        
    valeurs_flux = [np.percentile(donnees_valides, pc) for pc in niveaux_pc]
    epaisseurs = np.linspace(1.0, 2.3, len(niveaux_pc))
    ax.contour(carte_data, levels=valeurs_flux, colors=couleur, linewidths=epaisseurs, zorder=8)
    elements_legende.append(Line2D([0], [0], color=couleur, lw=1.8, label=label_groupe))


def configurer_axes_superieurs(ax_base, dist_av_z, dist_arcsec_z, dist_pc_z):
    """Synchronise et verrouille géométriquement les doubles échelles supérieures."""
    facteur_av = dist_arcsec_z[-1] / dist_av_z[-1] if dist_av_z[-1] != 0 else 1.0
    facteur_pc = dist_pc_z[-1] / dist_arcsec_z[-1] if dist_arcsec_z[-1] != 0 else 1.0

    ax_arcsec = ax_base.twiny()
    ax_arcsec.set_xlim(dist_av_z.min() * facteur_av, dist_av_z.max() * facteur_av)
    ax_arcsec.set_xlabel(r"Angular Scale (arcsec) at 414.0 pc", fontsize=11, weight='bold')

    ax_pc = ax_base.twiny()
    ax_pc.spines['top'].set_position(('outward', 45))
    ax_pc.set_xlim(dist_av_z.min() * facteur_av * facteur_pc, dist_av_z.max() * facteur_av * facteur_pc)
    ax_pc.set_xlabel(r"Physical Distance (pc)", fontsize=11, weight='bold', color='darkred')
    ax_pc.tick_params(axis='x', colors='darkred')
    return ax_arcsec, ax_pc


def injecter_colorbars_verticales(fig, cmap_h2, cmap_v1, cmap_v2, norm_v1, norm_v2, vmin_h2, vmax_h2):
    """Génère le triplet de barres colorées compactes en marge droite."""
    cax1 = fig.add_axes([0.88, 0.62, 0.02, 0.16])
    fig.colorbar(cm.ScalarMappable(cmap=cmap_h2, norm=Normalize(vmin=vmin_h2, vmax=vmax_h2)), cax=cax1).set_label(r"Niveau d'excitation $H_2$ (S(J))", fontsize=9)
    
    cax2 = fig.add_axes([0.88, 0.37, 0.02, 0.16])
    fig.colorbar(cm.ScalarMappable(cmap=cmap_v1, norm=norm_v1), cax=cax2).set_label(r"Niveau de rotation $J$ ($v=1 \rightarrow 0$)", fontsize=9)
    
    cax3 = fig.add_axes([0.88, 0.12, 0.02, 0.16])
    fig.colorbar(cm.ScalarMappable(cmap=cmap_v2, norm=norm_v2), cax=cax3).set_label(r"Niveau de rotation $J$ ($v=2 \rightarrow 1$)", fontsize=9)
    
    
def injecter_colorbars_verticales_externes(fig, cmap_h2, cmap_v1, cmap_v2, norm_v1, norm_v2, vmin_h2, vmax_h2):
    """Génère le triplet de barres colorées de manière strictement externe sur la droite."""
    # En élargissant la boîte globale via subplots_adjust, ces coordonnées 
    # placent les axes des colorbars dans la marge blanche externe droite.
    cax1 = fig.add_axes([0.93, 0.65, 0.015, 0.20])
    fig.colorbar(cm.ScalarMappable(cmap=cmap_h2, norm=Normalize(vmin=vmin_h2, vmax=vmax_h2)), cax=cax1).set_label(r"$H_2$ Excitation $S(J)$", fontsize=9)
    
    cax2 = fig.add_axes([0.93, 0.38, 0.015, 0.20])
    fig.colorbar(cm.ScalarMappable(cmap=cmap_v1, norm=norm_v1), cax=cax2).set_label(r"CO $v=1 \rightarrow 0$ ($J$)", fontsize=9)
    
    cax3 = fig.add_axes([0.93, 0.11, 0.015, 0.20])
    fig.colorbar(cm.ScalarMappable(cmap=cmap_v2, norm=norm_v2), cax=cax3).set_label(r"CO $v=2 \rightarrow 1$ ($J$)", fontsize=9)