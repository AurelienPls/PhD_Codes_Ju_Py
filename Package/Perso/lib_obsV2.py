#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib_obs.py - Bibliothèque d'analyse spatiale, spectrale et géométrique pour cubes JWST.
Centralise l'intégralité des outils d'imagerie, de découpage et d'extraction 1D.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import Regions

# =========================================================================
# I. ENTRÉES / SORTIES & NAVIGATION SPECTRO-SPATIALE
# =========================================================================

def charger_fits(chemin_fits):
    """
    Charge un cube de données FITS et extrait ses composantes spatiales et spectrales.
    Sécurisé avec fobj=hdul pour éviter le crash des coordonnées tabulées (-TAB).
    """
    with fits.open(chemin_fits) as hdul:
        ext = 'SCI' if 'SCI' in hdul else 0
        hdr = hdul[ext].header
        data = hdul[ext].data
        
        # Initialisation robuste de la WCS (liaison au fichier ouverte pour les tables -TAB)
        wcs = WCS(hdr, fobj=hdul)
        unite_flux = hdr.get("BUNIT", "flux_unit")
        
        # Reconstruction du vecteur des longueurs d'onde (conversion systématique en micromètres)
        if wcs.naxis >= 3:
            idx_wave = 2
            n_wave = data.shape[0]
            crpix = wcs.wcs.crpix[idx_wave]
            crval = wcs.wcs.crval[idx_wave]
            cdelt = wcs.wcs.cdelt[idx_wave]
            cunit = wcs.wcs.cunit[idx_wave]
            
            lambdas = crval + (np.arange(n_wave) + 1 - crpix) * cdelt
            if str(cunit).lower() in ["m", "meters"]:
                lambdas *= 1e6  
        else:
            lambdas = np.array([0.0])
            
    return data, wcs, lambdas, unite_flux


def calculer_carte_moyenne_raies(liste_cles, catalogue, cube, wcs_cube, lambdas_arr, obs_module, methode='lambda', lambda_width=0.005):
    """
    Extrait et calcule la moyenne d'intensité pixel par pixel pour une liste de raies.
    Fait le pont avec votre module de base 'obs_DS9' pour l'extraction 2D.
    """
    cartes_2d = []
    for cle in liste_cles:
        if cle in catalogue:
            if methode == 'lambda':
                # Approche standard de CubeStudy
                carte_indiv, _, _ = obs_module.obtenir_carte_2d(
                    cube, wcs_cube, lambdas_arr, 
                    methode='lambda', lambda_cible=catalogue[cle]["val"]
                )
            elif methode == 'mean_band':
                # Approche par intégration de bande étroite
                l_cible = catalogue[cle]["val"]
                carte_indiv, _, _ = obs_module.obtenir_carte_2d(
                    cube, wcs_cube, lambdas_arr, methode='mean_band', 
                    lambda_min=l_cible - lambda_width, lambda_max=l_cible + lambda_width
                )
                
            if carte_indiv is not None and not np.all(np.isnan(carte_indiv)):
                cartes_2d.append(carte_indiv)
        else:
            print(f"[Attention] La raie '{cle}' est introuvable dans le catalogue.")
            
    if not cartes_2d:
        return None
    return np.mean(cartes_2d, axis=0)


def croper_autour_region_2d(carte_2d, wcs_2d, chemin_region, nom_region, padding=12):
    """
    Découpe une carte 2D et sa WCS associée autour des limites d'une région DS9 (.reg).
    Version optimisée sans besoin d'injecter une dimension spectrale artificielle.
    """
    regs = Regions.read(chemin_region, format='ds9')
    reg_cible = None
    for r in regs:
        if r.meta.get('text') == nom_region or nom_region is None:
            reg_cible = r
            break
            
    if reg_cible is None:
        return carte_2d.copy(), wcs_2d.deepcopy()
        
    mask = reg_cible.to_pixel(wcs_2d)
    bbox = mask.bounding_box
    
    ymin = max(0, bbox.ymin - padding)
    ymax = min(carte_2d.shape[0], bbox.ymax + padding)
    xmin = max(0, bbox.xmin - padding)
    xmax = min(carte_2d.shape[1], bbox.xmax + padding)
    
    carte_crop = carte_2d[ymin:ymax, xmin:xmax]
    wcs_crop = wcs_2d.deepcopy()
    wcs_crop.wcs.crpix[0] -= xmin
    wcs_crop.wcs.crpix[1] -= ymin
    
    return carte_crop, wcs_crop

# =========================================================================
# II. GÉOMÉTRIE DE COUPE ET MASQUES TOPOLOGIQUES
# =========================================================================

def extraire_geometrie_coupe(wcs_2d, coord_start, coord_end, coord_IF_ref):
    """
    Calcule la matrice de projection de la fente (pixels, vecteurs directeurs, 
    vecteurs orthogonaux et projection stricte de l'IF).
    """
    x_start, y_start = wcs_2d.world_to_pixel(coord_start)
    x_end, y_end = wcs_2d.world_to_pixel(coord_end)
    x_IF, y_IF = wcs_2d.world_to_pixel(coord_IF_ref)
    
    dx, dy = x_end - x_start, y_end - y_start
    norm = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux  # Vecteur perpendiculaire le long du front
    
    # Calcul du point d'impact orthogonal de l'IF sur la fente
    dot_IF = (x_IF - x_start) * ux + (y_IF - y_start) * uy
    x_IF_proj = x_start + dot_IF * ux
    y_IF_proj = y_start + dot_IF * uy
    
    return (x_start, y_start, x_end, y_end, x_IF_proj, y_IF_proj, ux, uy, px, py)


def generer_masques_spatial_if(shape, x_IF_proj, y_IF_proj, ux, uy):
    """
    Sépare géométriquement l'image en deux domaines étanches par rapport 
    à la ligne d'impact du front d'ionisation.
    """
    ny, nx = shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    dist_from_IF = (X - x_IF_proj) * ux + (Y - y_IF_proj) * uy
    
    masque_avant = dist_from_IF < 0   # Zone HII / Milieu atomique amont
    masque_apres = dist_from_IF >= 0  # PDR Neutre / Nuage moléculaire profond
    return masque_avant, masque_apres


def get_pixel_scale(wcs_2d):
    """Extrait proprement l'échelle de plaque en arcsecondes par pixel depuis la WCS."""
    try:
        return abs(wcs_2d.to_header()['CDELT2']) * 3600.0
    except Exception:
        return 0.06 # Valeur de repli NIRSpec standard

# =========================================================================
# III. EXTRACTION DE PROFILS 1D (PIXEL & COORDONNÉES CÉLESTES)
# =========================================================================

def extraire_profil_pixel(data, x_points, y_points):
    """Extrait les intensités brutes le long d'une liste de coordonnées de pixels (x, y)."""
    intensites = []
    for x, y in zip(x_points, y_points):
        x_idx, y_idx = int(np.round(x)), int(np.round(y))
        if 0 <= y_idx < data.shape[0] and 0 <= x_idx < data.shape[1]:
            intensites.append(data[y_idx, x_idx])
        else:
            intensites.append(np.nan)
    return np.array(intensites)


def generer_fente_celeste(coord_if, coord_end, num_points=400, max_arcsec=30.0):
    """Génère un échantillonnage régulier de coordonnées SkyCoord le long de la fente depuis l'IF."""
    pa = coord_if.position_angle(coord_end)
    arcsec_axe = np.linspace(0.0, max_arcsec, num_points)
    coords_fente = [coord_if.directional_offset_by(pa, d * u.arcsec) for d in arcsec_axe]
    return arcsec_axe, coords_fente


def extraire_profil_celeste_normalise(carte_data, wcs_native, coords_celestes):
    """
    Extrait le signal d'une carte native via coordonnées mondiales (RA/DEC)
    et applique une normalisation sur le maximum local nettoyé.
    """
    profil = []
    for sky_c in coords_celestes:
        try:
            x_px, y_px = wcs_native.world_to_pixel(sky_c)
            xi, yi = int(np.round(x_px)), int(np.round(y_px))
            if 0 <= yi < carte_data.shape[0] and 0 <= xi < carte_data.shape[1]:
                val = carte_data[yi, xi]
                profil.append(val if not np.isnan(val) else 0.0)
            else:
                profil.append(0.0)
        except Exception:
            profil.append(0.0)
            
    profil = np.array(profil)
    idx_valides = np.where(profil > 0.0)[0]
    if idx_valides.size > 0:
        profil = np.interp(np.arange(len(profil)), idx_valides, profil[idx_valides])
        
    max_p = np.max(profil)
    return profil / (max_p if max_p > 0 else 1.0)

# =========================================================================
# IV. HABILLAGE ET MOTEUR GRAPHIQUE ADVANCED
# =========================================================================

def superposer_regions_ds9(ax, wcs_2d, chemin_region, nom_cible=None, couleur='white'):
    """Décode et superpose les contours géométriques d'un fichier de régions .reg."""
    try:
        regs = Regions.read(chemin_region, format='ds9')
        for r in regs:
            text = r.meta.get('text', '')
            if nom_cible is None or text == nom_cible:
                reg_pixel = r.to_pixel(wcs_2d)
                reg_pixel.plot(ax=ax, edgecolor=couleur, facecolor='none', lw=1.5, zorder=10)
                if text:
                    cen = reg_pixel.center if hasattr(reg_pixel, 'center') else reg_pixel.vertices[0]
                    ax.text(cen.x, cen.y, text, color=couleur, weight='bold', fontsize=9, ha='center', va='center', zorder=11)
    except Exception as e:
        print(f"[Erreur Régions] Impossible de tracer : {e}")


def ajouter_echelle_au_v2(ax, carte_2d, distance_pc=414.0, taille_au=200.0, arcsec_per_pixel=0.1, couleur='white', position='lower right'):
    """Affiche une barre d'échelle physique pure, exempte de boîtes opaques parasites."""
    taille_arcsec = taille_au / distance_pc
    longueur_pixels = taille_arcsec / arcsec_per_pixel
    ny, nx = carte_2d.shape
    
    if 'right' in position:
        x_fin = nx - (nx * 0.08)
        x_deb = x_fin - longueur_pixels
    else:
        x_deb = nx * 0.08
        x_fin = x_deb + longueur_pixels
        
    if 'upper' in position:
        y_pos = ny - (ny * 0.08)
        y_offset_text = -(ny * 0.02)
        va = 'top'
    else:
        y_pos = ny * 0.06
        y_offset_text = ny * 0.02
        va = 'bottom'
        
    ax.plot([x_deb, x_fin], [y_pos, y_pos], color=couleur, linewidth=2.0, zorder=12)
    ax.text((x_deb + x_fin) / 2.0, y_pos + y_offset_text, f"{int(taille_au)} AU", 
            color=couleur, weight='bold', fontsize=10, ha='center', va=va, zorder=12)


def dessiner_ticks_image(ax, x_IF_proj, y_IF_proj, ux, uy, px, py, px_per_arcsec, shape, max_arcsec=30, step=5):
    """Génère les graduations et étiquettes de distance en arcsecondes le long de la coupe fente."""
    taille_tick = 3
    for d in range(0, int(max_arcsec) + 1):
        x_tick = x_IF_proj + d * px_per_arcsec * ux
        y_tick = y_IF_proj + d * px_per_arcsec * uy
        if 0 <= x_tick < shape[1] and 0 <= y_tick < shape[0]:
            ax.plot([x_tick - taille_tick*px, x_tick + taille_tick*px], 
                    [y_tick - taille_tick*py, y_tick + taille_tick*py], color='red', lw=1.0, zorder=11)
            if d % step == 0 and d != 0:
                ax.text(x_tick + 6*px, y_tick + 6*py, f"{d}''", color='white', fontsize=8, weight='bold',
                        ha='center', va='center', zorder=12, bbox=dict(facecolor='black', alpha=0.5, pad=0.5, edgecolor='none'))


def tracer_multi_contours_percentiles(ax, carte_data, niveaux_pc, couleur, label_groupe, elements_legende):
    """Génère des contours d'isophotes d'intensités basés sur des percentiles sans NaNs."""
    if carte_data is None:
        return
    carte_propre = carte_data[~np.isnan(carte_data)]
    if carte_propre.size == 0:
        return
        
    valeurs_flux = [np.percentile(carte_propre, pc) for pc in niveaux_pc]
    epaisseurs = np.linspace(1.0, 2.3, len(niveaux_pc))
    
    ax.contour(carte_data, levels=np.sort(valeurs_flux), colors=couleur, linewidths=epaisseurs, zorder=8)
    if elements_legende is not None:
        elements_legende.append(Line2D([0], [0], color=couleur, lw=2, label=label_groupe))

# =========================================================================
# V. STATISTIQUES ET PROFILAGE CELLULAIRE
# =========================================================================

def extraire_et_tracer_histogramme_comparatif(obs_module, cube, wcs, lambdas_arr, chemin_region, nom_region, 
                                              cle_raie1, cle_raie2, catalogue, unite_flux):
    """Isole les pixels d'une région DS9 pour deux raies et dresse un histogramme comparatif log Astro-Dark."""
    carte_raie1, _, _ = obs_module.obtenir_carte_2d(cube, wcs, lambdas_arr, methode='lambda', lambda_cible=catalogue[cle_raie1]["val"])
    carte_raie2, _, _ = obs_module.obtenir_carte_2d(cube, wcs, lambdas_arr, methode='lambda', lambda_cible=catalogue[cle_raie2]["val"])
    
    masque = obs_module.obtenir_masque_region(carte_raie1.shape, wcs, chemin_region, nom_region)
    pixels_r1 = carte_raie1[masque]
    pixels_r2 = carte_raie2[masque]
    
    pixels_r1 = pixels_r1[~np.isnan(pixels_r1)]
    pixels_r2 = pixels_r2[~np.isnan(pixels_r2)]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pixels_r1, bins=30, alpha=0.6, color='magenta', label=catalogue[cle_raie1]['label'])
    ax.hist(pixels_r2, bins=30, alpha=0.6, color='cyan', label=catalogue[cle_raie2]['label'])
    
    ax.set_title(f"Distribution des intensités dans {nom_region}")
    ax.set_xlabel(f"Intensité ({unite_flux})")
    ax.set_ylabel("Nombre de pixels")
    ax.set_yscale('log')
    
    # Harmonisation Astro Dark Style
    ax.set_facecolor('#1c1c1c')
    fig.patch.set_facecolor('#1c1c1c')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(colors='white')
    ax.title.set_color('white')
    ax.legend(facecolor='black', labelcolor='white')
    ax.grid(True, linestyle=':', alpha=0.3, color='gray')
    plt.tight_layout()
    plt.show()


def configurer_axe_ua(ax_base, conversion_factor, ticks_step=5.0, label="Distance Physique (UA)"):
    """Génère et synchronise un axe supérieur en unités physiques réelles (UA / pc)."""
    ax_phys = ax_base.twiny()
    ax_phys.set_xlim(ax_base.get_xlim())
    x_min, x_max = ax_base.get_xlim()
    ticks = np.arange(np.ceil(x_min), np.floor(x_max) + 1, ticks_step)
    ax_phys.set_xticks(ticks)
    ax_phys.set_xticklabels([f"{t * conversion_factor:.1f}" if conversion_factor < 1 else f"{int(t * conversion_factor)}" for t in ticks])
    ax_phys.set_xlabel(label, fontsize=10, color='darkblue', weight='bold', labelpad=10)
    return ax_phys