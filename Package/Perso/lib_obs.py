#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:17:43 2026

@author: aurelien
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bibliothèque d'analyse spatiale et spectrale pour cubes JWST/NIRSpec.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def calculer_carte_moyenne_raies(liste_cles, catalogue, cube, wcs, lambdas_arr, obs_module):
    """
    Extrait et calcule la moyenne d'intensité pixel par pixel pour une liste de raies
    définie à partir d'un catalogue d'astrophysique.
    """
    cartes_2d = []
    for cle in liste_cles:
        if cle in catalogue:
            carte_indiv, _, _ = obs_module.obtenir_carte_2d(
                cube, wcs, lambdas_arr, 
                methode='lambda', lambda_cible=catalogue[cle]["val"]
            )
            if carte_indiv is not None and not np.all(np.isnan(carte_indiv)):
                cartes_2d.append(carte_indiv)
        else:
            print(f"[Attention] La raie '{cle}' est introuvable dans le catalogue.")
            
    if not cartes_2d:
        return None
    return np.mean(cartes_2d, axis=0)


def ajouter_echelle_au_v2(ax, carte_2d, distance_pc=414.0, taille_au=200.0, arcsec_per_pixel=0.1, couleur='red', position='lower right'):
    """
    Fonction V2 d'affichage d'échelle physique. Calcule la taille en pixels réels 
    à partir de l'échelle de plaque instrumentale (0.1"/px par défaut pour NIRSpec).
    Garantit l'absence d'artefacts ou de boîtes opaques sur la figure.
    """
    # Angle sous-tendu sur la sphère céleste : alpha (arcsec) = Taille (AU) / Distance (pc)
    taille_arcsec = taille_au / distance_pc
    longueur_pixels = taille_arcsec / arcsec_per_pixel
    
    ny, nx = carte_2d.shape
    
    # Détermination des ancrages graphiques selon la position demandée
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
        
    # Tracé de la barre physique et du texte associé
    ax.plot([x_deb, x_fin], [y_pos, y_pos], color=couleur, linewidth=2.0, zorder=12)
    ax.text((x_deb + x_fin) / 2.0, y_pos + y_offset_text, f"{int(taille_au)} AU", 
            color=couleur, weight='bold', fontsize=10, ha='center', va=va, zorder=12)

def tracer_multi_contours_percentiles(ax, carte_data, niveaux_pc, couleur, label_groupe, elements_legende):
    """
    Génère un ensemble de contours basé sur des percentiles d'intensité
    et peuple dynamiquement une liste d'éléments de légende personnalisés.
    """
    # CORRECTION : Nettoyage de la syntaxe d'évaluation du None
    if carte_data is None:
        return
    
    donnees_valides = carte_data[~np.isnan(carte_data)]
    if donnees_valides.size == 0:
        return
        
    valeurs_flux = [np.percentile(donnees_valides, pc) for pc in niveaux_pc]
    
    # Linewidths progressifs pour différencier les structures denses des zones diffuses
    épaisseurs = np.linspace(1.0, 2.5, len(niveaux_pc))
    ax.contour(carte_data, levels=valeurs_flux, colors=couleur, linewidths=épaisseurs, zorder=8)
    
    elements_legende.append(Line2D([0], [0], color=couleur, lw=2, label=label_groupe))
    
    
    
def extraire_et_tracer_histogramme_comparatif(obs_module, cube, wcs, lambdas_arr, chemin_region, nom_region, 
                                              cle_raie1, cle_raie2, catalogue, unite_flux):
    """
    Isole les pixels à l'intérieur d'une région DS9 pour deux raies données,
    génère un histogramme comparatif sur échelle logarithmique et applique un style astro sombre.
    """
    # 1. Extraction des cartes 2D monochromatiques
    data_r1 = catalogue[cle_raie1]
    data_r2 = catalogue[cle_raie2]
    
    carte_raie1, _, _ = obs_module.obtenir_carte_2d(cube, wcs, lambdas_arr, methode='lambda', lambda_cible=data_r1["val"])
    carte_raie2, _, _ = obs_module.obtenir_carte_2d(cube, wcs, lambdas_arr, methode='lambda', lambda_cible=data_r2["val"])
    
    # 2. Création et application du masque de région
    masque = obs_module.obtenir_masque_region(carte_raie1.shape, wcs, chemin_region, nom_region)
    
    pixels_r1 = carte_raie1[masque]
    pixels_r2 = carte_raie2[masque]
    
    # Nettoyage des NaNs
    pixels_r1 = pixels_r1[~np.isnan(pixels_r1)]
    pixels_r2 = pixels_r2[~np.isnan(pixels_r2)]
    
    # 3. Tracé de l'histogramme avec style dark
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pixels_r1, bins=30, alpha=0.6, color='magenta', label=f"{data_r1['label']} : {data_r1['val']:.4f} $\mu$m")
    ax.hist(pixels_r2, bins=30, alpha=0.6, color='cyan', label=f"{data_r2['label']} : {data_r2['val']:.4f} $\mu$m")
    
    ax.set_title(f"Distribution des intensités dans {nom_region}")
    ax.set_xlabel(f"Intensité ({unite_flux})")
    ax.set_ylabel("Nombre de pixels")
    ax.set_yscale('log')
    ax.legend()
    
    # Harmonisation esthétique (Style Astro Dark)
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
    
def integrer_raie_moment0(cube, lambdas_arr, lambda_cible, obs_module, v_min=0.0, v_max=20.0):
    """
    Calcule la carte de Moment 0 (intensité intégrée) pour une raie donnée 
    dans une fenêtre de vitesse spécifiée [v_min, v_max] en km/s.
    
    Parameters:
    -----------
    cube : ndarray
        Le cube de données NIRSpec de forme (n_lambda, ny, nx)
    lambdas_arr : ndarray
        Le tableau des longueurs d'onde du cube en microns
    lambda_cible : float
        La longueur d'onde théorique au repos de la raie en microns
    obs_module : module
        Le module 'obs' (obs_DS9) pour respecter la structure du code
    v_min, v_max : float
        Fenêtre de vitesse Doppler en km/s (par défaut 0 à 20)
    """
    import numpy as np
    c_kms = 299792.458 # Vitesse de la lumière en km/s
    
    # 1. Conversion de l'axe spectral en vitesses relatives (Effet Doppler)
    vitesses = c_kms * (lambdas_arr - lambda_cible) / lambda_cible
    
    # 2. Identification des indices de canaux dans la fenêtre [v_min, v_max]
    indices_fenetre = np.where((vitesses >= v_min) & (vitesses <= v_max))[0]
    
    if len(indices_fenetre) == 0:
        print(f"[Attention] Aucun canal trouvé entre {v_min} et {v_max} km/s pour {lambda_cible} µm.")
        return np.zeros((cube.shape[1], cube.shape[2]))
    
    # 3. Extraction des canaux du cube
    cube_fenetre = cube[indices_fenetre, :, :]
    lambdas_fenetre = lambdas_arr[indices_fenetre]
    
    # 4. Calcul du pas spectral d_lambda pour l'intégration numérique
    if len(lambdas_fenetre) > 1:
        d_lambdas = np.gradient(lambdas_fenetre)
    else:
        d_lambdas = np.array([0.0001])
        
    # 5. Intégration : Somme pondérée par d_lambda sur l'axe spectral (axis=0)
    carte_moment0 = np.sum(cube_fenetre * d_lambdas[:, np.newaxis, np.newaxis], axis=0)
    
    return carte_moment0


def calculer_carte_moment0_moyenne_famille(liste_cles, catalogue, cube, lambdas_arr, obs_module, v_min=0.0, v_max=20.0):
    """
    Calcule la moyenne des cartes de Moment 0 pour une famille de raies donnée.
    """
    import numpy as np
    cartes_moment0 = []
    
    for cle in liste_cles:
        if cle in catalogue:
            lambda_cible = catalogue[cle]["val"]
            carte_m0 = integrer_raie_moment0(
                cube, lambdas_arr, lambda_cible, obs_module, v_min=v_min, v_max=v_max
            )
            if carte_m0 is not None and not np.all(np.isnan(carte_m0)):
                cartes_moment0.append(carte_m0)
        else:
            print(f"[Attention] La raie '{cle}' est introuvable dans le catalogue.")
            
    if not cartes_moment0:
        return None
    return np.mean(cartes_moment0, axis=0)


def calculer_trajectoire_coupe(wcs_obj, coord_start, coord_end, coord_IF_ref, max_arcsec, pixel_scale):
    """
    Calcule les coordonnées en pixels de la ligne de coupe et génère les axes physiques.
    """
    x_start, y_start = wcs_obj.world_to_pixel(coord_start)
    x_end, y_end = wcs_obj.world_to_pixel(coord_end)
    x_IF, y_IF = wcs_obj.world_to_pixel(coord_IF_ref)

    dx, dy = x_end - x_start, y_end - y_start
    norme = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / norme, dy / norme

    prod_scal = (x_IF - x_start) * ux + (y_IF - y_start) * uy
    x_IF_proj, y_IF_proj = x_start + prod_scal * ux, y_start + prod_scal * uy

    pas_pixels = np.linspace(0.0, max_arcsec / pixel_scale, 400)
    x_pts = x_IF_proj + pas_pixels * ux
    y_pts = y_IF_proj + pas_pixels * uy
    
    return x_pts, y_pts, pas_pixels


def extraire_profils_bruts_3d(data_cube, lambdas_arr, catalogue, liste_cles, x_pts, y_pts):
    """
    Extrait le flux brut du cube 3D avec une micro-tolérance spatiale (3x3) et spectrale (Z±1).
    """
    nz, ny, nx = data_cube.shape
    profils_extraits = {}

    print("="*60)
    print("EXTRACTION 3D ROBUSTE (MICRO-VOISINAGE) - DIAGNOSTIC")
    print("="*60)

    for raie in liste_cles:
        if raie not in catalogue:
            print(f"⚠️ La raie {raie} n'est pas dans le catalogue.")
            continue

        lambda_cible = catalogue[raie]["val"]
        idx_z = int(np.argmin(np.abs(lambdas_arr - lambda_cible)))
        
        profil_courant = []
        
        for x, y in zip(x_pts, y_pts):
            ix, iy = int(np.round(x)), int(np.round(y))
            
            # Micro-voisinage (bloc 3x3x3)
            z_min, z_max = max(0, idx_z - 1), min(nz, idx_z + 2)
            y_min, y_max = max(0, iy - 1), min(ny, iy + 2)
            x_min, x_max = max(0, ix - 1), min(nx, ix + 2)
            
            bloc = data_cube[z_min:z_max, y_min:y_max, x_min:x_max]
            
            if bloc.size > 0 and not np.all(np.isnan(bloc)):
                profil_courant.append(np.nanmean(bloc))
            else:
                profil_courant.append(np.nan)
                
        profil_courant = np.array(profil_courant)
        
        # Diagnostic
        nb_nans = np.isnan(profil_courant).sum()
        pct_nan = (nb_nans / len(profil_courant)) * 100
        print(f"Raie: {raie:12} | Z={idx_z:4} | NaNs purs: {nb_nans:3}/{len(profil_courant)} ({pct_nan:5.1f}%)")
        
        # Bouchage des trous résiduels par interpolation
        indices_valides = np.where(~np.isnan(profil_courant))[0]
        if indices_valides.size > 0:
            profil_courant = np.interp(np.arange(len(profil_courant)), indices_valides, profil_courant[indices_valides])
        else:
            profil_courant = np.zeros_like(profil_courant)
            
        profils_extraits[raie] = profil_courant

    return profils_extraits


def tracer_profils_stratification(distances_dict, profils_dict, config_traceurs, 
                                  normaliser=False, xmin=None, xmax=None, titre="Profils de Stratification"):
    """
    Trace les profils sélectionnés avec gestion dynamique de la normalisation et des limites.
    """
    dist_av = distances_dict["av"]
    dist_arcsec = distances_dict["arcsec"]
    dist_pc = distances_dict["pc"]
    
    facteur_av = dist_arcsec[-1] / dist_av[-1] if dist_av[-1] != 0 else 1.0
    facteur_pc = dist_pc[-1] / dist_arcsec[-1] if dist_arcsec[-1] != 0 else 1.0

    fig_p, ax_main = plt.subplots(figsize=(14, 8))
    
    valeur_max_globale = 0.0

    for raie, cfg in config_traceurs.items():
        if raie in profils_dict:
            profil = profils_dict[raie].copy()
            
            if normaliser:
                p_max = np.nanmax(profil)
                if p_max > 0:
                    profil = profil / p_max
            
            valeur_max_globale = max(valeur_max_globale, np.nanmax(profil))
            ax_main.plot(dist_av, profil, color=cfg.get("color", "white"), 
                         ls=cfg.get("style", "-"), lw=cfg.get("lw", 2.0), label=cfg.get("label", raie))

    # Repère de l'IF
    ax_main.axvline(x=0.0, color='red', lw=2, zorder=1)
    ax_main.text(0.1, valeur_max_globale * 0.5 if valeur_max_globale > 0 else 0.5, 
                 "IF", color='red', weight='bold', rotation=90, va='center')

    # Gestion de l'axe X (Av)
    x_min_plot = xmin if xmin is not None else dist_av.min()
    x_max_plot = xmax if xmax is not None else dist_av.max()
    ax_main.set_xlim(x_min_plot, x_max_plot)
    
    ax_main.set_ylim(-0.05 * valeur_max_globale, valeur_max_globale * 1.15)
    
    label_y = "Intensité Normalisée ($I/I_{max}$)" if normaliser else "Flux Brut (Unités du Cube)"
    ax_main.set_xlabel(r"Profondeur visuelle cumulative estimée ($A_V$)", fontsize=12, weight='bold', color='darkgreen')
    ax_main.set_ylabel(label_y, fontsize=12, weight='bold')
    ax_main.grid(True, ls=':', alpha=0.5)
    ax_main.tick_params(axis='x', colors='darkgreen')

    # Axe supérieur 1 : Arcsec
    ax_arcsec = ax_main.twiny()
    ax_arcsec.set_xlim(x_min_plot * facteur_av, x_max_plot * facteur_av)
    ax_arcsec.set_xlabel(r"Distance angulaire projetée (arcsec)", fontsize=11, weight='bold')

    # Axe supérieur 2 : Parsecs
    ax_pc = ax_main.twiny()
    ax_pc.spines['top'].set_position(('outward', 45))
    ax_pc.set_xlim(x_min_plot * facteur_av * facteur_pc, x_max_plot * facteur_av * facteur_pc)
    ax_pc.set_xlabel(r"Distance Physique ($pc$)", fontsize=11, weight='bold', color='darkred')
    ax_pc.tick_params(axis='x', colors='darkred')

    plt.title(titre, fontsize=14, weight='bold', pad=75)
    ax_main.legend(loc='upper right', facecolor='black', labelcolor='white', framealpha=0.9, ncol=2)
    plt.subplots_adjust(top=0.82) 
    plt.show()