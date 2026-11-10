# Fichier : obs_DS9.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, AsinhNorm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u
from regions import Regions
from scipy.ndimage import rotate
import warnings

warnings.filterwarnings('ignore', category=UserWarning, append=True)

# ==========================================
# 1. CHARGEMENT
# ==========================================
def charger_fits(fichier_fits):
    """Charge le cube FITS et les longueurs d'onde."""
    hdul = fits.open(fichier_fits)
    ext = 1 if len(hdul) > 1 else 0 
    data = hdul[ext].data
    header = hdul[ext].header
    wcs = WCS(header, fobj=hdul)
    unite_flux = header.get('BUNIT', "Intensité")
    try:
        lambdas = wcs.spectral.pixel_to_world(np.arange(data.shape[0])).to(u.micron).value
    except:
        lambdas = np.arange(data.shape[0])
    hdul.close()
    return data, wcs, lambdas, unite_flux

# ==========================================
# 2. EXTRACTION ET CROP
# ==========================================
def obtenir_carte_2d(data_cube, wcs_cube, lambdas=None, methode='mean', lambda_cible=None, lambda_min=None, lambda_max=None):
    """Génère la carte 2D de manière sécurisée."""
    titre_auto = ""
    if methode == 'mean':
        carte_2d = np.nanmean(data_cube, axis=0)
        titre_auto = "Moyenne complète"
    elif methode == 'mean_band' and lambda_min is not None and lambda_max is not None:
        idx = np.where((lambdas >= lambda_min) & (lambdas <= lambda_max))[0]
        carte_2d = np.nanmean(data_cube[idx, :, :], axis=0)
        titre_auto = f"Bandgap [{lambda_min:.3f}-{lambda_max:.3f} µm]"
    elif methode == 'lambda' and lambda_cible is not None:
        idx = (np.abs(lambdas - lambda_cible)).argmin()
        carte_2d = data_cube[idx, :, :]
        titre_auto = f"λ ≈ {lambdas[idx]:.3f} µm"
    else:
        raise ValueError("Paramètres d'extraction invalides.")
    
    # Sécurité WCS : Si on a déjà un WCS 2D, on le garde, sinon on extrait la partie céleste
    wcs_2d_final = wcs_cube.celestial if hasattr(wcs_cube, 'celestial') else wcs_cube
    
    return carte_2d, wcs_2d_final, titre_auto
def croper_autour_region(data_cube, wcs_cube, chemin_region, nom_cible, padding=10):
    """Découpage 100% Numpy. Ignore totalement Cutout2D et ses bugs JWST/Astropy."""
    regions_list = Regions.read(chemin_region, format='ds9')
    
    # Extraction sécurisée du WCS 2D
    wcs_2d = wcs_cube.celestial if hasattr(wcs_cube, 'celestial') else wcs_cube
        
    reg = next((r for r in regions_list if nom_cible in r.meta.get('text', '')), None)
    if reg is None: raise ValueError(f"Région '{nom_cible}' introuvable.")
    
    pixel_reg = reg.to_pixel(wcs_2d)
    bbox = pixel_reg.bounding_box
    
    # 1. Calcul strict en nombres entiers pour les indices Numpy
    x_min = int(max(0, np.floor(bbox.ixmin - padding)))
    x_max = int(min(data_cube.shape[2], np.ceil(bbox.ixmax + padding)))
    y_min = int(max(0, np.floor(bbox.iymin - padding)))
    y_max = int(min(data_cube.shape[1], np.ceil(bbox.iymax + padding)))
    
    # 2. Découpage Numpy ultra-rapide (aucune erreur WCS possible ici)
    cube_crop = data_cube[:, y_min:y_max, x_min:x_max]
    
    # 3. Décalage manuel du WCS pour qu'il reste PILE sur les bons pixels
    wcs_crop_2d = wcs_2d.deepcopy()
    wcs_crop_2d.wcs.crpix[0] -= x_min
    wcs_crop_2d.wcs.crpix[1] -= y_min
    
    return cube_crop, wcs_crop_2d

def superposer_regions(ax, wcs_2d, chemin_region, nom_cible=None, couleur_forcee='white'):
    """Trace les régions .reg en contournant le bug du 'nœud papillon' (zigzag)."""
    regions_list = Regions.read(chemin_region, format='ds9')
    for reg in regions_list:
        texte = reg.meta.get('text', '')
        if nom_cible and nom_cible not in texte: continue
        
        pixel_reg = reg.to_pixel(wcs_2d)
        
        # --- SOLUTION ANTI ZIGZAG ---
        try:
            # On force la forme en polygone et on trace les contours manuellement
            poly = pixel_reg.to_polygon()
            x = np.append(poly.vertices.x, poly.vertices.x[0]) # Fermer la boucle
            y = np.append(poly.vertices.y, poly.vertices.y[0])
            ax.plot(x, y, color=couleur_forcee, lw=1.5)
        except Exception:
            # Sécurité si la région n'est pas un rectangle classique
            pixel_reg.plot(ax=ax, color=couleur_forcee, lw=1.5)
            
        if texte:
            try:
                bbox = pixel_reg.bounding_box
                ax.text((bbox.ixmin + bbox.ixmax)/2, bbox.iymax + 3, texte, 
                        color=couleur_forcee, fontsize=10, ha='center', va='bottom', 
                        bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'))
            except: pass

# À mettre dans obs_DS9.py

# À ajouter/modifier dans obs_DS9.py

def extraire_region_tournee(data_2d, wcs_2d, chemin_region, nom_cible, padding=5):
    """
    Extrait et redresse une tranche 2D. 
    On l'utilise pour le fond ET pour chaque raie de contour.
    """
    regions_list = Regions.read(chemin_region, format='ds9')
    reg = next((r for r in regions_list if nom_cible in r.meta.get('text', '')), None)
    if not reg: return None, None
    
    pixel_reg = reg.to_pixel(wcs_2d)
    cx, cy = pixel_reg.center.x, pixel_reg.center.y
    angle_deg = pixel_reg.angle.to('deg').value
    
    # 1. Crop large avant rotation pour la performance
    marge = int(max(pixel_reg.width, pixel_reg.height) * 1.5)
    y_min, y_max = int(max(0, cy-marge)), int(min(data_2d.shape[0], cy+marge))
    x_min, x_max = int(max(0, cx-marge)), int(min(data_2d.shape[1], cx+marge))
    
    # 2. Rotation
    carte_rot = rotate(np.nan_to_num(data_2d[y_min:y_max, x_min:x_max], nan=0.0), 
                       angle_deg, reshape=True, order=1)
    carte_rot[carte_rot == 0] = np.nan
    
    # 3. Crop final
    h, w = int(pixel_reg.height) + padding*2, int(pixel_reg.width) + padding*2
    mid_y, mid_x = carte_rot.shape[0]//2, carte_rot.shape[1]//2
    crop_final = carte_rot[mid_y - h//2 : mid_y + h//2, mid_x - w//2 : mid_x + w//2]
    
    # 4. Reconstruction du WCS (RA/Dec) comme précédemment
    sky_center = wcs_2d.pixel_to_world(cx, cy)
    wcs_rot = wcs_2d.deepcopy()
    wcs_rot.wcs.crpix = [crop_final.shape[1]/2 + 0.5, crop_final.shape[0]/2 + 0.5]
    wcs_rot.wcs.crval = [sky_center.ra.deg, sky_center.dec.deg]
    theta = np.deg2rad(angle_deg)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
    if wcs_rot.wcs.has_pc(): wcs_rot.wcs.pc = np.dot(wcs_rot.wcs.pc, rot_mat)
    elif wcs_rot.wcs.has_cd(): wcs_rot.wcs.cd = np.dot(wcs_rot.wcs.cd, rot_mat)
        
    return crop_final, wcs_rot

def superposer_contours_tournes(ax, carte_contour_rot, niveaux=[95, 98], couleur='cyan', label=""):
    """
    Superpose des contours sur une image déjà tournée.
    """
    vals = np.percentile(carte_contour_rot[~np.isnan(carte_contour_rot)], niveaux)
    ax.contour(carte_contour_rot, levels=vals, colors=couleur, linewidths=1.5)
    ax.plot([], [], color=couleur, label=label)

def afficher_region_tournee(carte_rot, extent, titre="Région Redressée", unite_flux="", cmap='viridis', afficher_grille=True):
    """Affiche l'image redressée avec une grille de pixels."""
    fig, ax = plt.subplots(figsize=(10, 5))
    vmin, vmax = np.nanpercentile(carte_rot, [1, 99]) # Échelle robuste au bruit
    
    im = ax.imshow(carte_rot, origin='lower', cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
    
    ax.set_xlabel("Pixels relatifs (X)")
    ax.set_ylabel("Pixels relatifs (Y)")
    ax.set_title(titre)
    
    if afficher_grille:
        ax.grid(color='white', alpha=0.2, linestyle=':')
    
    plt.colorbar(im, label=unite_flux)
    plt.tight_layout()
    return fig, ax


def obtenir_masque_region(shape_carte, wcs_2d, chemin_region, nom_cible):
    """Créé un masque True/False strict pour isoler les pixels d'une région."""
    regions_list = Regions.read(chemin_region, format='ds9')
    reg = next((r for r in regions_list if nom_cible in r.meta.get('text', '')), None)
    pixel_reg = reg.to_pixel(wcs_2d)
    return pixel_reg.to_mask(mode='center').to_image(shape_carte) > 0

# ==========================================
# 3. AFFICHAGE ET TRACÉS
# ==========================================
def afficher_carte(carte_2d, wcs_2d, titre="", titre_auto="", cmap='viridis', echelle='log', Size=(10,8),unite_flux="", inverser_x=False, afficher_grille=False):
    """Affiche une carte avec les paramètres par défaut exigés."""
    fig = plt.figure(figsize=Size)
    ax = fig.add_subplot(111, projection=wcs_2d)
    
    if echelle == 'log':
        vmin = np.nanpercentile(carte_2d[carte_2d>0], 5) if len(carte_2d[carte_2d>0]) > 0 else 0.001
        norm = LogNorm(vmin=vmin, vmax=np.nanpercentile(carte_2d, 99.5))
    elif echelle == 'asinh':
        vmin, vmax = np.nanpercentile(carte_2d, [1, 99.5])
        norm = AsinhNorm(linear_width=0.1, vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=np.nanpercentile(carte_2d, 5), vmax=np.nanpercentile(carte_2d, 95))

    im = ax.imshow(carte_2d, cmap=cmap, origin='lower', norm=norm)
    
    if inverser_x: ax.invert_xaxis()
    if afficher_grille: ax.coords.grid(color='white', alpha=0.3, linestyle='--')
    
    ax.set_title(f"{titre} {titre_auto}".strip())
    plt.colorbar(im, ax=ax, label=unite_flux, fraction=0.046, pad=0.04)
    return fig, ax
'''
def superposer_regions(ax, wcs_2d, chemin_region, nom_cible=None, couleur='white', afficher_nom=True):
    """
    Trace les régions .reg avec option d'affichage des noms.
    
    Paramètres:
    -----------
    ax : matplotlib.axes
        L'axe sur lequel tracer.
    wcs_2d : astropy.wcs.WCS
        Le système de coordonnées 2D.
    chemin_region : str
        Chemin vers le fichier .reg.
    nom_cible : str, optional
        Si précisé, n'affiche que la région contenant ce texte.
    couleur : str, optional
        Couleur du tracé (défaut 'white').
    afficher_nom : bool, optional
        Si True, affiche le texte de la région au-dessus (défaut True).
    """
    from regions import Regions
    import numpy as np
    
    regions_list = Regions.read(chemin_region, format='ds9')
    
    for reg in regions_list:
        texte = reg.meta.get('text', '')
        # Filtrage par nom si demandé
        if nom_cible and nom_cible not in texte:
            continue
        
        # Conversion en pixels
        pixel_reg = reg.to_pixel(wcs_2d)
        
        # --- TRACÉ GÉOMÉTRIQUE (Anti-zigzag) ---
        try:
            poly = pixel_reg.to_polygon()
            x = np.append(poly.vertices.x, poly.vertices.x[0])
            y = np.append(poly.vertices.y, poly.vertices.y[0])
            ax.plot(x, y, color=couleur, lw=1.5)
        except Exception:
            pixel_reg.plot(ax=ax, color=couleur, lw=1.5)
            
        # --- ANNOTATIONS (Conditionnelles) ---
        if afficher_nom and texte:
            try:
                bbox = pixel_reg.bounding_box
                ax.text((bbox.ixmin + bbox.ixmax)/2, bbox.iymax + 3, texte, 
                        color=couleur, fontsize=10, ha='center', va='bottom', 
                        bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'))
            except:
                pass
'''
def superposer_contours(ax, data_cube, lambdas, lambda_cible, methode_contour='percentile', niveaux=[90, 95], couleur='magenta', nom_raie=""):
    """Ajoute des isocontours (en % ou en intensité) sur l'axe fourni."""
    idx = (np.abs(lambdas - lambda_cible)).argmin()
    carte_slice = data_cube[idx, :, :]
    
    if methode_contour == 'percentile':
        valeurs_niveaux = np.percentile(carte_slice[~np.isnan(carte_slice)], niveaux)
    elif methode_contour == 'absolu':
        valeurs_niveaux = niveaux
        
    ax.contour(carte_slice, levels=valeurs_niveaux, colors=couleur, linewidths=1.5)
    
    label = f"{nom_raie} ({lambdas[idx]:.3f}µm)" if nom_raie else f"{lambdas[idx]:.3f}µm"
    ax.plot([], [], color=couleur, label=label)
    
#############SPECTURE 
def extraire_spectre_region(data_cube, wcs_2d, chemin_region, nom_cible):
    """Calcule le spectre moyen (moyenne spatiale) d'une région donnée."""
    masque = obtenir_masque_region(data_cube.shape[1:], wcs_2d, chemin_region, nom_cible)
    # On fait la moyenne sur les pixels du masque (Axe 1 et 2 du cube)
    spectre = np.nanmean(data_cube[:, masque], axis=1)
    return spectre

# À remplacer dans obs_DS9.py

def afficher_spectre(lambdas, flux, titre="Spectre", unite_flux="", lambda_highlight=None, lambda_lims=None, y_lims=None):
    """
    Affiche le spectre 1D avec option de surlignage et de limites d'axes.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(lambdas, flux, color='black', lw=1, drawstyle='steps-mid')
    
    if lambda_highlight:
        ax.axvline(lambda_highlight, color='red', alpha=0.5, linestyle='--')
        
    # --- AJOUT DES LIMITES PERSONNALISÉES ---
    if lambda_lims:
        ax.set_xlim(lambda_lims[0], lambda_lims[1])
    if y_lims:
        ax.set_ylim(y_lims[0], y_lims[1])
        
    ax.set_title(titre)
    ax.set_xlabel("Longueur d'onde (µm)")
    ax.set_ylabel(f"Flux ({unite_flux})")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig, ax

def superposer_regions(ax, wcs_2d, chemin_region, nom_cible=None, couleur='white', afficher_nom=True, padding=0):
    """
    Trace les régions .reg en gérant correctement les rectangles, cercles, et lignes.
    """
    from regions import Regions
    import numpy as np
    
    regions_list = Regions.read(chemin_region, format='ds9')
    
    for reg in regions_list:
        texte = reg.meta.get('text', '')
        if nom_cible and nom_cible not in texte:
            continue
        
        pixel_reg = reg.to_pixel(wcs_2d)
        
        # --- LA CORRECTION EST ICI ---
        # On ne modifie la taille que si c'est une forme qui a une largeur (ex: Rectangle, Ellipse)
        if padding > 0 and hasattr(pixel_reg, 'width') and hasattr(pixel_reg, 'height'):
            pixel_reg.width += 2 * padding
            pixel_reg.height += 2 * padding
            style_ligne = '--' # Pointillés pour montrer qu'on a ajouté du padding
        else:
            style_ligne = '-'  # Ligne pleine par défaut
            
        # --- DESSIN ---
        try:
            poly = pixel_reg.to_polygon()
            x = np.append(poly.vertices.x, poly.vertices.x[0])
            y = np.append(poly.vertices.y, poly.vertices.y[0])
            ax.plot(x, y, color=couleur, lw=1.5, linestyle=style_ligne)
        except Exception:
            # Sécurité pour les lignes simples ou les formes complexes
            pixel_reg.plot(ax=ax, color=couleur, lw=1.5)
            
        # --- ANNOTATIONS ---
        if afficher_nom and texte:
            try:
                bbox = pixel_reg.bounding_box
                ax.text((bbox.ixmin + bbox.ixmax)/2, bbox.iymax + 3, texte, 
                        color=couleur, fontsize=10, ha='center', va='bottom',
                        bbox=dict(facecolor='black', alpha=0.4, edgecolor='none'))
            except: 
                pass