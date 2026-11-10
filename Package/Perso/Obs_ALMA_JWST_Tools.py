#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'outils pour l'analyse multi-instrumentale PDR (ALMA, NIRSpec, MIRI).
Toutes les fonctions graphiques renvoient (fig, ax) ou (fig, axes).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from matplotlib.lines import Line2D
from astropy.coordinates import SkyCoord

# =============================================================================
# 1. LECTURE ET INGESTION
# =============================================================================

def open_and_clean_fits(file_path, hdu_idx=0):
    """
    Ouvre un fichier FITS 2D ou 3D/4D et renvoie une matrice 2D et son WCS celeste.
    """
    with fits.open(file_path) as hdul:
        ext = hdu_idx
        if isinstance(hdu_idx, str) and hdu_idx not in hdul:
            ext = 0
        data = hdul[ext].data
        header = hdul[ext].header

    if data is not None and data.ndim > 2:
        data = data.squeeze()
        while data.ndim > 2:
            data = data[0]

    wcs_2d = WCS(header).celestial
    return np.nan_to_num(data, nan=0.0), wcs_2d


# =============================================================================
# 2. ASTROMETRIE ET REPROJECTION
# =============================================================================

def apply_astrometric_shift(wcs_input, delta_ra_arcsec, delta_dec_arcsec):
    """
    Applique une translation rigide en secondes d'arc sur un WCS.
    """
    wcs_shifted = wcs_input.deepcopy()
    wcs_shifted.wcs.crval[0] += delta_ra_arcsec / 3600.0
    wcs_shifted.wcs.crval[1] += delta_dec_arcsec / 3600.0
    return wcs_shifted


def reproject_to_reference(data_source, wcs_source, wcs_target, shape_target):
    """
    Re-projette data_source (ayant wcs_source) sur le repere wcs_target de taille shape_target.
    """
    reproj_data, footprint = reproject_interp(
        (data_source, wcs_source),
        wcs_target,
        shape_out=shape_target
    )
    reproj_data = np.nan_to_num(reproj_data, nan=0.0)
    valid_mask = np.nan_to_num(footprint, nan=0.0) > 0.0
    return reproj_data, valid_mask


def get_footprint_polygon_pixels(wcs_source, shape_source, wcs_target):
    """
    Calcule les coordonnees pixels des 4 coins d'une carte source 
    projetes dans le repere pixel de la carte cible.
    """
    ny_s, nx_s = shape_source
    corners_pix_src = np.array([
        [0, 0],
        [nx_s - 1, 0],
        [nx_s - 1, ny_s - 1],
        [0, ny_s - 1]
    ])
    corners_sky = wcs_source.pixel_to_world(corners_pix_src[:, 0], corners_pix_src[:, 1])
    corners_pix_target = np.array([
        wcs_target.world_to_pixel(sky_coord) for sky_coord in corners_sky
    ])
    return corners_pix_target


def compute_bounding_box(mask, pad=10, max_shape=None):
    """
    Calcule les limites (xlim, ylim) pour zoomer autour d'un masque.
    """
    idx = np.argwhere(mask)
    if idx.size == 0:
        return (0, 100), (0, 100)
    ymin, xmin = idx.min(axis=0)
    ymax, xmax = idx.max(axis=0)

    if max_shape is not None:
        ny_max, nx_max = max_shape
        xlim = (min(nx_max, xmax + pad), max(0, xmin - pad))
        ylim = (max(0, ymin - pad), min(ny_max, ymax + pad))
    else:
        xlim = (xmax + pad, xmin - pad)
        ylim = (ymin - pad, ymax + pad)
    return xlim, ylim


# =============================================================================
# 3. TRACES GRAPHIQUES
# =============================================================================

def setup_celestial_axes(ax, grid_alpha=0.35, show_y_labels=True):
    """
    Formatage astronomique standard (RA croissante vers la gauche, grille fine).
    """
    ax.coords.grid(color='white', linestyle=':', linewidth=0.6, alpha=grid_alpha)
    ax.coords[0].set_axislabel('Right Ascension (RA)', fontsize=10)
    ax.coords[0].set_ticklabel_position('bt')
    ax.coords[0].ticklabels.set_visible(True)

    if show_y_labels:
        ax.coords[1].set_axislabel('Declination (DEC)', fontsize=10)
        ax.coords[1].set_ticklabel_position('lr')
        ax.coords[1].ticklabels.set_visible(True)
    else:
        ax.coords[1].set_axislabel('')
        plt.setp(ax.get_yticklabels(), visible=False)


def plot_single_map(data, wcs, title="Carte", cmap='magma', 
                    pct_limits=(2, 98), figsize=(8, 8), 
                    xlim=None, ylim=None, cbar_label=""):
    """
    Affiche une carte unique en projection WCS. Renvoie (fig, ax).
    """
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection=wcs)

    valid_vals = data[data > 0]
    vmin, vmax = np.nanpercentile(valid_vals, pct_limits) if valid_vals.size > 0 else (0, 1)

    im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    setup_celestial_axes(ax)

    ny, nx = data.shape
    ax.set_xlim(xlim if xlim is not None else (nx - 0.5, -0.5))
    ax.set_ylim(ylim if ylim is not None else (-0.5, ny - 0.5))
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    if cbar_label:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=10)

    return fig, ax


def plot_triptych(data_list, titles, cmaps, wcs_ref, 
                  pct_limits=(1, 99), figsize=(22, 7), 
                  cbar_labels=None):
    """
    Affiche 3 cartes cote a cote. Renvoie (fig, axes).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, subplot_kw={'projection': wcs_ref}, facecolor='white')
    plt.subplots_adjust(wspace=0.18)

    ny, nx = data_list[0].shape

    for i, (ax, data, title, cmap) in enumerate(zip(axes, data_list, titles, cmaps)):
        vals = data[data > 0]
        vmin, vmax = np.nanpercentile(vals, pct_limits) if vals.size > 0 else (0, 1)
        im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        setup_celestial_axes(ax, show_y_labels=(i == 0))

        ax.set_xlim(nx - 0.5, -0.5)
        ax.set_ylim(-0.5, ny - 0.5)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=8)
        if cbar_labels and len(cbar_labels) > i:
            cbar.set_label(cbar_labels[i], fontsize=9)

    return fig, axes


def add_intensity_contours(ax, data, levels_list=None, pct_list=None, 
                           mask=None, colors='cyan', linewidths=1.5, 
                           show_labels=True, label_color=None, label_suffix=""):
    """
    Trace des contours sur une carte donnee.
    """
    data_eval = data[mask & (data > 0)] if mask is not None else data[data > 0]
    if data_eval.size == 0:
        return None

    if levels_list is None:
        if pct_list is None:
            pct_list = [70, 85, 95]
        levels_list = np.nanpercentile(data_eval, pct_list)
    else:
        pct_list = [int(p) for p in np.linspace(50, 95, len(levels_list))]

    data_plot = np.where(mask, data, np.nan) if mask is not None else data
    cnt = ax.contour(data_plot, levels=levels_list, colors=colors, linewidths=linewidths, zorder=10)

    if show_labels:
        col_lbl = colors[0] if isinstance(colors, list) else colors
        if label_color is not None:
            col_lbl = label_color
        fmt_dict = {lvl: f"{int(p)}{label_suffix}" for lvl, p in zip(levels_list, pct_list)}
        ax.clabel(cnt, fmt=fmt_dict, inline=True, fontsize=8, colors=col_lbl)

    return cnt


def plot_comparison_duo(data_left, data_right, wcs_ref,
                        title_left="Gauche", title_right="Droite",
                        cmap_left='inferno', cmap_right='YlOrBr',
                        pct_limits=(2, 98), xlim=None, ylim=None, figsize=(16, 8)):
    """
    Trace deux panneaux comparatifs cote a cote. Renvoie (fig, (ax1, ax2)).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, subplot_kw={'projection': wcs_ref}, facecolor='white')
    plt.subplots_adjust(wspace=0.18)

    for ax, data, title, cmap in zip([ax1, ax2], [data_left, data_right], 
                                     [title_left, title_right], [cmap_left, cmap_right]):
        vals = data[data > 0]
        vmin, vmax = np.nanpercentile(vals, pct_limits) if vals.size > 0 else (0, 1)
        ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        setup_celestial_axes(ax)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    return fig, (ax1, ax2)

def plot_pdr_cross_comparison(data_ns, data_alma, wcs_ref, mask_footprint,
                              title_left="JWST NIRSpec H2 + Contours ALMA",
                              title_right="ALMA + Contours JWST H2",
                              cmap_left='inferno', cmap_right='YlOrBr',
                              pct_contours_alma=[50, 70, 85, 95],
                              pct_contours_ns=[70, 80, 90, 95],
                              colors_alma='cyan', colors_ns='deepskyblue',
                              coord_center=None, box_size_arcsec=None, pad_auto=25,
                              coins_footprint_pix=None,
                              cut_coords_pix=None, center_label=None,
                              path_region=None, obs_module=None,
                              show_region_labels=False,
                              figsize=(18, 9)):
    """
    Genere un duo miroir entre NIRSpec et ALMA avec contours croises et annotations explicites.
    Renvoie (fig, (ax_left, ax_right)).
    """
    ny_max, nx_max = data_alma.shape
    if coord_center is not None and box_size_arcsec is not None:
        deg_per_pix = np.abs(wcs_ref.wcs.cdelt[0])
        half_box_pix = int((box_size_arcsec / 2.0 / 3600.0) / deg_per_pix)
        xc_pix, yc_pix = wcs_ref.world_to_pixel(coord_center)
        xlim = (int(xc_pix + half_box_pix), int(xc_pix - half_box_pix))
        ylim = (int(yc_pix - half_box_pix), int(yc_pix + half_box_pix))
    else:
        xlim, ylim = compute_bounding_box(mask_footprint, pad=pad_auto, max_shape=(ny_max, nx_max))

    fig, (ax_l, ax_r) = plot_comparison_duo(
        data_left=data_ns, data_right=data_alma, wcs_ref=wcs_ref,
        title_left=title_left, title_right=title_right,
        cmap_left=cmap_left, cmap_right=cmap_right,
        pct_limits=(2, 98), xlim=xlim, ylim=ylim, figsize=figsize
    )

    # 1. Contours ALMA sur NIRSpec (Cyan)
    add_intensity_contours(
        ax_l, data_alma, pct_list=pct_contours_alma, mask=mask_footprint,
        colors=colors_alma, linewidths=1.5, show_labels=True
    )
    # 2. Contours NIRSpec sur ALMA (Couleur complementaire, ex: Bleu profond sur fond orange)
    add_intensity_contours(
        ax_r, data_ns, pct_list=pct_contours_ns, mask=mask_footprint,
        colors=colors_ns, linewidths=1.5, show_labels=True
    )

    for ax in (ax_l, ax_r):
        if coins_footprint_pix is not None:
            poly = Polygon(coins_footprint_pix, closed=True, edgecolor='white',
                           facecolor='none', lw=1.6, ls='--', zorder=11)
            ax.add_patch(poly)

        if cut_coords_pix is not None:
            (x_s, y_s), (x_e, y_e) = cut_coords_pix
            ax.plot([x_s, x_e], [y_s, y_e], color='red', lw=1.6, alpha=0.8, zorder=12)

        if path_region is not None and obs_module is not None:
            try:
                obs_module.superposer_regions(ax, wcs_ref, path_region, nom_cible=None, couleur='white')
                if not show_region_labels:
                    for txt_obj in ax.texts:
                        txt_obj.set_visible(False)
            except Exception:
                pass

        if coord_center is not None and center_label is not None:
            xc_p, yc_p = wcs_ref.world_to_pixel(coord_center)
            ax.scatter(xc_p, yc_p, marker='+', s=120, color='yellow', lw=2.0, zorder=15)
            ax.text(xc_p + 1.2, yc_p + 0.8, center_label, color='yellow', fontsize=10,
                    fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=1.5))

    # Cartouches explicatifs sur les contours (fond sombre semi-transparent)
    lbl_str_alma = str(pct_contours_alma).replace('[', '').replace(']', '')
    lbl_str_ns = str(pct_contours_ns).replace('[', '').replace(']', '')

    ax_l.text(0.03, 0.05, f"Fond : NIRSpec H2 (1-0 S1)\nContours Cyan : ALMA 12CO ({lbl_str_alma})",
              transform=ax_l.transAxes, color='cyan', fontsize=10, fontweight='bold',
              bbox=dict(facecolor='black', alpha=0.75, edgecolor='cyan', boxstyle='round,pad=0.4'), zorder=20)

    ax_r.text(0.03, 0.05, f"Fond : ALMA 12CO Total\nContours Bleus : NIRSpec H2 ({lbl_str_ns})",
              transform=ax_r.transAxes, color=colors_ns, fontsize=10, fontweight='bold',
              bbox=dict(facecolor='black', alpha=0.75, edgecolor=colors_ns, boxstyle='round,pad=0.4'), zorder=20)

    return fig, (ax_l, ax_r)




from scipy.ndimage import map_coordinates


def flip_data_across_cut(data, cut_coords_pix):
    """
    Applique une symétrie miroir orthogonale par rapport au segment de coupe 1D.
    Permet de vérifier visuellement si une inversion d'axe existe.
    """
    (xs, ys), (xe, ye) = cut_coords_pix
    ny, nx = data.shape
    y_idx, x_idx = np.indices((ny, nx))
    
    # Vecteur directeur unitaire de la coupe
    dx, dy = xe - xs, ye - ys
    norm = np.hypot(dx, dy)
    if norm == 0:
        return data.copy()
    ux, uy = dx / norm, dy / norm
    # Vecteur normal
    nx_v, ny_v = -uy, ux

    # Projection et reflexion orthogonale
    vx, vy = x_idx - xs, y_idx - ys
    dist_perp = vx * nx_v + vy * ny_v
    x_reflected = x_idx - 2.0 * dist_perp * nx_v
    y_reflected = y_idx - 2.0 * dist_perp * ny_v

    flipped = map_coordinates(data, [y_reflected.ravel(), x_reflected.ravel()], order=1, mode='constant', cval=0.0)
    return flipped.reshape((ny, nx))


def plot_pdr_triptych_overlay(data_ns, data_alma, wcs_ref, mask_footprint,
                              coord_center=None, box_size_arcsec=24.0, pad_auto=20,
                              pct_contours_alma=[55, 62, 68, 74, 80, 85, 90, 94, 97],
                              pct_contours_ns=[65, 75, 83, 90, 95, 98],
                              cmap_alma='YlOrBr',
                              cmap_ns_overlay='viridis',
                              cmap_alma_cnt='Blues',
                              cmap_ns_cnt=None,
                              alpha_overlay=0.38,
                              coins_footprint_pix=None, cut_coords_pix=None,
                              path_region=None, obs_module=None,
                              figsize=(22, 8)):
    """
    Triptyque PDR haute lisibilité adaptable à tous les traceurs ALMA :
      - Panneau 1 : Fond NIRSpec H2 + Contours fins ALMA avec gradient
      - Panneau 2 : Fond ALMA (cmap_alma) + Calque NIRSpec translucide contrasté (cmap_ns_overlay)
      - Panneau 3 : Fond ALMA + Contours fins NIRSpec H2
    """
    ny_max, nx_max = data_alma.shape

    # 1. Cadrage du zoom
    if coord_center is not None and box_size_arcsec is not None:
        deg_per_pix = np.abs(wcs_ref.wcs.cdelt[0])
        half_box_pix = int((box_size_arcsec / 2.0 / 3600.0) / deg_per_pix)
        xc_pix, yc_pix = wcs_ref.world_to_pixel(coord_center)
        xlim = (int(xc_pix + half_box_pix), int(xc_pix - half_box_pix))
        ylim = (int(yc_pix - half_box_pix), int(yc_pix + half_box_pix))
    else:
        xlim, ylim = compute_bounding_box(mask_footprint, pad=pad_auto, max_shape=(ny_max, nx_max))

    # 2. Adaptation automatique de la couleur des contours selon le fond ALMA
    if cmap_ns_cnt is None:
        if 'Blue' in cmap_alma:
            cmap_ns_cnt = 'autumn_r'   # Jaune-Orange vif sur fond bleu
        elif 'magma' in cmap_alma:
            cmap_ns_cnt = 'cyan'       # Cyan électrique sur fond noir/magenta
        else:
            cmap_ns_cnt = 'winter_r'   # Cyan-Bleu profond sur fond orange/cuivre

    fig, axes = plt.subplots(1, 3, figsize=figsize, subplot_kw={'projection': wcs_ref}, facecolor='white')
    plt.subplots_adjust(wspace=0.12)
    ax1, ax2, ax3 = axes

    # Dynamiques d'affichage locales
    sub_ns = data_ns[ylim[0]:ylim[1], xlim[1]:xlim[0]]
    sub_al = data_alma[ylim[0]:ylim[1], xlim[1]:xlim[0]]
    v_min_ns, v_max_ns = np.nanpercentile(sub_ns[sub_ns > 0], [2, 98.5])
    v_min_al, v_max_al = np.nanpercentile(sub_al[sub_al > 0], [2, 98.5])

    # -------------------------------------------------------------------------
    # PANNEAU 1 : Fond NIRSpec H2 + Contours fins ALMA (Gradient)
    # -------------------------------------------------------------------------
    ax1.imshow(data_ns, cmap='inferno', origin='lower', vmin=v_min_ns, vmax=v_max_ns)
    ax1.set_title("1: NIRSpec H2 + Contours ALMA", fontsize=11, fontweight='bold', pad=10)

    val_al = data_alma[mask_footprint & (data_alma > 0)]
    if val_al.size > 0:
        levels_al = np.nanpercentile(val_al, pct_contours_alma)
        alma_fente = np.where(mask_footprint, data_alma, np.nan)
        # Lignes fines progressives (0.8 à 1.5) sans chiffres écrasés
        ax1.contour(alma_fente, levels=levels_al, cmap=cmap_alma_cnt,
                    linewidths=np.linspace(0.8, 1.6, len(levels_al)), zorder=10)

    # -------------------------------------------------------------------------
    # PANNEAU 2 : Fond ALMA + NIRSpec translucide contrasté
    # -------------------------------------------------------------------------
    ax2.imshow(data_alma, cmap=cmap_alma, origin='lower', vmin=v_min_al, vmax=v_max_al)
    ns_masked = np.where(mask_footprint, data_ns, np.nan)
    ax2.imshow(ns_masked, cmap=cmap_ns_overlay, origin='lower', vmin=v_min_ns, vmax=v_max_ns,
               alpha=alpha_overlay, zorder=6)
    ax2.set_title(f"2: Fond ALMA ({cmap_alma}) + Overlay NIRSpec ({cmap_ns_overlay})", 
                  fontsize=11, fontweight='bold', pad=10)

    # -------------------------------------------------------------------------
    # PANNEAU 3 : Fond ALMA + Contours fins NIRSpec H2
    # -------------------------------------------------------------------------
    ax3.imshow(data_alma, cmap=cmap_alma, origin='lower', vmin=v_min_al, vmax=v_max_al)
    ax3.set_title("3: Fond ALMA + Contours NIRSpec H2", fontsize=11, fontweight='bold', pad=10)

    val_ns = data_ns[mask_footprint & (data_ns > 0)]
    if val_ns.size > 0:
        levels_ns = np.nanpercentile(val_ns, pct_contours_ns)
        ax3.contour(data_ns, levels=levels_ns, cmap=cmap_ns_cnt,
                    linewidths=np.linspace(0.8, 1.6, len(levels_ns)), zorder=10)

    # -------------------------------------------------------------------------
    # FORMATAGE DES AXES, EMPREINTES ET RÉGIONS
    # -------------------------------------------------------------------------
    for idx, ax in enumerate(axes):
        setup_celestial_axes(ax, show_y_labels=(idx == 0))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if coins_footprint_pix is not None:
            poly = Polygon(coins_footprint_pix, closed=True, edgecolor='white',
                           facecolor='none', lw=1.3, ls='--', zorder=11)
            ax.add_patch(poly)

        if cut_coords_pix is not None:
            (x_s, y_s), (x_e, y_e) = cut_coords_pix
            ax.plot([x_s, x_e], [y_s, y_e], color='red', lw=1.4, alpha=0.8, zorder=12)

        if path_region is not None and obs_module is not None:
            try:
                obs_module.superposer_regions(ax, wcs_ref, path_region, nom_cible=None, couleur='white')
                for txt_obj in ax.texts:
                    txt_obj.set_visible(False)
            except Exception:
                pass

    # Légendes compactes
    ax1.text(0.03, 0.04, f"Contours ALMA : {len(pct_contours_alma)} niveaux [{pct_contours_alma[0]} a {pct_contours_alma[-1]}]\nGradient '{cmap_alma_cnt}'",
             transform=ax1.transAxes, color='deepskyblue', fontsize=8.5, fontweight='bold',
             bbox=dict(facecolor='black', alpha=0.75, edgecolor='deepskyblue', boxstyle='round,pad=0.3'), zorder=20)

    ax2.text(0.03, 0.04, f"Overlay semi-transparent (alpha={alpha_overlay})\nVert/Jaune (NIRSpec) sur Fond ({cmap_alma})",
             transform=ax2.transAxes, color='white', fontsize=8.5, fontweight='bold',
             bbox=dict(facecolor='black', alpha=0.75, edgecolor='white', boxstyle='round,pad=0.3'), zorder=20)

    ax3.text(0.03, 0.04, f"Contours NIRSpec : {len(pct_contours_ns)} niveaux [{pct_contours_ns[0]} a {pct_contours_ns[-1]}]\nGradient adapte",
             transform=ax3.transAxes, color='cyan', fontsize=8.5, fontweight='bold',
             bbox=dict(facecolor='black', alpha=0.75, edgecolor='cyan', boxstyle='round,pad=0.3'), zorder=20)

    return fig, axes







# =============================================================================
# OUTILS AVANCÉS : BEAMS ET EXTRACTION SPECTRALE DYNAMIQUE (CUBE NIRSPEC)
# =============================================================================

from matplotlib.patches import Ellipse


def add_telescope_beams(ax, wcs_ref, beam_alma_arcsec=1.0, beam_jwst_arcsec=0.10,
                        loc='bottom_right', color_alma='cyan', color_jwst='white',
                        label_beams=True):
    """
    Trace la taille physique des faisceaux (ALMA et JWST) dans un coin de l'axe.
    
    Paramètres :
      - beam_alma_arcsec : FWHM ALMA en arcsec (defaut 1.0'' - Goicoechea+16)
      - beam_jwst_arcsec : FWHM JWST NIRSpec en arcsec (defaut 0.10'' - Peeters+24)
      - loc : 'bottom_right', 'bottom_left', 'top_right', 'top_left'
    """
    deg_per_pix = np.abs(wcs_ref.wcs.cdelt[0])
    arcsec_per_pix = deg_per_pix * 3600.0

    b_alma_pix = beam_alma_arcsec / arcsec_per_pix
    b_jwst_pix = beam_jwst_arcsec / arcsec_per_pix

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Largeurs de champ
    x_min, x_max = min(xlim), max(xlim)
    y_min, y_max = min(ylim), max(ylim)
    dx = x_max - x_min
    dy = y_max - y_min

    # Coordonnées du coin en pixels
    if 'right' in loc:
        xc = x_min + 0.15 * dx if xlim[0] > xlim[1] else x_max - 0.15 * dx
    else:
        xc = x_max - 0.15 * dx if xlim[0] > xlim[1] else x_min + 0.15 * dx

    yc = y_min + 0.10 * dy if 'bottom' in loc else y_max - 0.10 * dy

    # Faisceau ALMA (grand cercle ouvert)
    patch_alma = Ellipse((xc, yc), width=b_alma_pix, height=b_alma_pix,
                         edgecolor=color_alma, facecolor='none', lw=1.5, zorder=30)
    ax.add_patch(patch_alma)

    # Faisceau JWST (petit cercle plein ou centre net)
    patch_jwst = Ellipse((xc, yc), width=b_jwst_pix, height=b_jwst_pix,
                         edgecolor=color_jwst, facecolor=color_jwst, lw=1.0, zorder=31)
    ax.add_patch(patch_jwst)

    if label_beams:
        ax.text(xc, yc + b_alma_pix * 0.8,
                f"ALMA ({beam_alma_arcsec:.1f}\")\nJWST ({beam_jwst_arcsec:.2f}\")",
                color='white', fontsize=7.5, ha='center', va='bottom',
                bbox=dict(facecolor='black', alpha=0.65, edgecolor='none', pad=1.5),
                zorder=32)


def extraire_carte_raie_cube(cube_ns, lambdas_ns, lambda_0_microns,
                             v_lsr=10.5, v_turb_half=15.0, sub_continuum=True):
    """
    Extrait une raie spectrale du cube NIRSpec en trouvant la longueur d'onde 
    la plus proche ajustee du decalage Doppler et de la dispersion turbulente.
    """
    c_kms = 299792.458
    lambda_obs = lambda_0_microns * (1.0 + v_lsr / c_kms)
    d_lambda_turb = lambda_obs * (v_turb_half / c_kms)

    l_min = min(lambda_obs - d_lambda_turb, lambda_obs + d_lambda_turb)
    l_max = max(lambda_obs - d_lambda_turb, lambda_obs + d_lambda_turb)

    # Masque direct
    masque_raie = (lambdas_ns >= l_min) & (lambdas_ns <= l_max)

    # Repli automatique : si aucun point strict (pas trop large ou micro-gap), on prend les canaux les plus proches
    if not np.any(masque_raie):
        idx_proche = np.argmin(np.abs(lambdas_ns - lambda_obs))
        print(f"-> Avertissement : Longueur d'onde exacte non echantillonnee. "
              f"Repli sur le canal le plus proche : {lambdas_ns[idx_proche]:.5f} µm (cible : {lambda_obs:.5f} µm)")
        # On prend une fenetre de 3 canaux autour du plus proche
        idx_start = max(0, idx_proche - 1)
        idx_end = min(len(lambdas_ns), idx_proche + 2)
        masque_raie = np.zeros(len(lambdas_ns), dtype=bool)
        masque_raie[idx_start:idx_end] = True

    # Integration trapeze ou somme le long de l'axe spectral
    sous_cube_raie = cube_ns[masque_raie, :, :]
    lambdas_locaux = lambdas_ns[masque_raie]

    if len(lambdas_locaux) > 1:
        # Tri croissant pour trapz
        ordre = np.argsort(lambdas_locaux)
        # Compatible NumPy 2.0+ (np.trapezoid) et anciennes versions (np.trapz)
        trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        carte_raie = trapz_func(sous_cube_raie[ordre, :, :], lambdas_locaux[ordre], axis=0)
    else:
        carte_raie = np.squeeze(sous_cube_raie)

    # Soustraction du continuum local
    if sub_continuum:
        d_cont = 3.0 * d_lambda_turb
        m_cont = ((lambdas_ns >= l_min - d_cont) & (lambdas_ns < l_min)) | \
                 ((lambdas_ns > l_max) & (lambdas_ns <= l_max + d_cont))
        
        if np.any(m_cont):
            cont_moyen = np.nanmedian(cube_ns[m_cont, :, :], axis=0)
            delta_l = np.ptp(lambdas_locaux) if len(lambdas_locaux) > 1 else np.median(np.diff(np.sort(lambdas_ns)))
            carte_raie -= cont_moyen * delta_l

    return np.nan_to_num(carte_raie, nan=0.0)


def plot_cube_line_overlay(cube_ns, lambdas_ns, wcs_ns,
                           lambda_line_microns, line_name,
                           v_lsr=10.5, v_turb=25.0,
                           pct_contours=[70, 80, 88, 95],
                           cmap_contours='cool',
                           color_line='cyan',
                           xlim=None, ylim=None,
                           add_beams=True,
                           figsize=(16, 7)):
    """
    Trace la mosaïque NIRSpec en format paysage ciel avec contours d'une raie.
    Renvoie (fig, ax).
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fond_gris = np.nanmedian(cube_ns, axis=0)
    fond_gris = np.nan_to_num(fond_gris, nan=0.0)

    # Extraction propre de la raie
    carte_raie = extraire_carte_raie_cube(
        cube_ns, lambdas_ns, lambda_line_microns,
        v_lsr=v_lsr, v_turb_half=v_turb, sub_continuum=True
    )

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection=wcs_ns)

    # Masque des zones sans données (bords noirs du cube)
    masque_donnees = fond_gris > 0
    v1, v2 = np.nanpercentile(fond_gris[masque_donnees], [3, 98.5])
    im = ax.imshow(fond_gris, cmap='gray', origin='lower', vmin=v1, vmax=v2)

    # Contours calculés uniquement sur les pixels valides à l'intérieur de la fente
    valides_raie = carte_raie[masque_donnees & (carte_raie > 0)]
    if valides_raie.size > 0:
        levels = np.nanpercentile(valides_raie, pct_contours)
        raie_masquee = np.where(masque_donnees, carte_raie, np.nan)
        ax.contour(raie_masquee, levels=levels, cmap=cmap_contours,
                   linewidths=np.linspace(1.0, 2.2, len(levels)), zorder=10)

    # Formatage propre sans les warnings Astropy récents
    ax.coords.grid(color='white', linestyle=':', linewidth=0.6, alpha=0.35)
    ax.coords[0].set_axislabel('Right Ascension (RA)', fontsize=11)
    ax.coords[1].set_axislabel('Declination (DEC)', fontsize=11)

    # Orientation paysage adaptée
    ny, nx = fond_gris.shape
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(nx - 0.5, -0.5)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(-0.5, ny - 0.5)

    if add_beams:
        add_telescope_beams(ax, wcs_ns, beam_alma_arcsec=1.0, beam_jwst_arcsec=0.10,
                            loc='bottom_right', color_alma='cyan', color_jwst='white')

    ax.set_title(f"JWST NIRSpec : {line_name} ({lambda_line_microns:.5f} µm)\nFond : Continuum Moyen (Gris)",
                 fontsize=12, fontweight='bold', pad=14)

    pct_str = str(pct_contours).replace('[', '').replace(']', '')
    ax.text(0.02, 0.05,
            f"Raie : {line_name}\n"
            f"Lambda_repos : {lambda_line_microns:.5f} µm\n"
            f"v_lsr = {v_lsr} km/s (Delta_v = +/-{v_turb} km/s)\n"
            f"Contours ({pct_str})",
            transform=ax.transAxes, color=color_line, fontsize=9.5, fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.75, edgecolor=color_line, boxstyle='round,pad=0.4'),
            zorder=20)

    plt.tight_layout()
    return fig, ax

def plot_cube_multilines_subregion(cube_ns, lambdas_ns, wcs_ns,
                                   lines_config,
                                   lambda_bg=2.12183,
                                   v_lsr=10.5, v_turb=25.0,
                                   coord_cut_str=('5h35m20.7s', '-5d25m18s'),
                                   pct_contours=[80, 90, 95],
                                   add_beams=True,
                                   figsize=(9, 9)):
    """
    Rogne la mosaïque NIRSpec au niveau de la coupure exacte et trace les contours configurés.
    """
    from astropy.coordinates import SkyCoord
    from matplotlib.lines import Line2D

    wcs_cel = wcs_ns.celestial

    # 1. Extraction du fond H2
    fond_h2 = extraire_carte_raie_cube(
        cube_ns, lambdas_ns, lambda_bg,
        v_lsr=v_lsr, v_turb_half=v_turb, sub_continuum=True
    )
    ny, nx = fond_h2.shape

    # 2. Calcul robuste de l'abscisse pixel de coupure
    try:
        ra_str, dec_str = coord_cut_str
        coord_cible = SkyCoord(ra_str, dec_str, frame='fk5')
        pix_coords = wcs_cel.world_to_pixel(coord_cible)
        x_cut_float = float(pix_coords[0])
        if np.isnan(x_cut_float):
            raise ValueError("world_to_pixel a renvoyé NaN")
        x_cut_pix = int(np.clip(np.round(x_cut_float), 0, nx - 1))
    except Exception as e:
        print(f"-> Avertissement calcul WCS ({e}), utilisation de la coupure médiane par défaut.")
        x_cut_pix = int(nx * 0.48)

    # 3. Masquage de la zone hors PDR
    masque_pdr = np.zeros_like(fond_h2, dtype=bool)
    masque_pdr[:, x_cut_pix:] = (fond_h2[:, x_cut_pix:] > 0)
    fond_h2_rogne = np.where(masque_pdr, fond_h2, np.nan)

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection=wcs_cel)

    valides_fond = fond_h2[masque_pdr]
    v1, v2 = np.nanpercentile(valides_fond, [3, 98.5]) if valides_fond.size > 0 else (0, 1)
    ax.imshow(fond_h2_rogne, cmap='gray', origin='lower', vmin=v1, vmax=v2)

    # 4. Tracé des contours par raie
    custom_legend_handles = []
    lws = [1.1, 1.6, 2.2]

    for cfg in lines_config:
        nom = cfg["name"]
        l_0 = cfg["lambda"]
        cmap_c = cfg.get("cmap", "Blues")
        pcts = cfg.get("pct", pct_contours)

        carte_raie = extraire_carte_raie_cube(
            cube_ns, lambdas_ns, l_0,
            v_lsr=v_lsr, v_turb_half=v_turb, sub_continuum=True
        )

        valides_raie = carte_raie[masque_pdr & (carte_raie > 0)]
        if valides_raie.size > 0:
            levels = np.nanpercentile(valides_raie, pcts)
            raie_m = np.where(masque_pdr, carte_raie, np.nan)
            cnt = ax.contour(raie_m, levels=levels, cmap=cmap_c,
                             linewidths=lws, zorder=10)
            
            col_repr = cnt.collections[len(cnt.collections) // 2].get_edgecolor()[0]
            custom_legend_handles.append(
                Line2D([0], [0], color=col_repr, lw=2.2, label=f"{nom} ({l_0:.4f} µm)")
            )

    # 5. Démarcation et cadrage carré
    ax.axvline(x=x_cut_pix, color='red', linestyle='--', linewidth=2.0, alpha=0.9, zorder=15)
    ax.set_xlim(nx - 0.5, x_cut_pix - 1)
    ax.set_ylim(-0.5, ny - 0.5)

    ax.coords.grid(color='white', linestyle=':', linewidth=0.6, alpha=0.35)
    ax.coords[0].set_axislabel('Right Ascension (RA)', fontsize=11)
    ax.coords[1].set_axislabel('Declination (DEC)', fontsize=11)

    if add_beams:
        add_telescope_beams(ax, wcs_cel, beam_alma_arcsec=1.0, beam_jwst_arcsec=0.10,
                            loc='bottom_right', color_alma='cyan', color_jwst='white')

    ax.set_title(r"JWST NIRSpec : Region PDR (Coupure a $5^\mathrm{h}35^\mathrm{m}20{,}7^\mathrm{s},\ -5^\circ25'18''$)",
                 fontsize=12, fontweight='bold', pad=12)

    pct_txt = ", ".join([f"{p}%" for p in pct_contours])
    leg = ax.legend(handles=custom_legend_handles, loc='lower left',
                    title=f"Fond : H2 1-0 S(1)\nContours : [{pct_txt}]",
                    fontsize=9, title_fontsize=9.5, framealpha=0.85,
                    facecolor='black', edgecolor='white')
    leg.get_title().set_color('white')
    for text in leg.get_texts():
        text.set_color('white')

    plt.tight_layout()
    return fig, ax




def extraire_moyenne_multi_raies(cube_ns, lambdas_ns, list_lambdas_microns,
                                 v_lsr=10.5, v_turb_half=25.0):
    """
    Extrait et moyenne un groupe de raies spectrales pour rehausser le S/B.
    """
    cartes = []
    for l_0 in list_lambdas_microns:
        c_i = extraire_carte_raie_cube(cube_ns, lambdas_ns, l_0,
                                       v_lsr=v_lsr, v_turb_half=v_turb_half,
                                       sub_continuum=True)
        cartes.append(c_i)
    return np.nanmean(cartes, axis=0)

def plot_cube_stacked_vibrational_lines(cube_ns, lambdas_ns, wcs_ns,
                                        groupe_v1, groupe_v2,
                                        groupe_v3=None,
                                        lambda_bg=2.12183,
                                        v_lsr=10.5, v_turb=25.0,
                                        coord_cut_str=('5h35m20.7s', '-5d25m18s'),
                                        pct_contours=[80, 90, 95],
                                        add_beams=True,
                                        figsize=(9, 9)):
    """
    Empile et moyenne les spectres de deux ou trois groupes de raies (v1, v2, v3 optionnel).
    Affiche chaque groupe avec un gradient de couleur distinct et une légende avec barres d'échantillons.
    Renvoie (fig, ax).
    """
    wcs_cel = wcs_ns.celestial

    # 1. Fond H2
    fond_h2 = extraire_carte_raie_cube(
        cube_ns, lambdas_ns, lambda_bg,
        v_lsr=v_lsr, v_turb_half=v_turb, sub_continuum=True
    )
    ny, nx = fond_h2.shape

    # 2. Conversion de la coupure en coordonnées pixels
    try:
        ra_str, dec_str = coord_cut_str
        coord_cible = SkyCoord(ra_str, dec_str, frame='fk5')
        pix_coords = wcs_cel.world_to_pixel(coord_cible)
        x_cut_float = float(pix_coords[0])
        x_cut_pix = int(np.clip(np.round(x_cut_float), 0, nx - 1))
    except Exception:
        x_cut_pix = int(nx * 0.48)

    masque_pdr = np.zeros_like(fond_h2, dtype=bool)
    masque_pdr[:, x_cut_pix:] = (fond_h2[:, x_cut_pix:] > 0)
    fond_h2_rogne = np.where(masque_pdr, fond_h2, np.nan)

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection=wcs_cel)

    valides_fond = fond_h2[masque_pdr]
    v1, v2 = np.nanpercentile(valides_fond, [3, 98.5]) if valides_fond.size > 0 else (0, 1)
    ax.imshow(fond_h2_rogne, cmap='gray', origin='lower', vmin=v1, vmax=v2)

    lws = [1.1, 1.6, 2.2]
    custom_handles = []

    # Liste des groupes à traiter séquentiellement
    groupes_a_tracer = [
        (groupe_v1, groupe_v1.get("cmap", "Reds")),
        (groupe_v2, groupe_v2.get("cmap", "Blues"))
    ]
    if groupe_v3 is not None:
        groupes_a_tracer.append((groupe_v3, groupe_v3.get("cmap", "spring")))

    # 3. Extraction, calcul des percentiles et tracé pour chaque groupe
    for grp, cmap_grp in groupes_a_tracer:
        carte_grp = extraire_moyenne_multi_raies(
            cube_ns, lambdas_ns, grp["lambdas"],
            v_lsr=v_lsr, v_turb_half=v_turb
        )
        val_grp = carte_grp[masque_pdr & (carte_grp > 0)]
        pcts_grp = grp.get("pct", pct_contours)

        if val_grp.size > 0:
            lvl_grp = np.nanpercentile(val_grp, pcts_grp)
            c_m = np.where(masque_pdr, carte_grp, np.nan)
            cnt = ax.contour(c_m, levels=lvl_grp, cmap=cmap_grp, linewidths=lws, zorder=10)
            col_repr = cnt.collections[len(cnt.collections) // 2].get_edgecolor()[0]
            
            nb_raies = len(grp["lambdas"])
            libelle = f"{grp['name']} ({nb_raies} raie{'s' if nb_raies > 1 else ''})"
            custom_handles.append(Line2D([0], [0], color=col_repr, lw=2.4, label=libelle))

    # 4. Trait de démarcation vertical rouge
    ax.axvline(x=x_cut_pix, color='red', linestyle='--', linewidth=2.0, alpha=0.9, zorder=15)

    # 5. Cadrage carré
    ax.set_xlim(nx - 0.5, x_cut_pix - 1)
    ax.set_ylim(-0.5, ny - 0.5)

    ax.coords.grid(color='white', linestyle=':', linewidth=0.6, alpha=0.35)
    ax.coords[0].set_axislabel('Right Ascension (RA)', fontsize=11)
    ax.coords[1].set_axislabel('Declination (DEC)', fontsize=11)

    if add_beams:
        add_telescope_beams(ax, wcs_cel, beam_alma_arcsec=1.0, beam_jwst_arcsec=0.10,
                            loc='bottom_right', color_alma='cyan', color_jwst='white')

    ax.set_title(r"JWST NIRSpec : Somme Spectrale CO $v=1{-}0$, $v=2{-}1$ et $\mathrm{H_2}$",
                 fontsize=12, fontweight='bold', pad=12)

    pct_txt = ", ".join([f"{p}%" for p in pct_contours])
    leg = ax.legend(handles=custom_handles, loc='lower left',
                    title=f"Fond : H2 1-0 S(1)\nContours : [{pct_txt}]",
                    fontsize=9, title_fontsize=9.5, framealpha=0.85,
                    facecolor='black', edgecolor='white')
    leg.get_title().set_color('white')
    for text in leg.get_texts():
        text.set_color('white')

    plt.tight_layout()
    return fig, ax
























# =============================================================================
# EXTRACTION 1D LE LONG DE LA COUPE ORIENTÉE VERS THETA1 ORI C
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u
from scipy.ndimage import map_coordinates


def generer_geometrie_coupe(wcs_ref, coord_start, coord_end, coord_star,
                            num_points=500, d_orion_pc=414.0):
    """
    Calcule les vecteurs de la fente, les coordonnées pixels échantillonnées
    et les distances angulaires (arcsec) et physiques (pc) à Theta1 Ori C.
    """
    wcs_cel = wcs_ref.celestial
    arcsec_to_pc = d_orion_pc / 206265.0

    xs, ys = wcs_cel.world_to_pixel(coord_start)
    xe, ye = wcs_cel.world_to_pixel(coord_end)

    dx, dy = xe - xs, ye - ys
    norme = np.hypot(dx, dy)
    ux, uy = dx / norme, dy / norme
    nx_perp, ny_perp = -uy, ux

    x_line = np.linspace(xs, xe, num_points)
    y_line = np.linspace(ys, ye, num_points)

    sky_pts = wcs_cel.pixel_to_world(x_line, y_line)
    distances_arcsec = sky_pts.separation(coord_star).arcsecond
    distances_pc = distances_arcsec * arcsec_to_pc

    geom = {
        'x_line': x_line,
        'y_line': y_line,
        'ux': ux, 'uy': uy,
        'nx_perp': nx_perp, 'ny_perp': ny_perp,
        'dist_arcsec': distances_arcsec,
        'dist_pc': distances_pc,
        'wcs_cel': wcs_cel,
        'd_orion_pc': d_orion_pc,
        'arcsec_to_pc': arcsec_to_pc
    }
    return geom


def extraire_profil_1d_moyenne(carte_2d, geom, largeur_fente_pix=3, num_perp=5):
    """
    Extrait un profil 1D le long de la coupe en moyennant orthogonalement
    sur plusieurs pixels pour débruiter la mesure (largeur_fente_pix).
    """
    x_c = geom['x_line']
    y_c = geom['y_line']
    nx_p = geom['nx_perp']
    ny_p = geom['ny_perp']

    if largeur_fente_pix <= 1 or num_perp <= 1:
        profil = map_coordinates(carte_2d, [y_c, x_c], order=1, mode='nearest')
        return np.nan_to_num(profil, nan=0.0)

    offsets = np.linspace(-largeur_fente_pix / 2.0, largeur_fente_pix / 2.0, num_perp)
    echantillons = []
    for off in offsets:
        xp = x_c + off * nx_p
        yp = y_c + off * ny_p
        prof_i = map_coordinates(carte_2d, [yp, xp], order=1, mode='nearest')
        echantillons.append(prof_i)

    profil_moyen = np.nanmean(echantillons, axis=0)
    return np.nan_to_num(profil_moyen, nan=0.0)


def normaliser_profil_local(profil, masque_fenetre):
    """Normalise localement min-max dans la fenêtre [132'', 135'']."""
    tranche = profil[masque_fenetre]
    p_min = np.nanmin(tranche)
    p_max = np.nanmax(tranche)
    if (p_max - p_min) > 0:
        return (tranche - p_min) / (p_max - p_min)
    return np.zeros_like(tranche)




def tracer_profils_4panneaux_df3(geom, profils_dict,
                                 raies_h2_overview,
                                 raies_h2_rovib,
                                 raies_co_v1,
                                 raies_co_v2,
                                 zoom_arcsec=(132.0, 135.0),
                                 res_angulaire_arcsec=0.15,
                                 figsize=(13, 17)):
    """
    Trace les 4 panneaux synchronisés le long de la coupe DF3 [132" - 135"].
    - Lignes verticales pour les sommets MIRI tracés.
    - CO v=1-0 en gradient Blues (clair -> foncé).
    - CO v=2-1 en gradient Reds (clair -> foncé).
    - Estimation des séparations angulaires (avec incertitude +/- 0.15") et affichage console/figure.
    """
    dist_arcsec = geom['dist_arcsec']
    dist_pc = geom['dist_pc']
    z_min, z_max = zoom_arcsec
    d_orion_pc = geom.get('d_orion_pc', 414.0)
    arcsec_to_pc = d_orion_pc / 206265.0

    masque_zoom = (dist_arcsec >= z_min) & (dist_arcsec <= z_max)
    x_arc = dist_arcsec[masque_zoom]
    x_pc = dist_pc[masque_zoom]

    def normaliser_signal(prof):
        tr = np.nan_to_num(prof[masque_zoom], nan=0.0)
        p_min, p_max = np.nanmin(tr), np.nanmax(tr)
        return (tr - p_min) / (p_max - p_min) if (p_max - p_min) > 1e-12 else np.zeros_like(tr)

    # 1. Front H/H2 via H2 1-0 S(1) NIRSpec
    p_h2_10s1 = normaliser_signal(profils_dict.get("H2_10S1", profils_dict[list(profils_dict.keys())[0]]))
    x_front_h2 = x_arc[np.argmax(p_h2_10s1)]

    # 2. Sommets des raies MIRI
    sommets_miri = {}
    couleurs_miri = {
        "H2_00S1_MIRI": ("forestgreen", r"MIRI $0{-}0\,\mathrm{S}(1)$"),
        "H2_00S3_MIRI": ("darkorange",  r"MIRI $0{-}0\,\mathrm{S}(3)$"),
        "H2_00S8_MIRI": ("darkviolet",  r"MIRI $0{-}0\,\mathrm{S}(8)$")
    }
    for cle, (coul, nom) in couleurs_miri.items():
        if cle in profils_dict:
            p_m = normaliser_signal(profils_dict[cle])
            if np.any(p_m > 0):
                x_max_m = x_arc[np.argmax(p_m)]
                sommets_miri[cle] = (x_max_m, coul, nom)

    # 3. Détection des pics CO v=1-0 et v=2-1
    co_v1_peaks = []
    for _, cle, _ in raies_co_v1:
        if cle in profils_dict:
            p = normaliser_signal(profils_dict[cle])
            if np.any(p > 0):
                co_v1_peaks.append(x_arc[np.argmax(p)])

    co_v2_peaks = []
    for _, cle, _ in raies_co_v2:
        if cle in profils_dict:
            p = normaliser_signal(profils_dict[cle])
            if np.any(p > 0):
                co_v2_peaks.append(x_arc[np.argmax(p)])

    v1_start, v1_end = (np.min(co_v1_peaks), np.max(co_v1_peaks)) if co_v1_peaks else (133.20, 133.70)
    v2_start, v2_end = (np.min(co_v2_peaks), np.max(co_v2_peaks)) if co_v2_peaks else (133.30, 133.60)

    # --- CALCULS ET BILAN STATISTIQUE ---
    sep_v1_start = v1_start - x_front_h2
    sep_v1_end   = v1_end - x_front_h2
    sep_v2_start = v2_start - x_front_h2
    sep_v2_end   = v2_end - x_front_h2

    print("\n==========================================================================")
    print(f"📏 MESURES DES STRATIFICATIONS PAR RAPPORT AU FRONT H/H2 (1-0 S1 = {x_front_h2:.3f}\")")
    print(f"   Résolution angulaire NIRSpec : +/- {res_angulaire_arcsec:.2f}\" (+/- {res_angulaire_arcsec * arcsec_to_pc * 1e3:.1f} mpc)")
    print("--------------------------------------------------------------------------")
    print(f" • CO v=1-0 Peak Start : {sep_v1_start:+.3f}\" +/- {res_angulaire_arcsec:.2f}\" ({sep_v1_start * arcsec_to_pc * 1e3:+.1f} mpc)")
    print(f" • CO v=1-0 Peak End   : {sep_v1_end:+.3f}\" +/- {res_angulaire_arcsec:.2f}\" ({sep_v1_end * arcsec_to_pc * 1e3:+.1f} mpc)")
    print(f" • CO v=2-1 Peak Start : {sep_v2_start:+.3f}\" +/- {res_angulaire_arcsec:.2f}\" ({sep_v2_start * arcsec_to_pc * 1e3:+.1f} mpc)")
    print(f" • CO v=2-1 Peak End   : {sep_v2_end:+.3f}\" +/- {res_angulaire_arcsec:.2f}\" ({sep_v2_end * arcsec_to_pc * 1e3:+.1f} mpc)")
    for cle, (x_m, _, nom) in sommets_miri.items():
        sep_m = x_m - x_front_h2
        print(f" • {nom:20s} : {sep_m:+.3f}\" ({sep_m * arcsec_to_pc * 1e3:+.1f} mpc)")
    print("==========================================================================\n")

    # --- CONFIGURATION GRAPHIQUE ---
    plt.rcParams['font.family'] = 'serif'
    fig, (ap1, ap2, ap3, ap4) = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.0)
    liste_axes = [ap1, ap2, ap3, ap4]

    def config_panel(ax, titre):
        ax.set_ylim(-0.05, 1.25)
        ax.grid(True, ls=':', alpha=0.3)
        ax.set_ylabel("Norm. Intensity", fontsize=10, weight='bold')
        ax.tick_params(axis='both', colors='black', labelsize=10)
        ax.text(0.01, 0.94, titre, transform=ax.transAxes,
                fontsize=10, weight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.2'))

    # ------ PANEL A : OVERVIEW H2 ------
    for nom, cle, coul, ls, mk in raies_h2_overview:
        if cle in profils_dict:
            p = normaliser_signal(profils_dict[cle])
            ap1.plot(x_arc, p, color=coul, lw=2.0, ls=ls, marker=mk, markevery=7, label=nom)
    config_panel(ap1, r"Panel A: Reference Individual $\mathrm{H}_2$ Transitions Overview")
    ap1.legend(loc='upper right', fontsize=8, framealpha=0.9, ncols=2)

    ax_top = ap1.twiny()
    ax_top.set_xlim(x_pc.min(), x_pc.max())
    ax_top.set_xlabel(r"Physical Distance from $\theta^1$ Ori C (pc)", fontsize=10, weight='bold', labelpad=10)
    ax_top.tick_params(axis='x', colors='black', labelsize=10)

    # ------ PANEL B : NIRSPEC H2 RO-VIBRATIONNEL ------
    cmap_h2 = plt.get_cmap("viridis")
    for i, (nom, cle) in enumerate(raies_h2_rovib):
        if cle in profils_dict:
            p = normaliser_signal(profils_dict[cle])
            c = cmap_h2(0.20 + (i / max(1, len(raies_h2_rovib) - 1)) * 0.70)
            ap2.plot(x_arc, p, color=c, lw=2.0, label=nom)
    config_panel(ap2, r"Panel B: NIRSpec Ro-Vibrational $\mathrm{H}_2$ Transitions")
    ap2.legend(loc='upper right', fontsize=8, ncols=3)

    # ------ PANEL C : CO v=1-0 (GRADIENT BLUES) ------
    cmap_v1 = plt.get_cmap("Blues")
    n_v1 = len(raies_co_v1)
    for i, (nom, cle, j_val) in enumerate(raies_co_v1):
        if cle in profils_dict:
            p = normaliser_signal(profils_dict[cle])
            c_val = 0.35 + (i / max(1, n_v1 - 1)) * 0.60  # Du bleu clair au bleu très foncé
            ap3.plot(x_arc, p, color=cmap_v1(c_val), lw=1.6, label=f"CO 1-0 J={j_val}")
    config_panel(ap3, r"Panel C: Fundamental CO $v=1{-}0$ Molecular Flow")
    ap3.legend(loc='upper right', fontsize=7.5, ncols=4)

    # ------ PANEL D : CO v=2-1 (GRADIENT REDS) ------
    cmap_v2 = plt.get_cmap("Reds")
    n_v2 = len(raies_co_v2)
    for i, (nom, cle, j_val) in enumerate(raies_co_v2):
        if cle in profils_dict:
            p = normaliser_signal(profils_dict[cle])
            c_val = 0.35 + (i / max(1, n_v2 - 1)) * 0.60  # Du rose/rouge clair au rouge très foncé
            ap4.plot(x_arc, p, color=cmap_v2(c_val), lw=1.6, label=f"CO 2-1 J={j_val}")
    config_panel(ap4, r"Panel D: Highly Excited CO $v=2{-}1$ Molecular Flow")
    ap4.legend(loc='upper right', fontsize=7.5, ncols=4)

    # ------ MARQUEURS ET LIGNES REPÈRES SYNCHRONISÉES ------
    for ax in liste_axes:
        # Front H/H2
        ax.axvline(x=x_front_h2, color='black', lw=2.0, ls='--', zorder=20)

        # Pics MIRI (pointillés fins colorés)
        for cle, (x_m, coul_m, _) in sommets_miri.items():
            ax.axvline(x=x_m, color=coul_m, lw=1.1, ls=':', alpha=0.75, zorder=15)

        # Pics CO v=1-0 (Bleu soutenu)
        ax.axvline(x=v1_start, color='royalblue', lw=1.3, ls='--', alpha=0.9, zorder=18)
        ax.axvline(x=v1_end,   color='royalblue', lw=1.3, ls='--', alpha=0.9, zorder=18)

        # Pics CO v=2-1 (Rouge soutenu)
        ax.axvline(x=v2_start, color='crimson',   lw=1.3, ls='-.', alpha=0.9, zorder=18)
        ax.axvline(x=v2_end,   color='crimson',   lw=1.3, ls='-.', alpha=0.9, zorder=18)

    # Annotations textuelles des distances
    ap1.text(x_front_h2 - 0.02, 1.18, r"$\mathrm{H}/\mathrm{H}_2\ \mathrm{Front}\ (1{-}0\,\mathrm{S}1)$",
             color='black', fontsize=8.5, weight='bold', rotation=90, ha='right', va='top')

    # Annotations sommets MIRI sur Panel A
    for cle, (x_m, coul_m, nom_m) in sommets_miri.items():
        ap1.text(x_m + 0.02, 1.14, f"Peak {nom_m.split()[-1]}",
                 color=coul_m, fontsize=7.5, weight='bold', rotation=90, ha='left', va='top')

    # Textes des décalages relatifs sur Panels C et D
    ap3.text(v1_start - 0.02, 1.15, f"v=1 Start (+{sep_v1_start:.2f}\"±{res_angulaire_arcsec:.2f}\")",
             color='royalblue', fontsize=7.5, rotation=90, ha='right', va='top', weight='bold')
    ap3.text(v1_end + 0.02, 1.15, f"v=1 End (+{sep_v1_end:.2f}\"±{res_angulaire_arcsec:.2f}\")",
             color='royalblue', fontsize=7.5, rotation=90, ha='left', va='top', weight='bold')

    ap4.text(v2_start - 0.02, 1.15, f"v=2 Start (+{sep_v2_start:.2f}\"±{res_angulaire_arcsec:.2f}\")",
             color='crimson', fontsize=7.5, rotation=90, ha='right', va='top', weight='bold')
    ap4.text(v2_end + 0.02, 1.15, f"v=2 End (+{sep_v2_end:.2f}\"±{res_angulaire_arcsec:.2f}\")",
             color='crimson', fontsize=7.5, rotation=90, ha='left', va='top', weight='bold')

    ap4.set_xlabel(r"Absolute projected distance from ionizing star $\theta^1$ Ori C (arcsec)",
                   fontsize=11, weight='bold', color='black')
    ap4.set_xlim(z_min, z_max)
    for hidden_ax in [ap1, ap2, ap3]:
        plt.setp(hidden_ax.get_xticklabels(), visible=False)

    plt.tight_layout()
    return fig, liste_axes
























import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u
from reproject import reproject_interp


def charger_cartes_miri(path_miri_dir, wcs_cible, shape_cible):
    """
    Charge les cartes H2 MIRI décontaminées du continuum et les reprojette
    sur la grille WCS du cube NIRSpec de référence.
    Renvoie un dictionnaire { 'H2_00S1': array_2d, ... }.
    """
    wcs_cible_cel = wcs_cible.celestial if hasattr(wcs_cible, 'celestial') else wcs_cible
    cartes_miri = {}

    for j in range(1, 9):
        nom_fichier = f"MIRI_Map_H2_0-0S({j})_contsub.fits"
        chemin = os.path.join(path_miri_dir, nom_fichier)
        cle = f"H2_00S{j}"

        if os.path.exists(chemin):
            try:
                with fits.open(chemin) as hdul:
                    ext = 'SCI' if 'SCI' in hdul else (1 if len(hdul) > 1 else 0)
                    data_raw = np.nan_to_num(hdul[ext].data, nan=0.0)
                    wcs_raw = WCS(hdul[ext].header, fobj=hdul)
                    
                    reproj_data, _ = reproject_interp(
                        (data_raw, wcs_raw),
                        wcs_cible_cel,
                        shape_out=shape_cible
                    )
                    cartes_miri[cle] = np.nan_to_num(reproj_data, nan=0.0)
                    print(f"   [MIRI] {cle} aligné (max: {np.max(cartes_miri[cle]):.2e})")
            except Exception as e:
                print(f"   [MIRI] Erreur sur {nom_fichier} : {e}")
    return cartes_miri


def tracer_comparaison_2d_miri_multiraies(carte_ns, cartes_miri, wcs_ref,
                                          coord_star, coord_start, coord_end,
                                          raies_miri_cfg,
                                          nom_raie_ns=r"NIRSpec $\mathrm{H}_2\ 1{-}0\,\mathrm{S}(1)$",
                                          fenetre_arcsec=(128.0, 138.0),
                                          pct_contours=[80, 90, 95],
                                          figsize=(16, 7)):
    """
    Superpose le fond NIRSpec en niveaux de gris et plusieurs contours MIRI 
    avec des codes couleurs distincts pour valider la stratification et l'alignement WCS.
    """
    wcs_cel = wcs_ref.celestial if hasattr(wcs_ref, 'celestial') else wcs_ref
    ny, nx = carte_ns.shape

    # 1. Échantillonnage de l'axe de coupe 1D
    v_ra = (coord_end.ra.deg - coord_start.ra.deg)
    v_dec = (coord_end.dec.deg - coord_start.dec.deg)
    pts_ext = SkyCoord(
        np.linspace(coord_start.ra.deg - 0.7 * v_ra, coord_end.ra.deg + 0.7 * v_ra, 3000) * u.deg,
        np.linspace(coord_start.dec.deg - 0.7 * v_dec, coord_end.dec.deg + 0.7 * v_dec, 3000) * u.deg,
        frame='fk5'
    )
    x_ext, y_ext = wcs_cel.world_to_pixel(pts_ext)
    dist_pts_arcsec = pts_ext.separation(coord_star).arcsecond

    # Vecteur normal unitaire
    xs, ys = wcs_cel.world_to_pixel(coord_start)
    xe, ye = wcs_cel.world_to_pixel(coord_end)
    dx, dy = xe - xs, ye - ys
    norme = np.hypot(dx, dy)
    ux, uy = dx / norme, dy / norme
    nx_p, ny_p = -uy, ux

    # 2. Masque géométrique sur la fente active [fenetre_arcsec]
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    sky_grille = wcs_cel.pixel_to_world(X, Y)
    dist_grille = sky_grille.separation(coord_star).arcsecond
    masque_fenetre = (dist_grille >= fenetre_arcsec[0]) & (dist_grille <= fenetre_arcsec[1])

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection=wcs_cel)

    # 3. Fond NIRSpec
    valides_ns = carte_ns[carte_ns > 0]
    v1, v2 = np.nanpercentile(valides_ns, [3, 98.5]) if valides_ns.size > 0 else (0, 1)
    ax.imshow(carte_ns, cmap='gray', origin='lower', vmin=v1, vmax=v2)

    # 4. Superposition des contours pour chaque raie MIRI configurée
    legend_handles = [
        Line2D([0], [0], color='none', label=f"Fond : {nom_raie_ns}"),
        Line2D([0], [0], color='red', lw=1.5, label="Axe de coupe 1D")
    ]

    for cfg in raies_miri_cfg:
        cle = cfg["cle"]
        couleur = cfg.get("couleur", "cyan")
        style = cfg.get("style", "-")
        nom = cfg.get("nom", cle)

        if cle in cartes_miri:
            carte_m = cartes_miri[cle]
            valides_m = carte_m[masque_fenetre & (carte_m > 0)]
            if valides_m.size > 0:
                seuils = np.percentile(valides_m, pct_contours)
                carte_clip = np.where(masque_fenetre, carte_m, np.nan)
                ax.contour(carte_clip, levels=seuils, colors=couleur,
                           linestyles=style, linewidths=1.8, zorder=12)
                legend_handles.append(
                    Line2D([0], [0], color=couleur, lw=2, ls=style, label=f"MIRI {nom}")
                )

    # 5. Tracé de l'axe de coupe rouge
    ax.plot(x_ext, y_ext, color='red', lw=1.5, zorder=15)

    # 6. Ticks stricts : élimination de tout point hors de l'image (0 <= x < nx, 0 <= y < ny)
    for cible in np.arange(105, 150, 1):
        idx = np.argmin(np.abs(dist_pts_arcsec - cible))
        if np.abs(dist_pts_arcsec[idx] - cible) < 0.15:
            xt, yt = x_ext[idx], y_ext[idx]
            
            # FILTRE STRICT POUR NE PAS TRACER DANS LE NOIR HORS MATRICE
            if 0 <= xt < nx and 0 <= yt < ny:
                taille = 4.5 if cible % 5 == 0 else 2.0
                lw = 1.2 if cible % 5 == 0 else 0.7
                ax.plot([xt + taille * nx_p, xt - taille * nx_p],
                        [yt + taille * ny_p, yt - taille * ny_p],
                        color='red', lw=lw, zorder=20)
                
                # Écriture uniquement entre 115" et 135" (inclus)
                if cible % 5 == 0 and 115 <= cible <= 135:
                    ax.text(xt + (taille + 6) * nx_p, yt + (taille + 6) * ny_p,
                            f"{int(cible)}\"", color='red', fontsize=8.5, weight='bold',
                            ha='center', va='center', zorder=25)

    ax.coords.grid(color='white', linestyle=':', linewidth=0.6, alpha=0.35)
    ax.coords[0].set_axislabel('Right Ascension (RA)', fontsize=11)
    ax.coords[1].set_axislabel('Declination (DEC)', fontsize=11)

    ax.set_xlim(nx - 0.5, -0.5)
    ax.set_ylim(-0.5, ny - 0.5)

    pct_str = ", ".join([f"{p}%" for p in pct_contours])
    ax.legend(handles=legend_handles, loc='upper right', facecolor='black',
              labelcolor='white', framealpha=0.85, fontsize=9,
              title=f"Contours [{pct_str}] ({fenetre_arcsec[0]:.0f}'' - {fenetre_arcsec[1]:.0f}'')")
    ax.set_title(r"Contrôle Spatial 2D Multi-Raies : NIRSpec vs MIRI $\mathrm{H}_2\ 0{-}0\,\mathrm{S}(1), \mathrm{S}(3), \mathrm{S}(8)$",
                 fontsize=12, weight='bold', pad=12)

    plt.tight_layout()
    return fig, ax


def extraire_banque_profils_1d(cube_ns, lambdas_ns, wcs_ref, cartes_miri,
                               coord_star, coord_start, coord_end,
                               catalogue_raies, l_obs_module,
                               v_lsr=10.5, v_min=0.0, v_max=30.0,
                               num_points=600, d_orion_pc=414.0):
    """
    Extrait les profils 1D le long du segment physique joignant coord_start et coord_end.
    Garantit que la couverture angulaire englobe bien la fenêtre [130'' - 137''].
    """
    wcs_cel = wcs_ref.celestial if hasattr(wcs_ref, 'celestial') else wcs_ref
    ny, nx = cube_ns.shape[1], cube_ns.shape[2]

    # 1. Échantillonnage céleste direct le long du vecteur de coupe
    v_ra = (coord_end.ra.deg - coord_start.ra.deg)
    v_dec = (coord_end.dec.deg - coord_start.dec.deg)

    # Facteurs d'extension pour couvrir de ~110" à ~140"
    alpha_vals = np.linspace(-0.35, 1.35, num_points)
    ra_pts = coord_start.ra.deg + alpha_vals * v_ra
    dec_pts = coord_start.dec.deg + alpha_vals * v_dec

    pts_sky = SkyCoord(ra_pts * u.deg, dec_pts * u.deg, frame='fk5')

    # Conversion en pixels natifs NIRSpec
    x_pts, y_pts = wcs_cel.world_to_pixel(pts_sky)

    # Distances absolues à Theta1 Ori C en 2D (séparation angulaire exacte sur la sphère céleste)
    dist_arcsec = pts_sky.separation(coord_star).arcsecond
    arcsec_to_pc = d_orion_pc / 206265.0
    dist_pc = dist_arcsec * arcsec_to_pc

    # Filtrage des points strictement situés à l'intérieur de la fente / matrice NIRSpec
    masque_in = (x_pts >= 0) & (x_pts < nx) & (y_pts >= 0) & (y_pts < ny)

    # Extraction NIRSpec 3D
    toutes_raies = list(catalogue_raies.keys())
    profils = l_obs_module.extraire_profils_bruts_3d(
        cube_ns, lambdas_ns, catalogue_raies, toutes_raies, x_pts, y_pts,
        v_min=v_min, v_max=v_max, v_lsr=v_lsr
    )

    # Pour les points hors matrice, on impose 0
    for k in profils:
        profils[k][~masque_in] = 0.0

    # Échantillonnage des cartes MIRI
    px_x = np.clip(np.round(x_pts).astype(int), 0, nx - 1)
    px_y = np.clip(np.round(y_pts).astype(int), 0, ny - 1)
    for cle, carte_m in cartes_miri.items():
        val_m = carte_m[px_y, px_x]
        val_m[~masque_in] = 0.0
        profils[f"{cle}_MIRI"] = np.nan_to_num(val_m, nan=0.0)

    print(f"-> Domaine extrait : [{np.nanmin(dist_arcsec):.2f}\", {np.nanmax(dist_arcsec):.2f}\"] "
          f"({np.sum((dist_arcsec >= 132.0) & (dist_arcsec <= 135.0))} points dans [132'', 135''])")

    return profils, dist_arcsec, dist_pc