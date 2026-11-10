#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:27:56 2025

@author: aurelien
"""

import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy as np 
from math import floor
import pandas as pd 
from sklearn.linear_model import LinearRegression
import pyismtools as ism

from labellines import labelLine, labelLines
from pyismtools.classes import *

import re
import os
import imageio.v2 as imageio # Use imageio.v2
import shutil # For managing temporary directory

from matplotlib import ticker
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import plotly.express as px

import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"


import glob

plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['text.usetex'] = True


"""
chemin_modules = "/Users/aurelien/Desktop/PhD/Codes/Package/Perso/"
sys.path.append(chemin_modules)
import read as read 

path = "/Users/aurelien/Desktop/Stage_M2/PDR/1.7/"
file_path = path +'all_co_lines_info.dat'
cols = ['Lines', 'vu', 'Ju', 'vl', 'Jl', 'gu', 'gl', 'Eu[K]', 'El[K]', 'Eul[K]', 'Aul[s-1]', 'Wavelenght[microns]']
df_info_co = pd.read_csv(file_path, 
                 skiprows=2, 
                 header=None, 
                 names=cols,
                 sep=r',\s*', 
                 engine='python')

int_cols = ['vu', 'Ju', 'vl', 'Jl', 'gu', 'gl']
for col in int_cols:
    df_info_co[col] = df_info_co[col].astype(int)
    
"""
    
#Pour Boltzman en CGS 
clum= 2.99792458e10
h = 6.62607015e-27
k = 1.380649e-16
k1 = 2.0*h*clum**2
k2 = h*clum/k

#Normalisation en Habing
Habings=5.6e-14 # Facteur de normalisation de Habings utilisé dans le code PDR 

#Autres
palette=["g","r","c","m","y","k","tab:orange","tab:brown","tab:grey","salmon","olive","gold","orchid"]#"b",
marqueur=['^','o']
hatch = ['/', '\\', '+', 'x', '.', '*']
#hatches = hatch[i % len(hatch)]



def generer_liste_couleurs_matplotlib(n_col,carte="viridis"):
    palette = plt.cm.get_cmap(carte, n_col)
    couleurs = [palette(i) for i in np.linspace(0, 1, n_col)]
    liste_couleurs = []

    for i in range(n_col):
      liste_couleurs.append(couleurs[i]) # on ajoute les couleurs de la palette en boucle

    return liste_couleurs

#liste_800_couleurs = generer_liste_couleurs_matplotlib(800,"viridis")

# def generer_liste_couleurs_matplotlib(n_col, carte="viridis"):
#     """Génère une liste de couleurs en utilisant une palette matplotlib."""
#     # Nouvelle méthode officielle de Matplotlib
#     palette = mpl.colormaps[carte]
    
#     couleurs = [palette(i) for i in np.linspace(0, 1, n_col)]
#     liste_couleurs = []

#     for i in range(n_col):
#       liste_couleurs.append(couleurs[i]) 

#     return liste_couleurs


def planck_function(wl, T, k1, k2):
    """Plot the Bloack body of T, according to the wavelength wl, and k1 and k2 not to move, 
    en cgs, prend wl en cm  et T en degrés pour retourner des erg cm-2 s-1 A-1"""
    radiance = k1 / ( ((wl*1e8)*(wl**4)) * (np.exp(k2/(wl*(T+273))) - 1.0) )
    return (radiance) 

def moyenne_glissante (val,intervalle):
    """Determine de avg from a too resolved spectrum : Params : val=List intensity, intervalle=nb of points to make the avg"""
    start=(intervalle-1)//2
    list_moy=[sum(val[i-start:i+start+1])/intervalle for i in range (start,len(val)-start)]
    return (list_moy)


def Grid_PthGo(name): 
    """Names taken = Full_Grid_Stage , SubGrid_Test"""
    if name not in ["SubGrid_Test","Full_Grid_Stage"] :
        print("Name wrongly written")
        return None
    if name=="Full_Grid_Stage" : 
        
        Go = [10, 20, 40, 80, 100, 200, 400, 800, 1e3, 2e3, 5e3, 1e4, 5e4, 1e5]
        Pth = [5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 2e8, 5e8, 8e8, 1e9, 5e9]
        
    elif name=="SubGrid_Test": 
        Go = [50, 150, 300, 900, 2700, 8100, 24300]
        Pth = [6e5, 1.8e6, 5.4e6, 1.6e7, 4.8e7, 1.4e8, 4.2e8]
         
    xx, yy = np.meshgrid(Pth, Go)

    print("Pth, Go dans l'ordre")
    return (xx.flatten(),yy.flatten)


def Annotate(xlist,ylist,namelist,decalage=(10,10),ax=None,fontsize=20,color="blue",alpha=1) :
    if ax == None : 
        for i,nom in enumerate(namelist):
            plt.annotate(nom, (xlist[i], ylist[i]), textcoords="offset points", xytext=decalage, ha='center',fontsize=fontsize,color=color,alpha=alpha)
    else :  
        for i,nom in enumerate(namelist):
            ax.annotate(nom, (xlist[i], ylist[i]), textcoords="offset points", xytext=decalage, ha='center',fontsize=fontsize,color=color,alpha=alpha)
            
            
# =============================================================================
#                        Params pour les label des figures 
# =============================================================================

def Setting_Figure(keyword): 
    plt.rcParams['axes.linewidth'] = 2     # Épaisseur des bords de la figure
    
    # Paramètres pour les ticks majeurs
    mpl.rcParams['xtick.major.size'] = 13    # Longueur des ticks X
    mpl.rcParams['xtick.major.width'] = 2  # Épaisseur des ticks X
    mpl.rcParams['ytick.major.size'] = 13    # Longueur des ticks Y
    mpl.rcParams['ytick.major.width'] = 2  # Épaisseur des ticks Y
    
    # Paramètres pour les ticks mineurs (optionnel)
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    
    mpl.rcParams['xtick.minor.size'] = 7     # Longueur des ticks X
    mpl.rcParams['xtick.minor.width'] = 1.5  # Épaisseur des ticks X
    mpl.rcParams['ytick.minor.size'] = 7     # Longueur des ticks Y
    mpl.rcParams['ytick.minor.width'] = 1.5  # Épaisseur des ticks Y

    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['axes.titlesize'] = 15
    mpl.rcParams['legend.fontsize'] = 15
    
    
    if keyword=="Intensity": 
        mpl.rcParams['xtick.labelsize']=10
        mpl.rcParams['ytick.labelsize']=20
        mpl.rcParams["axes.labelsize"]=25
        mpl.rcParams["figure.figsize"]=21,9


    elif keyword=="Contour" : 
        mpl.rcParams['xtick.labelsize']=30
        mpl.rcParams['ytick.labelsize']=30


    elif keyword=="Excitation":
        mpl.rcParams['xtick.labelsize']=30
        mpl.rcParams['ytick.labelsize']=30
        
        
    elif keyword=="Regular":
        mpl.rcParams['xtick.labelsize']=15
        mpl.rcParams['ytick.labelsize']=15       
        
    elif keyword=="Article" : 
        mpl.rcParams['xtick.labelsize'] = 25
        mpl.rcParams['ytick.labelsize'] = 25
        mpl.rcParams['legend.fontsize'] = 25
        mpl.rcParams['axes.titlesize'] = 25
        mpl.rcParams['font.size'] = 25


# =============================================================================
#                               Profil de densité 
# =============================================================================
def create_dens_prof_simple(z ,n0 , z0 , gamma) : 
    """Crée un profil de densité simple comme proposé par Meriem 2024"""
    if z< z0 : 
        return ( n0 * (z/z0) ** gamma)
    elif z> z0 : 
        return (n0) 



# =============================================================================
#                            Grid plot   
# =============================================================================    






# =============================================================================
#                           Diagramme d'intensité
# =============================================================================
  

def Intensity_Diagram (Model,Observation, Name="Test", VisOP=True, color="darkorange",line="--") :
    """ Plot the intensity diagram between the model and the observations with a sigma on obs, 
    Params : Model, Obs, Name, VisOP=True, color"""
    
    Raie_list=Model.get_observable_names()
    Pth1,Go1=Model.get_parameter_vector()
    Int_list=Model.get_observable_vector()
    scale=Model.get_scaling_factor()
    
    err_list=Observation.get_error_vector()
    Obs_list=Observation.get_observable_vector()   
     
    plt.figure(figsize=(21,9))
    if VisOP==True : 
        for i in range (len(Raie_list)):
            
            if float(Raie_list[i].split(' ')[1].split(')')[0])%2==0:#Si pair, para 
                plt.errorbar(Raie_list[i], Obs_list[i],yerr=(0.3)*Obs_list[i],
                             marker="X",color="red",ecolor="black",capsize=3)
            else : 
                plt.errorbar(Raie_list[i], Obs_list[i],yerr=(0.3)*Obs_list[i],#si impair, ortho
                             marker="o",color="blue",ecolor="black",capsize=3)
    else : 
        for i in range(len(Raie_list)):
            plt.errorbar(Raie_list[i], Obs_list[i],yerr=(0.3)*Obs_list[i],
                         marker="s",color="black",ecolor="black",capsize=3)
    
    plt.plot(Raie_list,Int_list,
             linestyle=line,color=color)
    
    plt.semilogy() 
    plt.xticks(rotation="vertical")
    plt.ylabel("Intensity [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    plt.title(Name,fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(linestyle='dotted', alpha=0.5)
    plt.tight_layout()
    plt.xlim(-1,len(Obs_list))

def Intensity_by_v(Model, Observation, Name="Test",color="navy",save=False,output="Desktop"):
    """ Plot the intensity diagram for each vibrationnal level v between the model and the observations with a sigma on obs, 
    Params : Model, Obs, Name, VisOP=True, color"""

    Raie_list=Model.get_observable_names()
    Pth1,Go1=Model.get_parameter_vector()
    Int_list=Model.get_observable_vector()
    scale=Model.get_scaling_factor()
    
    err_list=Observation.get_error_vector()
    Obs_list=Observation.get_observable_vector()
    
    Lines_Vibrationnel=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]
    Obs_vibrationnel=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]
    Err_vibrationnel=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]
    Model_vibrationnel=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]

    for i  in range (len(Raie_list)) : 
        Vh=int(Raie_list[i].split("-")[0])
        Lines_Vibrationnel[Vh].append(Raie_list[i])
        Obs_vibrationnel[Vh].append(Obs_list[i])
        Err_vibrationnel[Vh].append(err_list[i])
        Model_vibrationnel[Vh].append(Int_list[i])

    for i in range(1,len(Lines_Vibrationnel)):
        plt.figure(figsize=(20,9))
        
        plt.plot(Lines_Vibrationnel[i],Model_vibrationnel[i],'--',
                 label=Name,color=color)

        for j in range (len(Lines_Vibrationnel[i])):
            Raie=Lines_Vibrationnel[i]
            if float(Raie[j].split(' ')[1].split(')')[0])%2==0:#Si pair = para 
                plt.errorbar(Raie[j], Obs_vibrationnel[i][j],yerr=(0.3)*Obs_vibrationnel[i][j],marker="^",color="red",ecolor="black",
                             elinewidth=3,capsize=5,ms=20)
            else : #si impair : ortho 
                plt.errorbar(Raie[j], Obs_vibrationnel[i][j],yerr=(0.3)*Obs_vibrationnel[i][j],marker="o",color="blue",ecolor="black",
                             elinewidth=3,capsize=5,ms=20)
                     
        plt.ylabel(r"Intensity [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        plt.title(Name+f" Vup={i}",fontsize=20)
        plt.semilogy() 
        plt.legend(fontsize=15)
        plt.xticks(rotation="vertical")
        plt.grid(linestyle='dotted', alpha=0.7)
        plt.tight_layout()
        if save==True : 
            plt.savefig(output+"_"+Name+f"_v{i}.pdf")
       
            
def Intensity_full_v(Model, Observation, Name="Test" ,color="navy",save=False,output="Desktop",n2=False):
    """ Plot the intensity diagram for each vibrationnal level v between the model and the observations with a sigma on obs, 
    Params : Model, Obs, Name, VisOP=True, color"""
    mpl.rcParams['xtick.labelsize']=15
    mpl.rcParams['ytick.labelsize']=15
    mpl.rcParams["axes.labelsize"]=15
    
    mpl.rcParams['xtick.major.width']=2
    mpl.rcParams['xtick.minor.width']=1
    mpl.rcParams['xtick.minor.size']=7
    mpl.rcParams['xtick.major.size']=10

    mpl.rcParams['ytick.minor.width']=1
    mpl.rcParams['ytick.major.width']=2
    mpl.rcParams['ytick.minor.size']=5
    mpl.rcParams['ytick.major.size']=10
    
    Raie_list=Model.get_observable_names()
    Pth1,Go1=Model.get_parameter_vector()
    Int_list=Model.get_observable_vector()
    scale=Model.get_scaling_factor()
    
    err_list=Observation.get_error_vector()
    Obs_list=Observation.get_observable_vector()
    #Initialisation des matrices pour les niveaux v
    Lines_Vibrationnel=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]
    Obs_vibrationnel=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]
    Err_vibrationnel=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]
    Model_vibrationnel=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]
    if n2!= False : #n2 est un éventuel deuxième modèle à implémenter 
        Model_vibrationnel2=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]
        Int_list2=n2.get_observable_vector()

    for i  in range (len(Raie_list)) : 
        #Réparition des raies dans des listes par v 
        Vh=int(Raie_list[i].split("-")[0])
        Lines_Vibrationnel[Vh].append(Raie_list[i])
        Obs_vibrationnel[Vh].append(Obs_list[i])
        Err_vibrationnel[Vh].append(err_list[i])
        Model_vibrationnel[Vh].append(Int_list[i])
        
        if n2!= False : 
            Model_vibrationnel2[Vh].append(Int_list2[i])

    fig, axes = plt.subplots(nrows=2,ncols=6, figsize=(22, 9))
    #Création d'une figure avec 6 colonnes et 2 lignes pour chaque v
    for i in range(1,len(Lines_Vibrationnel)):   
        
        if i<=6 : #premiere ligne
            ax=axes[0, i-1]#modèle
            ax.plot(Lines_Vibrationnel[i],Model_vibrationnel[i],'--',linewidth=2,
                     label=Name,color=color)
            ax.grid(linestyle='dotted', alpha=0.7)

            if n2!=False: 
                ax.plot(Lines_Vibrationnel[i],Model_vibrationnel2[i],'--',linewidth=2,
                         label=Name,color="red")
                
                
            for j in range (len(Lines_Vibrationnel[i])):
                Raie=Lines_Vibrationnel[i]#Observations 
                if float(Raie[j].split(' ')[1].split(')')[0])%2==0:#Pair=para 
                    ax.errorbar(Raie[j], Obs_vibrationnel[i][j],yerr=(0.3)*Obs_vibrationnel[i][j],
                                          marker="^",color="black",ecolor="black",
                                 elinewidth=1.5,capsize=3,ms=10)
                else : 
                    ax.errorbar(Raie[j], Obs_vibrationnel[i][j],yerr=(0.3)*Obs_vibrationnel[i][j],
                                          marker="o",color="black",ecolor="black",
                                  elinewidth=1.5,capsize=3,ms=10)
                         
                    ax.set_title(f" Vup={i}")
                    ax.set_xticklabels(ax.get_xticklabels(),rotation="vertical") 
                    ax.semilogy() 
                    ax.grid(linestyle='dotted', alpha=0.7)

        elif i>6 : #Deuxieme ligne 
            ax=axes[1, i%6-1]#modèle
            ax.plot(Lines_Vibrationnel[i],Model_vibrationnel[i],'--',linewidth=2,
                     label=Name,color=color)
            ax.grid(linestyle='dotted', alpha=0.7)

            if n2!=False: 
                ax.plot(Lines_Vibrationnel[i],Model_vibrationnel2[i],'--',linewidth=2,
                         label=Name,color="red")
            for j in range (len(Lines_Vibrationnel[i])):#observations
                Raie=Lines_Vibrationnel[i]
                if float(Raie[j].split(' ')[1].split(')')[0])%2==0:
                    ax.errorbar(Raie[j], Obs_vibrationnel[i][j],yerr=(0.3)*Obs_vibrationnel[i][j],
                                marker="^",color="black",ecolor="black",
                                 elinewidth=1.5,capsize=3,ms=10)
                else : 
                   ax.errorbar(Raie[j], Obs_vibrationnel[i][j],yerr=(0.3)*Obs_vibrationnel[i][j],
                               marker="o",color="black",ecolor="black",
                                 elinewidth=1.5,capsize=3,ms=10)
                ax.set_title(f" Vup={i}")
                ax.semilogy() 
                ax.grid(linestyle='dotted', alpha=0.7)
                ax.set_xticklabels(ax.get_xticklabels(),rotation="vertical") 

        axes[0,0].set_ylabel(r"Intensity [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        axes[1,0].set_ylabel(r"Intensity [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    #    handles, labels = ax.get_legend_handles_labels()
     #   fig.legend(handles, labels, loc='upper center')
        plt.tight_layout()
        if save==True : 
            plt.savefig(output+"_"+Name+f"_v{i}.pdf")


            
def Intensity_v_separated(Model, Observation, Name="Test" ,color="navy",save=False,output="Desktop",n2=False):
    #TODO Refaire le code pour prendre autant de modele qu'on veut 
    """ Plot the intensity diagram for each vibrationnal level v between the model and the observations with a sigma on obs, 
    Params : Model, Obs, Name, VisOP=True, color"""
    mpl.rcParams['xtick.labelsize']=15
    mpl.rcParams['ytick.labelsize']=15
    mpl.rcParams["axes.labelsize"]=15
    
    mpl.rcParams['xtick.major.width']=2
    mpl.rcParams['xtick.minor.width']=1
    mpl.rcParams['xtick.minor.size']=7
    mpl.rcParams['xtick.major.size']=10

    mpl.rcParams['ytick.minor.width']=1
    mpl.rcParams['ytick.major.width']=2
    mpl.rcParams['ytick.minor.size']=5
    mpl.rcParams['ytick.major.size']=10
    
    Raie_list=Model.get_observable_names()
    Pth1,Go1=Model.get_parameter_vector()
    Int_list=Model.get_observable_vector()
    scale=Model.get_scaling_factor()
    
    err_list=Observation.get_error_vector()
    Obs_list=Observation.get_observable_vector()
    #Initialisation des matrices pour les niveaux v
    Lines_Vibrationnel=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]
    Obs_vibrationnel=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]
    Err_vibrationnel=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]
    Model_vibrationnel=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]
    if n2!= False : #n2 est un éventuel deuxième modèle à implémenter 
        Model_vibrationnel2=[[] for i in range(int(Raie_list[-1].split('-')[0])+1)]
        Int_list2=n2.get_observable_vector()

    for i  in range (len(Raie_list)) : 
        #Réparition des raies dans des listes par v 
        Vh=int(Raie_list[i].split("-")[0])
        Lines_Vibrationnel[Vh].append(Raie_list[i])
        Obs_vibrationnel[Vh].append(Obs_list[i])
        Err_vibrationnel[Vh].append(err_list[i])
        Model_vibrationnel[Vh].append(Int_list[i])
        
        if n2!= False : 
            Model_vibrationnel2[Vh].append(Int_list2[i])

    fig, axes = plt.subplots(nrows=2,ncols=6, figsize=(22, 9))
    #Création d'une figure avec 6 colonnes et 2 lignes pour chaque v
    for i in range(1,len(Lines_Vibrationnel)):   
        
        if i<=6 : #premiere ligne
            ax=axes[0, i-1]#modèle
            ax.plot(Lines_Vibrationnel[i],Model_vibrationnel[i],'--',linewidth=2,
                     label=Name,color=color)
            ax.grid(linestyle='dotted', alpha=0.7)

            if n2!=False: 
                ax.plot(Lines_Vibrationnel[i],Model_vibrationnel2[i],'--',linewidth=2,
                         label=Name,color="red")
                
                
            for j in range (len(Lines_Vibrationnel[i])):
                Raie=Lines_Vibrationnel[i]#Observations 
                if float(Raie[j].split(' ')[1].split(')')[0])%2==0:#Pair=para 
                    ax.errorbar(Raie[j], Obs_vibrationnel[i][j],yerr=(0.3)*Obs_vibrationnel[i][j],
                                          marker="^",color="black",ecolor="black",
                                 elinewidth=1.5,capsize=3,ms=10)
                else : 
                    ax.errorbar(Raie[j], Obs_vibrationnel[i][j],yerr=(0.3)*Obs_vibrationnel[i][j],
                                          marker="o",color="black",ecolor="black",
                                  elinewidth=1.5,capsize=3,ms=10)
                         
                    ax.set_title(f" Vup={i}")
                    ax.set_xticklabels(ax.get_xticklabels(),rotation="vertical") 
                    ax.semilogy() 
                    ax.grid(linestyle='dotted', alpha=0.7)

        elif i>6 : #Deuxieme ligne 
            ax=axes[1, i%6-1]#modèle
            ax.plot(Lines_Vibrationnel[i],Model_vibrationnel[i],'--',linewidth=2,
                     label=Name,color=color)
            ax.grid(linestyle='dotted', alpha=0.7)

            if n2!=False: 
                ax.plot(Lines_Vibrationnel[i],Model_vibrationnel2[i],'--',linewidth=2,
                         label=Name,color="red")
            for j in range (len(Lines_Vibrationnel[i])):#observations
                Raie=Lines_Vibrationnel[i]
                if float(Raie[j].split(' ')[1].split(')')[0])%2==0:
                    ax.errorbar(Raie[j], Obs_vibrationnel[i][j],yerr=(0.3)*Obs_vibrationnel[i][j],
                                marker="^",color="black",ecolor="black",
                                 elinewidth=1.5,capsize=3,ms=10)
                else : 
                   ax.errorbar(Raie[j], Obs_vibrationnel[i][j],yerr=(0.3)*Obs_vibrationnel[i][j],
                               marker="o",color="black",ecolor="black",
                                 elinewidth=1.5,capsize=3,ms=10)
                ax.set_title(f" Vup={i}")
                ax.semilogy() 
                ax.grid(linestyle='dotted', alpha=0.7)
                ax.set_xticklabels(ax.get_xticklabels(),rotation="vertical") 

        axes[0,0].set_ylabel(r"Intensity [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        axes[1,0].set_ylabel(r"Intensity [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    #    handles, labels = ax.get_legend_handles_labels()
     #   fig.legend(handles, labels, loc='upper center')
        plt.tight_layout()
        if save==True : 
            plt.savefig(output+"_"+Name+f"_v{i}.pdf")
            
            
            
# =============================================================================
#                       Diagramme d'excitation 
# ============================================================================    

def Excitation_Diagram(Model, Observation, NameList,style, Name=None,color="blue", Reg=False, Save=False, Path="",
                        Annotation=False, Full=False, Level=None, FillIdx =None, threshold=True,
                        Front="Model",Details=True,matchcolor=False,Size=(21,7),coeff_Y=1.2,thick=2,sizepoint=5,
                        mol="h2",OP_visible=True,
                        ObsCol=None) :
    clum= 2.99792458e10
    h = 6.62607015e-27
    
    """Crée un diagramme d'excitation en séparant par v, et en mettant en évidence les différences O et P"""
    palette=["g","r","c","m","y","k","tab:orange","tab:brown","tab:grey","salmon","olive","gold","orchid"]#"b",
    palette = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9","red","blue","green","black","dimgrey"]

    marqueur=['^','o']#Para = pair = 0 et ortho = impair = 1

    Raies=Observation.get_observable_names()
    I_Obs=Observation.get_observable_vector()
    E_list=Observation.get_error_vector()

    scaling_list,I_Model=[],[]
    if Model!=None:
        for i in range(len(Model)):
            
            scaling_list.append(Model[i].get_scaling_factor())
            I_Model.append(Model[i].get_observable_vector())
            for el in I_Model[i] : 
                el=el*scaling_list[i]
    
    NameList2= []
    for i in range (len(Model)) : 
        Params = Model[i].get_parameter_vector()
        echelle = Model[i].get_scaling_factor()
        if Details == True :
            #NameList2.append(r": P$_{\text{th}}$"+f" = {format(Params[0], '.2e')}" +r" K.$cm^{-3}$, $G_0$ = " + f"{format(Params[1],'.2e')}," +r" $ \Omega $"+f"={format(echelle,'.1e')}")
            NameList2.append(r": P$_{\text{th}}$" + f" = {Params[0]:.2e} K.$cm^{{-3}}$, $G_0$ = {Params[1]:.2e}, $\Omega$={echelle:.1e}")
        elif Details == False : 
            NameList2.append("")
            
    #Comparaison avec la data base du code
    main_root="/Users/aurelien/Desktop/Stage_M2/"
    rootbase=main_root+"PDR/1.7/"
    namebase=f"all_{mol}_lines_info"
    if mol == "co_new": 
        #rootbase = "/Users/aurelien/Desktop/PhD/Traitement/"
        #namebase = "all_co_lines_info_new"

        rootbase = "/Users/aurelien/Desktop/PhD/Data/"
        namebase ="co_lines_info_HITRAN"
    doc_base=open(rootbase+namebase+".dat","r")
    file_base=doc_base.read();  ligne_base=file_base.split("\n")[2:];  header_base=ligne_base[0:2]
    Data_base=[]
    """
    for i in range (len(Raies)) :
        temp=i
        nom_raie=Raies[i]
        #print("Raies",nom_raie)
        ligne_sans_espaces = ligne_base[temp].strip()
        Line = ligne_sans_espaces.split(',', 1)[0]
        print(f"Line : |{Line}|")
        
        while nom_raie!=Line and temp<len(ligne_base):
            temp+=1
            ligne_sans_espaces = ligne_base[temp].strip()
            Line = ligne_sans_espaces.split(',', 1)[0]
            
            print(f"Line : |{Line}|")

        if nom_raie==ligne_base[temp].split(',',1)[0] : 
            Data_base.append(ligne_base[temp])
        else : 
            print(f'Error : |{nom_raie}| Not in database, please check if nomenclature is correct.')
    """
    for i in range(len(Raies)):
        temp = i
        nom_raie = Raies[i]
    
        ligne_sans_espaces = ligne_base[temp].strip()
        Line = ligne_sans_espaces.split(',', 1)[0]
        #print(f"Line : |{Line}|")
    
    # Recherche de la ligne
        while nom_raie != Line and temp < len(ligne_base) - 1: # J'ai ajouté "- 1" ici (voir explication plus bas)
            temp += 1
            ligne_sans_espaces = ligne_base[temp].strip()
            Line = ligne_sans_espaces.split(',', 1)[0]
            #print(f"Line : |{Line}|")
    
        # CORRECTION ICI : on utilise la variable "Line" qui est déjà nettoyée
        if nom_raie == Line: 
            Data_base.append(ligne_base[temp])
        else: 
            print(f'Error : |{nom_raie}| Not in database, please check if nomenclature is correct.')
    #print("Data_Base",Data_base)
    
    
    
    
    #Récupération des paramètres nécessaires     
    Wave_list,Eul_list,A_list,Eu_list,gu_list,ju_list=[],[],[],[],[],[]    
    for i in range (len(Data_base)):
        raie_sel=Data_base[i].split(',')
        if mol == "co_new":
            Wave_list.append(float(raie_sel[11].split('\t')[-1]))
        else :
            Wave_list.append(float(raie_sel[11].split('\t')[-1]))

        A_list.append(float(raie_sel[10].split('\t')[-1]))
        Eul_list.append(float(raie_sel[9].split('\t')[-1]))
        Eu_list.append(float(raie_sel[7].split('\t')[-1]))
        gu_list.append(float(raie_sel[5].split('\t')[-1]))
        ju_list.append(float(raie_sel[2].split('\t')[-1]))
    

#Calcul de la colonne densité pour les observations dans Nu_Obs
    Nu_Obs=[4*np.pi * (Wave_list[i] * 1e-4)   * I_Obs[i]/(h*clum*A_list[i]) for i in range (len(A_list))] #Attnetion lambda data en micron -> cm 
    
    #print(raie_sel[0],A_list[0],gu_list[0],Wave_list[0],"\n")
    #print(format(I_Obs[0],'.2e'),format(Nu_Obs[0],'.2e'))
    #print(format(I_Obs[1],'.2e'),format(Nu_Obs[1],'.2e'))

    if threshold ==True : 
        Nu_Err=[4*np.pi * (Wave_list[i] * 1e-4)  * (0.3*I_Obs[i])/(h*clum*A_list[i]) for i in range (len(A_list))]
    elif threshold ==False : 
        Nu_Err=[4*np.pi * (Wave_list[i] * 1e-4)  * (E_list[i])/(h*clum*A_list[i]) for i in range (len(A_list))]
    
    Y_Obs=[(Nu_Obs[i]/gu_list[i]) for i in range (len(Nu_Obs))]
    #print(format(Y_Obs[0],".2e"))
    Y_Err=[(Nu_Err[i]/gu_list[i]) for i in range (len(Nu_Obs))]
    
    Nu_Model,Y_Model=[],[]
    for k in range (len(Model)):
        Nu_Model.append( [4*np.pi * (Wave_list[i] * 1e-4) *I_Model[k][i]/(h*clum*A_list[i]) for i in range (len(A_list))] )
        Y_Model.append([(Nu_Model[k][i]/gu_list[i]) for i in range (len(Nu_Obs))] )
    
    
    max_v = max([int(r.split("-")[0]) for r in Raies]) + 1

    Lines_Vibrationnel=[[] for i in range(max_v)]
    Y_Obs_vibrationnel=[[] for i in range(max_v)]
    Error_Vib=[[] for i in range(max_v)]
    E_vibrationnel=[[] for i in range(max_v)]

    #Répartition par niveau vibrationnel 
    #Lines_Vibrationnel=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
    #Y_Obs_vibrationnel=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
    
    #Error_Vib=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
    #E_vibrationnel=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
    Y_model_vibrationnel=[ [ [] for i in range(int(Raies[-1].split('-')[0])+1) ] for k in range (len(Model))]
    #print(len(Raies))
    #print(Raies)
    for i in range (len(Raies)): 
        Vh=int(Raies[i].split("-")[0])
        #print(len(Y_Obs_vibrationnel[Vh]))
        #print(Vh)
        #print(i)
        Lines_Vibrationnel[Vh].append(Raies[i])
        Y_Obs_vibrationnel[Vh].append(Y_Obs[i])
        Error_Vib[Vh].append(Y_Err[i])
        E_vibrationnel[Vh].append(Eu_list[i])
        for k in range(len(Model)):
            Y_model_vibrationnel[k][Vh].append(Y_Model[k][i])
    
    compteur = 0
    while E_vibrationnel[0]== [] : #Check si rotationel ou pas 
        compteur +=1
        E_vibrationnel = E_vibrationnel[1:]   
        Y_Obs_vibrationnel = Y_Obs_vibrationnel[1:]
        Lines_Vibrationnel = Lines_Vibrationnel[1:]
        Error_Vib = Error_Vib[1:]
        for k in range (len(Model)):
            Y_model_vibrationnel[k]=Y_model_vibrationnel[k][1:]

    resid_list=[]
    #print(compteur)
    if Level==None : 
        minV = 0
        maxV=len(Lines_Vibrationnel)
        
    elif Level!= None : 
        minV=Level[0]
        maxV=Level[1]
    
    
    if Front=="Model": 
        order =[1,2]
    elif Front=="Obs":
        order=[2,1]
        
        #-------------- Graphique----------------------------------
    if ObsCol!=None : 
        palette=[ObsCol for i in range (20)]
        
    if Full==True : #Tracé de tous les niveaux sur le meme diagramme 
        fig, ax = plt.subplots(figsize=Size)
        
        ax.scatter(E_vibrationnel[0][0],Y_Obs_vibrationnel[0][0] ,c=palette[0],
                     marker=marqueur[0], label= "Para levels")
        
        ax.scatter(E_vibrationnel[0][0],Y_Obs_vibrationnel[0][0] ,c=palette[0],
                     marker=marqueur[1], label= "Ortho levels")
        
        for k in range(len(Model)):
            ax.plot(E_vibrationnel[0],Y_model_vibrationnel[k][0],linestyle=style[k],
                     color=color[k],alpha=0.9,label={NameList[k]+NameList2[k]},marker="X",linewidth=thick,markersize=sizepoint)
            
            
        if FillIdx!=None :
             #print("\nE_vibrationnel[1]:", E_vibrationnel[1])
             #print("\nY_model_vibrationnel[FillIdx[0]][1]:", Y_model_vibrationnel[FillIdx[0]][1])
             #print("\nY_model_vibrationnel[FillIdx[1]][1]:", Y_model_vibrationnel[FillIdx[1]][1])
             """
             plt.fill_between(np.array(E_vibrationnel[1]),
                              np.array(Y_model_vibrationnel[ FillIdx[0] ][1]),
                              np.array(Y_model_vibrationnel[ FillIdx[1] ][1]),
                              'peachpuff', 
                              alpha=0.5)
    
            """
            
    #print(minV,maxV)
    #print(palette)
    for i in range (minV,maxV): #Par V
        if Full==False : #Tracé par v séparé
            fig, ax = plt.subplots(figsize=Size)#15,6.5))
            
            
    
        for j in range (len(Lines_Vibrationnel[i])):# On trace les observations par niveau vibrationnel
            #print(j)    
            if OP_visible ==True :
                OP=int((Lines_Vibrationnel[i][j].split(',')[0].split(' ')[1]).split(')')[0])%2
            else :
                OP=0
            ax.errorbar(E_vibrationnel[i][j],Y_Obs_vibrationnel[i][j] , yerr=Error_Vib[i][j],c=palette[i],
                         markersize=15,marker=marqueur[OP],markeredgecolor='white',
                         capsize=3, elinewidth= 2 , ecolor=(0, 0, 0, 0.7),zorder=order[0])
            
            yerr = np.array([np.array([0.5 * Y_Obs_vibrationnel[i][j]]), np.array([1 * Y_Obs_vibrationnel[i][j]]) ])
            #yerr = np.array([np.array([Y_Obs_vibrationnel[i][j]])])
    
            ax.errorbar(E_vibrationnel[i][j],Y_Obs_vibrationnel[i][j] , yerr=yerr,c=palette[i],
                         fmt="none", capsize=1, elinewidth= 2 , ecolor=palette[i],zorder=order[0],alpha=0.2)

            ax.annotate(text=f"v={compteur+i}", xy=(E_vibrationnel[i][0],Y_Obs_vibrationnel[i][0]),
                        xytext=(E_vibrationnel[i][0]+500,Y_Obs_vibrationnel[i][0]*1.8),
                        color=palette[i],fontweight='bold',fontsize=25)
            
        for k in range(len(Model)) :   
            if matchcolor == True : 
                color = palette
                idxcolor = i
            elif matchcolor ==False : 
                idxcolor = k
                
            if Full==False : 
                ax.plot(E_vibrationnel[i],Y_model_vibrationnel[k][i],linestyle=style[k],color=color[idxcolor],alpha=0.9,
                        label={NameList[k]+NameList2[k]},zorder=order[1],marker="+",linewidth=thick,markersize=sizepoint)
                #labelLines(ax.get_lines(), zorder=2)
    
            elif Full==True : #Que pour le label 
                ax.plot(E_vibrationnel[i],Y_model_vibrationnel[k][i],linestyle=style[k],
                        color=color[idxcolor],alpha=0.9,zorder=order[1],marker="+",linewidth=thick,markersize=sizepoint)    
                
    
                #labelLines(ax.get_lines(), zorder=2)
    
            
        if Reg==True and len(E_vibrationnel[i])>2:#Si on ajoute la régression
            X=np.array(E_vibrationnel[i]).reshape(-1,1)
            model = LinearRegression();model.fit(X,np.log10(Y_Obs_vibrationnel[i]))
            coef = format(model.coef_[0],'.2e');r2 =int( model.score(X, np.log10(Y_Obs_vibrationnel[i])*100))
            y_pred = 10**model.predict(X)
            residuals=(np.log10(abs(Y_Obs_vibrationnel[i] - y_pred)))
    
            resid_list.append(residuals)
            ax.plot(X, y_pred, color='red',linestyle=":",label=f"Vup={i},T={-int(1/float(coef))} K ")
            
        xmin,xmax,ymin,ymax=np.min(E_list),np.max(E_list),np.min(Y_Obs)/2,np.max(Y_Obs)*3
    
        if Annotation==True :             #Annotation niveaux 
            for j in range (len(E_vibrationnel[i])): 
                if Full==True : 
                    ymax=np.max(Y_Obs_vibrationnel[0])
                ax.axvline(x=E_vibrationnel[i][j], color='black', linewidth=1, ymin=0.85,ymax=0.9)
                ax.annotate(Lines_Vibrationnel[i][j],
                             xy=( E_vibrationnel[i][j] , (ymax)*1.25 ), 
                             xytext=(E_vibrationnel[i][j] + 20 ,  (ymax)*1.3 ),
                             fontsize=10,rotation=90)
    
        #ax.grid(linestyle='dotted', alpha=0.7)
        ax.semilogy()
        ax.set_ylim(ymin/coeff_Y,ymax*coeff_Y)
    
    
            
        if Full==False : 
            plt.annotate(f"{Name} Excitation Vup={i}",xy=(0.5,0.9))
            plt.legend(fontsize=20)
            mpl.rcParams['xtick.labelsize']=25
            mpl.rcParams['ytick.labelsize']=25
            ax.set_ylabel(r'$N_u / g_u$ $[cm^{-2}]$',fontsize=25)
            ax.set_xlabel(r"$E_u$ [K]",fontsize=25)
    
    
    
        elif Full ==True : 
            if Name !=None : 
                plt.title(f"{Name} Excitation Diagram all v")
            ax.legend(fontsize=20)
            ax.set_ylabel(r'$N_u / g_u$ $[cm^{-2}]$',fontsize=25)
            ax.set_xlabel(r"$E_u$ [K]",fontsize=25)
    
        plt.tight_layout()
        
        if Save==True : 
            if Full==True : 
                plt.savefig(Path+Name+f"_excitation_diagram.pdf",dpi=200)
            elif  Full==False : 
                plt.savefig(Path+Name+f"_excitation_diagram_v{i}.pdf",dpi=200)        
    if Reg==True :     
        residus=[]
        for el in resid_list:
            for el2 in el :
                residus.append(el2)
        print("Moyenne, Max, Longueur des résidus écart (obs-LinReg)")
        #print(residus)
        print(np.mean(residus),np.max(residus),len(residus))
                
            
            
            
            
            
            
            
            
            
            
def Excitation_Diagram_test(Model, Observation, NameList,style, Name=None,color="blue", Reg=False, Save=False, Path="",
                        Annotation=False, Full=False, Level=None,FillIdx =None, threshold=True,
                        Front="Model",Details=True,matchcolor=False,mol="h2",OP_visible=True,
                        ObsCol=None,Size=(21,7),coeff_Y=1.5) :
    """Crée un diagramme d'excitation en séparant par v, et en mettant en évidence les différences O et P"""
    palette=["g","r","c","m","y","k","tab:orange","tab:brown","tab:grey","salmon","olive","gold","orchid"]#"b",
    palette=["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]#"b",
    if OP_visible == True : 
        
        marqueur=['^','o']#Para = pair = 0 et ortho = impair = 1
    else : 
        marqueur = ["o","o"]
    Raies=Observation.get_observable_names()
    I_Obs=Observation.get_observable_vector()
    E_list=Observation.get_error_vector()

    scaling_list,I_Model=[],[]
    if Model!=None:
        for i in range(len(Model)):
            
            scaling_list.append(Model[i].get_scaling_factor())
            I_Model.append(Model[i].get_observable_vector())
            for el in I_Model[i] : 
                el=el*scaling_list[i]
    
    NameList2= []
    for i in range (len(Model)) : 
        Params = Model[i].get_parameter_vector()
        echelle = Model[i].get_scaling_factor()
        if Details == True :
            #NameList2.append(r": P$_{\text{th}}$"+f" = {format(Params[0], '.2e')}" +r" K.$cm^{-3}$, $G_0$ = " + f"{format(Params[1],'.2e')}," +r" $ \Omega $"+f"={format(echelle,'.1e')}")
            NameList2.append(r": P$_{\text{th}}$" + f" = {Params[0]:.2e} K.$cm^{{-3}}$, $G_0$ = {Params[1]:.2e}, $\Omega$={echelle:.1e}")
        elif Details == False : 
            NameList2.append("")
            
    #Comparaison avec la data base du code
    main_root="/Users/aurelien/Desktop/Stage_M2/"
    rootbase=main_root+"PDR/1.7/"
    namebase=f"all_{mol}_lines_info"
    if mol == "co_new": 
        rootbase = "/Users/aurelien/Desktop/PhD/Traitement/"
        namebase = "all_co_lines_info_new"
    doc_base=open(rootbase+namebase+".dat","r")
    file_base=doc_base.read();  ligne_base=file_base.split("\n");  header_base=ligne_base[0:2]
     
    Data_base=[]
    for i in range (len(Raies)) :
        temp=i
        nom_raie=Raies[i]

        ligne_sans_espaces = ligne_base[temp].strip()
        Line = ligne_sans_espaces.split(',', 1)[0]
        while nom_raie!=Line and temp<len(ligne_base):
            #print(Line)    
            temp+=1
            ligne_sans_espaces = ligne_base[temp].strip()
            Line = ligne_sans_espaces.split(',', 1)[0]
        if nom_raie==ligne_base[temp].split(',')[0] : 
            Data_base.append(ligne_base[temp])
        else : 
            print('Error : Not in database, please check if nomenclature is correct.')
            

    #Récupération des paramètres nécessaires     
    Wave_list,Eul_list,A_list,Eu_list,gu_list,ju_list=[],[],[],[],[],[]    
    for i in range (len(Data_base)):
        raie_sel=Data_base[i].split(',')
        Wave_list.append(float(raie_sel[11].split('\t')[-1]))
        A_list.append(float(raie_sel[10].split('\t')[-1]))
        Eul_list.append(float(raie_sel[9].split('\t')[-1]))
        Eu_list.append(float(raie_sel[7].split('\t')[-1]))
        gu_list.append(float(raie_sel[5].split('\t')[-1]))
        ju_list.append(float(raie_sel[2].split('\t')[-1]))
        
    #Calcul de la colonne densité pour les observations dans Nu_Obs
    Nu_Obs=[4*np.pi * (Wave_list[i] * 1e-4)   * I_Obs[i]/(h*clum*A_list[i]) for i in range (len(A_list))] #Attnetion lambda data en micron -> cm 
    if threshold ==True : 
        Nu_Err=[4*np.pi * (Wave_list[i] * 1e-4)  * (0.3*I_Obs[i])/(h*clum*A_list[i]) for i in range (len(A_list))]
    elif threshold ==False : 
        Nu_Err=[4*np.pi * (Wave_list[i] * 1e-4)  * (E_list[i])/(h*clum*A_list[i]) for i in range (len(A_list))]
        #print(Nu_Err)
    
    Y_Obs=[(Nu_Obs[i]/gu_list[i]) for i in range (len(Nu_Obs))]
    Y_Err=[(Nu_Err[i]/gu_list[i]) for i in range (len(Nu_Obs))]
    
    Nu_Model,Y_Model=[],[]
    for k in range (len(Model)):
        Nu_Model.append( [4*np.pi * (Wave_list[i] * 1e-4) *I_Model[k][i]/(h*clum*A_list[i]) for i in range (len(A_list))] )
        Y_Model.append([(Nu_Model[k][i]/gu_list[i]) for i in range (len(Nu_Obs))] )
    
    #Répartition par niveau vibrationnel 
    Lines_Vibrationnel=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
    Y_Obs_vibrationnel=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
    Error_Vib=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
    E_vibrationnel=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
    Y_model_vibrationnel=[ [ [] for i in range(int(Raies[-1].split('-')[0])+1) ] for k in range (len(Model))]
    
    for i in range (len(Raies)): 
        Vh=int(Raies[i].split("-")[0])
        Lines_Vibrationnel[Vh].append(Raies[i])
        Y_Obs_vibrationnel[Vh].append(Y_Obs[i])
        Error_Vib[Vh].append(Y_Err[i])
        E_vibrationnel[Vh].append(Eu_list[i])
        for k in range(len(Model)):
            Y_model_vibrationnel[k][Vh].append(Y_Model[k][i])
    
    compteur = 0
    while E_vibrationnel[0]== [] : #Check si rotationel ou pas 
        compteur +=1
        E_vibrationnel = E_vibrationnel[1:]   
        Y_Obs_vibrationnel = Y_Obs_vibrationnel[1:]
        Lines_Vibrationnel = Lines_Vibrationnel[1:]
        Error_Vib = Error_Vib[1:]
        for k in range (len(Model)):
            Y_model_vibrationnel[k]=Y_model_vibrationnel[k][1:]

    resid_list=[]
    
    if Level==None : 
        minV = 0
        maxV=len(Lines_Vibrationnel)
        
    elif Level!= None : 
        minV=Level[0]
        maxV=Level[1]
    
    
    if Front=="Model": 
        order =[1,2]
    elif Front=="Obs":
        order=[2,1]
        
        #-------------- Graphique----------------------------------
    if ObsCol!=None : 
        palette=[ObsCol for i in range (20)]
    
    if Full==True : #Tracé de tous les niveaux sur le meme diagramme 
        fig, ax = plt.subplots(figsize=Size)
        

        if OP_visible==True : 
            ax.scatter(E_vibrationnel[0][0],Y_Obs_vibrationnel[0][0] ,c=palette[0+compteur],
                         marker=marqueur[0], label= "Para levels")
            ax.scatter(E_vibrationnel[0][0],Y_Obs_vibrationnel[0][0] ,c=palette[0+compteur],
                       marker=marqueur[1], label= "Ortho levels")
        else :
            ax.scatter(E_vibrationnel[0][0],Y_Obs_vibrationnel[0][0] ,c=palette[0+compteur],
                         marker=marqueur[0], label= "Observations")
        for k in range(len(Model)):
            ax.plot(E_vibrationnel[0],Y_model_vibrationnel[k][0],linestyle=style[k],
                     color=color[k],alpha=0.9,label={NameList[k]+NameList2[k]},marker="+")
            
            
        if FillIdx!=None :
             #print("\nE_vibrationnel[1]:", E_vibrationnel[1])
             #print("\nY_model_vibrationnel[FillIdx[0]][1]:", Y_model_vibrationnel[FillIdx[0]][1])
             #print("\nY_model_vibrationnel[FillIdx[1]][1]:", Y_model_vibrationnel[FillIdx[1]][1])
             """
             plt.fill_between(np.array(E_vibrationnel[1]),
                              np.array(Y_model_vibrationnel[ FillIdx[0] ][1]),
                              np.array(Y_model_vibrationnel[ FillIdx[1] ][1]),
                              'peachpuff', 
                              alpha=0.5)
    
            """
            
            
    for i in range (minV,maxV): #Par V
        if Full==False : #Tracé par v séparé
            fig, ax = plt.subplots(figsize=(15,6.5))
            
    
    
        for j in range (len(Lines_Vibrationnel[i])):# On trace les observations par niveau vibrationnel
            if OP_visible ==True :
                OP=int((Lines_Vibrationnel[i][j].split(',')[0].split(' ')[1]).split(')')[0])%2
            else :
                OP=0
            ax.errorbar(E_vibrationnel[i][j],Y_Obs_vibrationnel[i][j] , yerr=Error_Vib[i][j],c=palette[i+compteur],
                         markersize=15,marker=marqueur[OP],markeredgecolor='white',
                         capsize=3, elinewidth= 2 , ecolor=(0, 0, 0, 0.7),zorder=order[0])
            
            yerr = np.array([np.array([0.5 * Y_Obs_vibrationnel[i][j]]), np.array([1 * Y_Obs_vibrationnel[i][j]]) ])
            #yerr = np.array([np.array([Y_Obs_vibrationnel[i][j]])])
    
            ax.errorbar(E_vibrationnel[i][j],Y_Obs_vibrationnel[i][j] , yerr=yerr,c=palette[i],
                         fmt="none", capsize=1, elinewidth= 2 , ecolor=palette[i+compteur],zorder=order[0],alpha=0.2)

            ax.annotate(text=f"v={i+compteur}", xy=(E_vibrationnel[i][0],Y_Obs_vibrationnel[i][0]),
                        xytext=(np.mean(E_vibrationnel[i])+20,np.mean(Y_Obs_vibrationnel[i])*1.8),
                        color=palette[i+compteur],fontweight='bold')
            
        for k in range(len(Model)) :   
            if matchcolor == True : 
                color = palette
                idxcolor = i
            elif matchcolor ==False : 
                idxcolor = k
                
            if Full==False : 
                ax.plot(E_vibrationnel[i],Y_model_vibrationnel[k][i],linestyle=style[k],color=color[idxcolor],alpha=0.9,
                        label={NameList[k]+NameList2[k]},zorder=order[1],marker="x")
                #labelLines(ax.get_lines(), zorder=2)
    
            elif Full==True : #Que pour le label 
                ax.plot(E_vibrationnel[i],Y_model_vibrationnel[k][i],linestyle=style[k],
                        color=color[idxcolor],alpha=0.9,zorder=order[1],marker="x",markersize=6)    
                
    
                #labelLines(ax.get_lines(), zorder=2)
    
            
        if Reg==True and len(E_vibrationnel[i])>2:#Si on ajoute la régression
            X=np.array(E_vibrationnel[i]).reshape(-1,1)
            model = LinearRegression();model.fit(X,np.log10(Y_Obs_vibrationnel[i]))
            coef = format(model.coef_[0],'.2e');r2 =int( model.score(X, np.log10(Y_Obs_vibrationnel[i])*100))
            y_pred = 10**model.predict(X)
            residuals=(np.log10(abs(Y_Obs_vibrationnel[i] - y_pred)))
    
            resid_list.append(residuals)
            ax.plot(X, y_pred, color='red',linestyle=":",label=f"Vup={i},T={-int(1/float(coef))} K ")
            
        xmin,xmax,ymin,ymax=np.min(E_list),np.max(E_list),np.min(Y_Obs)/2,np.max(Y_Obs)*3
    
        if Annotation==True :             #Annotation niveaux 
            for j in range (len(E_vibrationnel[i])): 
                if Full==True : 
                    ymax=np.max(Y_Obs_vibrationnel[0])
                ax.axvline(x=E_vibrationnel[i][j], color='black', linewidth=1, ymin=0.85,ymax=0.9)
                ax.annotate(Lines_Vibrationnel[i][j],
                             xy=( E_vibrationnel[i][j] , (ymax)*1.25 ), 
                             xytext=(E_vibrationnel[i][j]  ,  (ymax)*1.3 ),
                             fontsize=10,rotation=90)
    
        ax.grid(linestyle='dotted', alpha=0.7)
        ax.semilogy()
        ax.set_ylim(ymin/coeff_Y,ymax*coeff_Y)
        #.set_xlim(,xmin+500)
        x_min_actuel = np.min(E_vibrationnel[i])
        x_max_actuel = np.max(E_vibrationnel[i])
        
        ax.set_xlim(x_min_actuel - 100, x_max_actuel + 100)
    
            
        if Full==False : 
            plt.annotate(f"{Name} Excitation Vup={i}",xy=(0.5,0.9))
            plt.legend(fontsize=20)
            mpl.rcParams['xtick.labelsize']=25
            mpl.rcParams['ytick.labelsize']=25
            ax.set_ylabel(r'$N_u / g_u$ $[cm^{-2}]$',fontsize=25)
            ax.set_xlabel(r"$E_u$ [K]",fontsize=25)
    
    
    
        elif Full ==True : 
            if Name !=None : 
                plt.title(f"{Name} Excitation Diagram all v")
            ax.legend(fontsize=20)
            ax.set_ylabel(r'$N_u / g_u$ $[cm^{-2}]$',fontsize=25)
            ax.set_xlabel(r"$E_u$ [K]",fontsize=25)
    
        plt.tight_layout()
        
        if Save==True : 
            if Full==True : 
                plt.savefig(Path+Name+f"_excitation_diagram.pdf",dpi=200)
            elif  Full==False : 
                plt.savefig(Path+Name+f"_excitation_diagram_v{i}.pdf",dpi=200)        
    if Reg==True :     
        residus=[]
        for el in resid_list:
            for el2 in el :
                residus.append(el2)
        print("Moyenne, Max, Longueur des résidus écart (obs-LinReg)")
        #print(residus)
        print(np.mean(residus),np.max(residus),len(residus))
                
            
                        
            
            

def Excitation_Diagram2(Model, Observation, NameList, style, Name=None, color="blue", Reg=False, Save=False, Path="",
                       Annotation=False, Full=False, Level=None, FillIdx=None, threshold=True,
                       Front="Model", Details=True, matchcolor=False, mol="h2",
                       OP_visible=True):
    """
    Crée un diagramme d'excitation en séparant par v, et en mettant en évidence les différences O et P.
    """
    marqueur = ['^', 'o']
    Raies = Observation.get_observable_names()
    I_Obs = Observation.get_observable_vector()
    E_list = Observation.get_error_vector()

    scaling_list, I_Model = [], []
    if Model is not None:
        for i in range(len(Model)):
            scaling_list.append(Model[i].get_scaling_factor())
            temp_I_Model = Model[i].get_observable_vector()
            
            scaled_vector = [val * scaling_list[i] for val in temp_I_Model]
            I_Model.append(scaled_vector)
            
    NameList2 = []
    if Model is not None:
        for i in range(len(Model)):
            Params = Model[i].get_parameter_vector()
            echelle = Model[i].get_scaling_factor()
            if Details:
                NameList2.append(
                    r": P$_{\text{th}}$" + f" = {Params[0]:.2e} K.$cm^{{-3}}$, $G_0$ = {Params[1]:.2e}, $\Omega$={echelle:.1e}"
                )
            else:
                NameList2.append("")

    # Comparaison avec la data base du code
    main_root = "/Users/aurelien/Desktop/Stage_M2/"
    rootbase = main_root + "PDR/1.7/"
    namebase = f"all_{mol}_lines_info"
    doc_base_path = f"{rootbase}{namebase}.dat"

    db_lookup = {}
    try:
        with open(doc_base_path, 'r') as doc_base:
            for ligne in doc_base:
                # Ignorer les lignes de commentaires ou les lignes vides
                if ligne.strip() and not ligne.strip().startswith('#'):
                    match = re.search(r'^(.*?)\s*,', ligne)
                    if match:
                        nom_raie = match.group(1).strip()
                        db_lookup[nom_raie] = ligne.strip()
    except FileNotFoundError:
        print(f"Erreur : Le fichier de base de données '{doc_base_path}' n'a pas été trouvé.")
        return 
    
    

    Data_base = []
    
    for nom_raie in Raies:
        ligne_trouvee = db_lookup.get(nom_raie)
        if ligne_trouvee:
            Data_base.append(ligne_trouvee)
        else:
            print(f"Erreur : La raie '{nom_raie.strip()}' n'a pas été trouvée dans la base de données.")
    #Récupération des paramètres nécessaires     
    Wave_list,Eul_list,A_list,Eu_list,gu_list,ju_list=[],[],[],[],[],[]    
    for i in range (len(Data_base)):
        raie_sel=Data_base[i].split(',')
        Wave_list.append(float(raie_sel[11].split('\t')[-1]))
        A_list.append(float(raie_sel[10].split('\t')[-1]))
        Eul_list.append(float(raie_sel[9].split('\t')[-1]))
        Eu_list.append(float(raie_sel[7].split('\t')[-1]))
        gu_list.append(float(raie_sel[5].split('\t')[-1]))
        ju_list.append(float(raie_sel[2].split('\t')[-1]))
        
    #Calcul de la colonne densité pour les observations dans Nu_Obs
    Nu_Obs=[4*np.pi * (Wave_list[i] * 1e-4)   * I_Obs[i]/(h*clum*A_list[i]) for i in range (len(A_list))] #Attnetion lambda data en micron -> cm 
    if threshold ==True : 
        Nu_Err=[4*np.pi * (Wave_list[i] * 1e-4)  * (0.3*I_Obs[i])/(h*clum*A_list[i]) for i in range (len(A_list))]
    elif threshold ==False : 
        Nu_Err=[4*np.pi * (Wave_list[i] * 1e-4)  * (E_list[i])/(h*clum*A_list[i]) for i in range (len(A_list))]

    Y_Obs=[(Nu_Obs[i]/gu_list[i]) for i in range (len(Nu_Obs))]
    Y_Err=[(Nu_Err[i]/gu_list[i]) for i in range (len(Nu_Obs))]

    Nu_Model,Y_Model=[],[]
    for k in range (len(Model)):
        Nu_Model.append( [4*np.pi * (Wave_list[i] * 1e-4) *I_Model[k][i]/(h*clum*A_list[i]) for i in range (len(A_list))] )
        Y_Model.append([(Nu_Model[k][i]/gu_list[i]) for i in range (len(Nu_Obs))] )
    
    #Répartition par niveau vibrationnel 
    Lines_Vibrationnel=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
    Y_Obs_vibrationnel=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
    Error_Vib=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
    E_vibrationnel=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
    Y_model_vibrationnel=[ [ [] for i in range(int(Raies[-1].split('-')[0])+1) ] for k in range (len(Model))]
    
    for i in range (len(Raies)): 
        Vh=int(Raies[i].split("-")[0])
        Lines_Vibrationnel[Vh].append(Raies[i])
        Y_Obs_vibrationnel[Vh].append(Y_Obs[i])
        Error_Vib[Vh].append(Y_Err[i])
        E_vibrationnel[Vh].append(Eu_list[i])
        for k in range(len(Model)):
            Y_model_vibrationnel[k][Vh].append(Y_Model[k][i])
                
    if E_vibrationnel[0]== [] : #Check si rotationel ou pas 
        E_vibrationnel = E_vibrationnel[1:]   
        Y_Obs_vibrationnel = Y_Obs_vibrationnel[1:]
        Lines_Vibrationnel = Lines_Vibrationnel[1:]
        Error_Vib = Error_Vib[1:]
        for k in range (len(Model)):
            Y_model_vibrationnel[k]=Y_model_vibrationnel[k][1:]
            
            
    resid_list=[]
    
    if Level==None : 
        minV = 0
        maxV=len(Lines_Vibrationnel)
        
    elif Level!= None : 
        minV=Level[0]
        maxV=Level[1]


    if Front=="Model": 
        order =[1,2]
    elif Front=="Obs":
        order=[2,1]
        
        #-------------- Graphique----------------------------------
    if Full==True : #Tracé de tous les niveaux sur le meme diagramme 
        fig, ax = plt.subplots(figsize=(21,9))
        
        ax.scatter(E_vibrationnel[0][0],Y_Obs_vibrationnel[0][0] ,c=palette[0],
                     marker=marqueur[0], label= "Para levels")
        
        ax.scatter(E_vibrationnel[0][0],Y_Obs_vibrationnel[0][0] ,c=palette[0],
                     marker=marqueur[1], label= "Ortho levels")
        
        for k in range(len(Model)):
            ax.plot(E_vibrationnel[0],Y_model_vibrationnel[k][0],linestyle=style[k],
                     color=color[k],alpha=0.9,label={NameList[k]+NameList2[k]},marker="+")
            
        if FillIdx!=None :
             print("\nE_vibrationnel[1]:", E_vibrationnel[1])
             print("\nY_model_vibrationnel[FillIdx[0]][1]:", Y_model_vibrationnel[FillIdx[0]][1])
             print("\nY_model_vibrationnel[FillIdx[1]][1]:", Y_model_vibrationnel[FillIdx[1]][1])
             """
             plt.fill_between(np.array(E_vibrationnel[1]),
                              np.array(Y_model_vibrationnel[ FillIdx[0] ][1]),
                              np.array(Y_model_vibrationnel[ FillIdx[1] ][1]),
                              'peachpuff', 
                              alpha=0.5)

            """
    for i in range (minV,maxV): #Par V
        if Full==False : #Tracé par v séparé
            fig, ax = plt.subplots(figsize=(16,9))
            


        for j in range (len(Lines_Vibrationnel[i])):# On trace les observations par niveau vibrationnel
            if OP_visible ==True :
                OP=int((Lines_Vibrationnel[i][j].split(',')[0].split(' ')[1]).split(')')[0])%2
            else :
                OP=0
            ax.errorbar(E_vibrationnel[i][j],Y_Obs_vibrationnel[i][j] , yerr=Error_Vib[i][j],c=palette[i],
                         markersize=15,marker=marqueur[OP],markeredgecolor='white',
                         capsize=3, elinewidth= 2 , ecolor=(0, 0, 0, 0.7),zorder=order[0])
            
            yerr = np.array([np.array([0.5 * Y_Obs_vibrationnel[i][j]]), np.array([1 * Y_Obs_vibrationnel[i][j]]) ])
            #yerr = np.array([np.array([Y_Obs_vibrationnel[i][j]])])

            ax.errorbar(E_vibrationnel[i][j],Y_Obs_vibrationnel[i][j] , yerr=yerr,c=palette[i],
                         fmt="none", capsize=1, elinewidth= 2 , ecolor=palette[i],zorder=order[0],alpha=0.2)
            
            ax.annotate(text=f"v={i+1}", xy=(E_vibrationnel[i][0],Y_Obs_vibrationnel[i][0]),
                        xytext=(E_vibrationnel[i][0]+500,Y_Obs_vibrationnel[i][0]*3),
                        color=palette[i],fontweight='bold')
            
        for k in range(len(Model)) :   
            if matchcolor == True : 
                color = palette
                idxcolor = i
            elif matchcolor ==False : 
                idxcolor = k
                
            if Full==False : 
                ax.plot(E_vibrationnel[i],Y_model_vibrationnel[k][i],linestyle=style[k],color=color[idxcolor],alpha=0.9,
                        label={NameList[k]+NameList2[k]},zorder=order[1],marker="x")
                #labelLines(ax.get_lines(), zorder=2)

            elif Full==True : #Que pour le label 
                ax.plot(E_vibrationnel[i],Y_model_vibrationnel[k][i],linestyle=style[k],
                        color=color[idxcolor],alpha=0.9,zorder=order[1],marker="x",markersize=6)    
                

                #labelLines(ax.get_lines(), zorder=2)

            
        if Reg==True and len(E_vibrationnel[i])>2:#Si on ajoute la régression
            X=np.array(E_vibrationnel[i]).reshape(-1,1)
            model = LinearRegression();model.fit(X,np.log10(Y_Obs_vibrationnel[i]))
            coef = format(model.coef_[0],'.2e');r2 =int( model.score(X, np.log10(Y_Obs_vibrationnel[i])*100))
            y_pred = 10**model.predict(X)
            residuals=(np.log10(abs(Y_Obs_vibrationnel[i] - y_pred)))

            resid_list.append(residuals)
            ax.plot(X, y_pred, color='red',linestyle=":",label=f"Vup={i},T={-int(1/float(coef))} K ")
            
        xmin,xmax,ymin,ymax=np.min(E_list),np.max(E_list),np.min(Y_Obs)/10,np.max(Y_Obs)*10
    
        if Annotation==True :             #Annotation niveaux 
            for j in range (len(E_vibrationnel[i])): 
                if Full==True : 
                    ymax=np.max(Y_Obs_vibrationnel[0])
                ax.axvline(x=E_vibrationnel[i][j], color='black', linewidth=1, ymin=0.85,ymax=0.9)
                ax.annotate(Lines_Vibrationnel[i][j],
                             xy=( E_vibrationnel[i][j] , (ymax)*1.25 ), 
                             xytext=(E_vibrationnel[i][j] + 20 ,  (ymax)*1.3 ),
                             fontsize=10,rotation=90)

        ax.grid(linestyle='dotted', alpha=0.7)
        ax.semilogy()
        ax.set_ylim(ymin,ymax*1.2)


            
        if Full==False : 
            #ax.title(f"{Name} Excitation Vup={i}")
            ax.legend(fontsize=20)
            mpl.rcParams['xtick.labelsize']=35
            mpl.rcParams['ytick.labelsize']=35
            ax.set_ylabel(r'$N_u / g_u$ $[cm^{-2}]$',fontsize=30)
            ax.set_xlabel(r"$E_u$ [K]",fontsize=30)



        elif Full ==True : 
            if Name !=None : 
                plt.title(f"{Name} Excitation Diagram all v")
            ax.legend(fontsize=20)
            ax.set_ylabel(r'$N_u / g_u$ $[cm^{-2}]$',fontsize=25)
            ax.set_xlabel(r"$E_u$ [K]",fontsize=25)

        plt.tight_layout()
        
        if Save==True : 
            if Full==True : 
                plt.savefig(Path+Name+f"_excitation_diagram.pdf",dpi=200)
            elif  Full==False : 
                plt.savefig(Path+Name+f"_excitation_diagram_v{i}.pdf",dpi=200)        
    if Reg==True :     
        residus=[]
        for el in resid_list:
            for el2 in el :
                residus.append(el2)
        print("Moyenne, Max, Longueur des résidus écart (obs-LinReg)")
        #print(residus)
        print(np.mean(residus),np.max(residus),len(residus))
        


def df_to_excitation_pyplot(df, 
                            x_col='E_upper_K', 
                            y_col='Column_Density_Normalized_by_Degeneracy', 
                            color_col='v_u', 
                            molecule_name='CO', 
                            v_levels_to_plot=None,
                            cmap="viridis",
                            ax_target=None,
                            markers=None,
                            empty=None,
                            alpha=0.8): # Ajout de ax_target pour le tracé sur un axe existant
    """
    Crée un diagramme de population rotationnel statique avec Matplotlib pur,
    incluant support LaTeX, fond blanc et axes noirs.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        x_col (str): Colonne pour l'axe X (par défaut 'E_upper_K').
        y_col (str): Colonne pour l'axe Y (par défaut 'Column_Density_Normalized_by_Degeneracy').
        color_col (str): Colonne pour la séparation par couleur (par défaut 'v_u').
        molecule_name (str): Nom de la molécule pour le titre.
        v_levels_to_plot (list, optional): Liste des niveaux vibrationnels [v] à tracer. 
                                           Si None, tous les niveaux uniques sont tracés.
        cmap (str, optional): Nom de la colormap à utiliser (par défaut 'viridis').
        ax_target (matplotlib.axes.Axes, optional): Axe sur lequel tracer. Si None, un nouvel axe est créé.
    """
    
    # Assurez-vous que les colonnes nécessaires sont présentes
    if not all(col in df.columns for col in [x_col, y_col, color_col]):
        missing_cols = [col for col in [x_col, y_col, color_col] if col not in df.columns]
        raise ValueError(f"Le DataFrame ne contient pas les colonnes nécessaires : {missing_cols}")

    # --- Filtrage des données si v_levels_to_plot est spécifié ---
    if v_levels_to_plot is not None:
        df_filtered = df[df[color_col].isin(v_levels_to_plot)].copy()
        if df_filtered.empty:
            print(f"Avertissement : Aucun point trouvé pour les niveaux v_u spécifiés : {v_levels_to_plot}")
            return ax_target
        
        df_plot = df_filtered
        unique_v_u = sorted(v_levels_to_plot)
    else:
        df_plot = df.copy()
        unique_v_u = sorted(df_plot[color_col].unique())

    # --- Gestion de l'axe de tracé et Configuration du style ---
    
    if ax_target is None:
        # Création d'une nouvelle figure et axe si ax_target n'est pas fourni
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Activation de l'interprétation LaTeX et configuration du style si c'est une nouvelle figure
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        
        # Application des ajustements esthétiques de base (fond blanc, axes noirs, grille)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        
        ax.tick_params(axis='x', colors='black', labelsize=20)
        ax.tick_params(axis='y', colors='black', labelsize=20)
        ax.set_yscale('log')
        
        # Définition des labels et du titre pour la nouvelle figure
        ax.set_xlabel(r'\textbf{Upper level energy $E_u$ [K]}', fontsize=16)
        ax.set_ylabel(r'\textbf{Normalised column density $\frac{N_u}{g_u}$ [cm$^{-2}$]}', fontsize=16)
        #ax.set_title(r'\textbf{Diagramme de Population Rotationnel pour ' + molecule_name + r'}', fontsize=20)
        
    else:
        # Utilisation de l'axe cible existant
        ax = ax_target
    
    # --- Définition de la palette de couleurs ---
    try:
        colormap = cm.get_cmap(cmap)
        # Générer les couleurs pour les niveaux v_u uniques à partir de la colormap spécifiée
        colors = [colormap(i) for i in np.linspace(0, 1, len(unique_v_u))]
    except ValueError:
        print(f"Avertissement : La colormap '{cmap}' n'a pas été trouvée. Utilisation de 'viridis' par défaut.")
        colormap = cm.get_cmap('viridis')
        colors = [colormap(i) for i in np.linspace(0, 1, len(unique_v_u))]

    
    
    # --- Tracé des données ---
    
    # Itérer sur chaque niveau v_u à tracer
    for i, v_u in enumerate(unique_v_u):
        subset = df_plot[df_plot[color_col] == v_u]
        
        # S'assurer qu'il y a des données pour ce niveau avant de tracer
        if subset.empty:
            continue
        
        
        marker_list = ['o', 's', '^', 'D', 'v', 'P', 'X', 'H', '*', '+']
        color = colors[i % len(colors)]
        if markers == None : 
            marker = marker_list[i % len(marker_list)]
        elif markers != None : 
            marker = markers
            
        if empty == None : 
            ax.scatter(
                subset[x_col],
                subset[y_col],
                # Modifier le label pour inclure le nom de la molécule afin de différencier les tracés sur un même axe
                label=rf'{molecule_name} : $v_u={v_u}$', 
                color=color,
                marker=marker,
                s=80, # Taille des points
                alpha=alpha,
                edgecolors=None, 
                linewidths=0)
            
        elif empty != None : 
            ax.scatter(
            subset[x_col],
            subset[y_col],
            label=rf'{molecule_name} : $v_u={v_u}$', 
            marker=marker, 
            facecolors='none',    
            edgecolors=color,      
            linewidths=1.5,        
            s=80,
            alpha=alpha)
        
        

    # --- Légende et disposition ---
    # Nous appelons legend ici pour ajouter les points tracés dans ce run
    ax.legend(title='Vibrational levels', loc='best')

    # Ajustement de la disposition seulement si c'est la première fois ou si vous le désirez
    if ax_target is None:
        plt.tight_layout()

    # Si c'est le dernier appel, nous affichons le graphique.
    # Sinon, nous retournons l'axe pour permettre d'autres tracés.
    return ax








def plot_dual_excitation_and_error(
    df1, df2,
    x_col='E_upper_K',
    y_col='Column_Density_Normalized_by_Degeneracy',
    color_col='v_u',
    molecule_names=['Mol1', 'Mol2'], # List for names of df1 and df2
    v_levels_to_plot=[None, None], # List for v_levels for df1 and df2
    cmaps=["viridis", "plasma"], # List for colormaps for df1 and df2
    markers=[None, None], # List for markers for df1 and df2
    empty_styles=[None, None], # List for empty styles for df1 and df2
    alphas=[0.8, 0.8], # List for alpha transparency for df1 and df2
    error_type='relative_percent', # 'relative_percent', 'log_difference', 'ratio'
    error_y_lim_top=None # Optional: set upper limit for error y-axis
):
    """
    Crée un double subplot pour deux diagrammes de population rotationnels et leur erreur.

    Le panneau supérieur (ratio 3:1) affiche les diagrammes d'excitation pour df1 et df2.
    Le panneau inférieur affiche l'erreur (relative en %, différence log, ou ratio)
    entre les colonnes de densité normalisée des deux DataFrames en fonction de l'énergie du niveau supérieur.

    Args:
        df1 (pd.DataFrame): Le premier DataFrame de données.
        df2 (pd.DataFrame): Le second DataFrame de données.
        x_col (str): Nom de la colonne pour l'axe X (énergie du niveau supérieur).
        y_col (str): Nom de la colonne pour l'axe Y (densité de colonne normalisée).
        color_col (str): Nom de la colonne pour la séparation par couleur (niveaux vibrationnels).
        molecule_names (list): Liste de deux chaînes de caractères [nom_df1, nom_df2] pour les légendes.
        v_levels_to_plot (list): Liste de deux listes (ou None) [v_levels_df1, v_levels_df2] pour filtrer les niveaux v.
        cmaps (list): Liste de deux noms de colormap [cmap_df1, cmap_df2].
        markers (list): Liste de deux marqueurs (ou None) [marker_df1, marker_df2].
                        Si None, utilise un cycle de marqueurs.
        empty_styles (list): Liste de deux booléens (ou None) [empty_df1, empty_df2].
                             Si True, les marqueurs sont vides.
        alphas (list): Liste de deux floats [alpha_df1, alpha_df2] pour la transparence.
        error_type (str): Type d'erreur à tracer. 'relative_percent' (défaut), 'log_difference', 'ratio'.
        error_y_lim_top (float, optional): Limite supérieure de l'axe Y pour le graphique d'erreur.

    Returns:
        tuple: (fig, (ax_main, ax_error), df_merged) Les objets Figure et Axes pour les sous-graphes et le DataFrame fusionné.
    """

    # --- Vérification des DataFrames et des colonnes ---
    required_cols = [x_col, y_col, color_col]
    for i, df in enumerate([df1, df2]):
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"DataFrame {i+1} manque les colonnes nécessaires : {missing_cols}")

    # --- Configuration globale de Matplotlib pour LaTeX et style ---
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    # --- Création de la figure et des sous-graphes ---
    fig, (ax_main, ax_error) = plt.subplots(2, 1, figsize=(21, 10), # MODIFIED: Figure size
                                            gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # --- Configuration des styles d'axes ---
    # Axe principal (diagramme d'excitation)
    ax_main.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    for spine in ax_main.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    ax_main.tick_params(axis='x', colors='black', labelsize=25, labelbottom=False) # MODIFIED: Label size
    ax_main.tick_params(axis='y', colors='black', labelsize=25) # MODIFIED: Label size
    ax_main.set_yscale('log')
    ax_main.set_ylabel(r'\textbf{Normalised Column Density $\frac{N_u}{g_u}$ [cm$^{-2}$]}', fontsize=20) # MODIFIED: Font size
    # ax_main.set_title(r'\textbf{Diagrammes de Population Rotationnels}', fontsize=25) # Removed per your previous code's comment

    # Axe d'erreur
    ax_error.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    for spine in ax_error.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    ax_error.tick_params(axis='x', colors='black', labelsize=25) # MODIFIED: Label size
    ax_error.tick_params(axis='y', colors='black', labelsize=25) # MODIFIED: Label size
    ax_error.set_xlabel(r'\textbf{Upper level Energy $E_u$ [K]}', fontsize=20) # MODIFIED: Font size

    # Définition du label Y de l'erreur en fonction du type
    if error_type == 'relative_percent':
        ax_error.set_ylabel(r'\textbf{Relative Error [\%]}', fontsize=20) # MODIFIED: Font size
        ax_error.set_ylim(bottom=0) # Relative error is always >= 0
    elif error_type == 'log_difference':
        ax_error.set_ylabel(r'$\mathbf{\log_{10}(N_{u,df1}/g_{u,df1}) - \log_{10}(N_{u,df2}/g_{u,df2})}$', fontsize=20) # MODIFIED: Font size
        # This can be positive or negative
    elif error_type == 'ratio':
        ax_error.set_ylabel(r'$\mathbf{(N_{u,df1}/g_{u,df1}) / (N_{u,df2}/g_{u,df2})}$', fontsize=20) # MODIFIED: Font size
        ax_error.set_yscale('log') # Ratio usually better on log scale
        ax_error.set_ylim(bottom=1e-3) # Avoid log(0)
    else:
        raise ValueError(f"Type d'erreur '{error_type}' non supporté. Choisissez 'relative_percent', 'log_difference' ou 'ratio'.")

    if error_y_lim_top is not None:
        ax_error.set_ylim(top=error_y_lim_top)


    # --- Boucle pour tracer chaque DataFrame (df1 et df2) ---
    dataframes = [df1, df2]
    marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', 'H', '*', '+']

    for idx, current_df in enumerate(dataframes):
        name = molecule_names[idx]
        v_levels = v_levels_to_plot[idx]
        cmap_name = cmaps[idx]
        marker_style = markers[idx]
        empty_style = empty_styles[idx]
        alpha = alphas[idx]

        # Filtrage des données si v_levels_to_plot est spécifié pour ce DF
        if v_levels is not None:
            df_plot = current_df[current_df[color_col].isin(v_levels)].copy()
            if df_plot.empty:
                print(f"Avertissement : Aucun point trouvé dans {name} pour les niveaux '{color_col}' spécifiés : {v_levels}")
                continue
            unique_v_u = sorted(v_levels)
        else:
            df_plot = current_df.copy()
            unique_v_u = sorted(df_plot[color_col].unique())

        # Définition de la palette de couleurs
        try:
            colormap = cm.get_cmap(cmap_name)
            colors = [colormap(i) for i in np.linspace(0, 1, len(unique_v_u))]
        except ValueError:
            print(f"Avertissement : La colormap '{cmap_name}' n'a pas été trouvée. Utilisation de 'viridis' par défaut pour {name}.")
            colormap = cm.get_cmap('viridis')
            colors = [colormap(i) for i in np.linspace(0, 1, len(unique_v_u))]

        # Tracé des points pour le diagramme d'excitation
        for i, v_u in enumerate(unique_v_u):
            subset = df_plot[df_plot[color_col] == v_u]
            if subset.empty:
                continue

            current_marker = marker_style if marker_style is not None else marker_cycle[i % len(marker_cycle)]
            current_color = colors[i % len(colors)]

            if empty_style is None: # Marqueurs pleins par défaut
                ax_main.scatter(
                    subset[x_col],
                    subset[y_col],
                    label=rf'{name} : $v_u={v_u}$',
                    color=current_color,
                    marker=current_marker,
                    s=80,
                    alpha=alpha,
                    edgecolors=None,
                    linewidths=0
                )
            else: # Marqueurs vides
                ax_main.scatter(
                    subset[x_col],
                    subset[y_col],
                    label=rf'{name} : $v_u={v_u}$',
                    marker=current_marker,
                    facecolors='none',
                    edgecolors=current_color,
                    linewidths=1.5,
                    s=80,
                    alpha=alpha
                )

    # --- Calcul et tracé de l'erreur ---
    # Fusionner les DataFrames sur les colonnes clés (E_upper_K et v_u)
    # Utiliser un outer merge pour s'assurer que tous les points sont considérés pour le calcul de l'erreur
    # et gérer les cas où un point n'est pas dans les deux DFs
    df_merged = pd.merge(
        df1[[x_col, y_col, color_col]],
        df2[[x_col, y_col, color_col]],
        on=[x_col, color_col],
        suffixes=('_df1', '_df2'),
        how='outer' # Use outer merge to see all points, then filter for error calc
    )

    # Filtrer pour ne conserver que les points présents dans les deux DFs pour l'erreur
    df_merged_for_error = df_merged.dropna(subset=[y_col + '_df1', y_col + '_df2']).copy()

    if not df_merged_for_error.empty:
        # Gérer la division par zéro ou log(0)
        # Remplacer les 0 dans le dénominateur ou l'argument de log par NaN ou une petite valeur
        df_merged_for_error['temp_df1_y_col'] = df_merged_for_error[y_col + '_df1'].replace(0, np.nan)
        df_merged_for_error['temp_df2_y_col'] = df_merged_for_error[y_col + '_df2'].replace(0, np.nan)

        if error_type == 'relative_percent':
            df_merged_for_error['Error_Value'] = np.abs(
                (df_merged_for_error['temp_df1_y_col'] - df_merged_for_error['temp_df2_y_col']) / df_merged_for_error['temp_df1_y_col']
            ) * 100
            # Remove NaNs created by division by zero in temp_df1_y_col
            df_merged_error_plot = df_merged_for_error.dropna(subset=['Error_Value']).copy()

        elif error_type == 'log_difference':
            # Ensure values are positive before taking log
            df_merged_for_error['Error_Value'] = (
                np.log10(df_merged_for_error['temp_df1_y_col']) - np.log10(df_merged_for_error['temp_df2_y_col'])
            )
            df_merged_error_plot = df_merged_for_error.dropna(subset=['Error_Value']).copy()

        elif error_type == 'ratio':
            # Ensure values are positive before taking log for ratio denominator if plotting on log scale
            df_merged_for_error['Error_Value'] = (
                df_merged_for_error['temp_df1_y_col'] / df_merged_for_error['temp_df2_y_col']
            )
            df_merged_error_plot = df_merged_for_error.dropna(subset=['Error_Value']).copy()

        # Filtrer l'erreur relative par les niveaux v_u de df1 (si spécifié)
        if v_levels_to_plot[0] is not None:
            df_merged_error_plot = df_merged_error_plot[
                df_merged_error_plot[color_col].isin(v_levels_to_plot[0])
            ].copy()
            if df_merged_error_plot.empty:
                print(f"Avertissement : Aucun point d'erreur trouvé pour les niveaux '{color_col}' spécifiés dans df1 : {v_levels_to_plot[0]}. L'erreur ne sera pas tracée pour ces niveaux.")
                return fig, (ax_main, ax_error), df_merged # Return before plotting error

        # Récupérer les niveaux v_u uniques *après* le filtrage final pour le tracé d'erreur
        unique_v_u_error = sorted(df_merged_error_plot[color_col].unique())
        error_colormap = cm.get_cmap(cmaps[0]) # Use df1's colormap for error consistency
        error_colors = [error_colormap(i) for i in np.linspace(0, 1, len(unique_v_u_error))]


        for i, v_u in enumerate(unique_v_u_error):
            subset_error = df_merged_error_plot[df_merged_error_plot[color_col] == v_u]
            if subset_error.empty:
                continue

            current_marker = markers[0] if markers[0] is not None else marker_cycle[i % len(marker_cycle)]
            current_color = error_colors[i % len(error_colors)]

            ax_error.scatter(
                subset_error[x_col],
                subset_error['Error_Value'], # Utilisez la colonne générique 'Error_Value'
                label=rf'Erreur $v_u={v_u}$',
                color=current_color,
                marker=current_marker,
                s=60,
                alpha=0.8
            )
        ax_error.legend(title='Erreur par Niveau', loc='best', fontsize=15) # MODIFIED: Font size

    else:
        print("Avertissement : Aucun point correspondant trouvé entre df1 et df2 pour calculer l'erreur.")

    # --- Légendes et ajustements finaux ---
    ax_main.legend(title='Données par Molécule et Niveau', loc='best', fontsize=15, ncol=2) # MODIFIED: Font size
    plt.tight_layout()
    plt.subplots_adjust(hspace=0) # Réduit l'espace vertical entre les subplots
    plt.show()

    return fig, (ax_main, ax_error), df_merged








def df_to_excitation_interactive(df, molecule_name='CO', fig_target=None, empty=False, v_u_levels_to_plot=None):
    """
    Creates an interactive population diagram using Plotly Express, 
    allowing chaining plots on the same figure and customizing marker appearance.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        molecule_name (str): Name of the molecule for labeling.
        fig_target (plotly.graph_objects.Figure, optional): An existing figure to add traces to. 
                                                            If None, a new figure is created.
        empty (bool, optional): If True, markers will be empty (hollow) with a contour. 
                                If False, markers will be filled.
        v_u_levels_to_plot (list, optional): A list of v_u levels (integers) to include in the plot.
                                            If None, all v_u levels will be plotted.
    """
   
    # Set default Plotly renderer for compatibility (e.g., in Spyder)
    pio.renderers.default = "browser"

    # Ensure required columns are present
    required_cols = ['E_upper_K', 'Column_Density_Normalized_by_Degeneracy', 'v_u', 'J_u']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    # Filter by v_u_levels_to_plot if specified
    if v_u_levels_to_plot is not None:
        df = df[df['v_u'].isin(v_u_levels_to_plot)].copy()
        if df.empty:
            print(f"Warning: No data found for specified v_u levels: {v_u_levels_to_plot}. Returning an empty figure if no fig_target was provided.")
            if fig_target is None:
                return go.Figure() # Return an empty figure if no data and no target
            else:
                return fig_target # Return the existing figure without adding new traces
    
    # Ensure v_u is treated as a string for discrete coloring
    df['v_u_str'] = df['v_u'].astype(str)

    # --- Plotly Express scatter plot setup ---
   
    # Plotly Express is generally used for the initial setup. 
    # If we are adding to an existing figure (fig_target), we will handle it differently.
   
    if fig_target is None:
        # Create a new Plotly figure using Plotly Express
        fig = px.scatter(
            df,
            x='E_upper_K',
            y='Column_Density_Normalized_by_Degeneracy',
            color='v_u_str', # Use the string version for discrete colors
            hover_data={
                'v_u': True,  # Use original numeric v_u for hover
                'J_u': True,
                'v_l' : True,  # Use original numeric v_u for hover
                'J_l': True,
                'E_upper_K': ':.2f',
                'Column_Density_Normalized_by_Degeneracy': ':.2e'
            },
            log_y=True,
            title=f'Rotational Population Diagram for {molecule_name}',
            labels={
                'E_upper_K': r'Eu[K]',
                'Column_Density_Normalized_by_Degeneracy': r'Nu/gu [cm^{-2}]',
                'v_u_str': r'vu',
                'J_u': r'Ju',
                'v_l_str': r'vl',
                'J_l': r'Jl'
            }
        )
       
        # Apply initial layout settings (white background, black axes, labels)
        fig.update_layout(
            template='plotly_white',
            xaxis=dict(showline=True, linecolor='black', mirror=True, 
                       title_text=r'Énergie du Niveau Supérieur $E_u$ [K]', 
                       title_font={"size": 70}),
            yaxis=dict(showline=True, linecolor='black', mirror=True, 
                       title_text=r'Densité de Colonne Normalisée $\frac{N_u}{g_u}$ [cm$^{-2}$]', 
                       title_font={"size": 70})
        )

        # --- Marker Customization for new figure ---
        if empty:
            # Set fill color to transparent and line color to the trace's assigned color
            for trace in fig.data:
                if isinstance(trace, go.Scatter) and trace.mode == 'markers':
                    trace.marker = dict(
                        size=10, 
                        opacity=1.0, 
                        color='rgba(0,0,0,0)', # Transparent fill color
                        line=dict(
                            width=2,
                            #color=trace.marker.color # Use the trace's assigned color for the line
                        )
                    )
        else:
            fig.update_traces(
                marker=dict(size=10, opacity=0.8),
                selector=dict(mode='markers')
            )

    else:
        # Use the existing figure and add a new scatter trace to it
        fig = fig_target
        
        # Keep track of the indices of the newly added traces
        new_trace_indices = []
        current_num_traces = len(fig.data)
        
        # Plotly Express assigns colors automatically. We can get these colors if we need them.
        # This is a bit of a workaround to get the colors Plotly Express would assign for `v_u_str`
        # if it were creating a new figure. For `fig_target` usage, Plotly.js will pick colors
        # from its default palette or extend the existing one.
        # It's simpler to directly use a color map or let Plotly assign it and then retrieve.
        
        # Store colors from the current figure's layout if possible, or use a default palette
        # This part is complex because Plotly Express's internal color mapping is not directly exposed
        # when adding traces to an existing figure. The colors are picked sequentially from a palette.
        # For simplicity, we'll let Plotly assign a color, then retrieve it for the line.
       
        # Add new traces for the current DataFrame
        for i, (v_level, group) in enumerate(df.groupby('v_u_str')):
            # Plotly will assign a color automatically. We'll retrieve it after adding the trace.
            fig.add_trace(go.Scatter(
                x=group['E_upper_K'],
                y=group['Column_Density_Normalized_by_Degeneracy'],
                mode='markers',
                name=f'{molecule_name} - v={v_level}', # Name for the legend entry
                customdata=group[['v_u', 'J_u', 'v_l', 'J_l','E_upper_K', 'Column_Density_Normalized_by_Degeneracy']],
                hovertemplate="<br>".join([
                    "<b>Molécule:</b> " + molecule_name,
                    "<b>v_u:</b> %{customdata[0]}",
                    "<b>J_u:</b> %{customdata[1]}",
                    "<b>v_l:</b> %{customdata[2]}",
                    "<b>J_l:</b> %{customdata[3]}",
                    "<b>Eu[K]:</b> %{customdata[4]:.2f}",
                    "<b>Nu/gu:</b> %{customdata[5]:.2e}",
                ])
            ))
            new_trace_indices.append(current_num_traces + i)

        # --- Marker Customization (Hollow vs. Filled) for NEW traces only ---
        if empty:
            for trace_idx in new_trace_indices:
                # Get the color that Plotly automatically assigned to this new trace
                assigned_color = fig.data[trace_idx].marker.color 
                if assigned_color is None: # Fallback if no color was automatically assigned (unlikely for scatter)
                    # This might happen if using a custom template or specific situations
                    # For safety, you could define a default list of colors here
                    assigned_color = 'blue' # Or some other default color

                fig.data[trace_idx].marker = dict(
                    size=10, 
                    opacity=1.0, 
                    color='rgba(0,0,0,0)', # Transparent fill color
                    line=dict(
                        width=2,
                        color=assigned_color # Use the assigned color for the line
                    )
                )
        else:
            # If 'empty' is False, use default filled markers for NEW traces
            for trace_idx in new_trace_indices:
                fig.data[trace_idx].marker = dict(size=10, opacity=0.8)


    # --- Display and Return ---
   
    # If a new figure was created (fig_target is None), display it now. 
    # If we are tracing on an existing figure, we return it for further manipulation or eventual display.
    if fig_target is None:
        fig.show()
   
    # Return the figure object, allowing the user to add more traces to it.
    return fig
        


def Obs_excitation (Observation_list,Name, Level=None, Annotation=False, Full=False,FillIdx =None) :
    """Crée un diagramme d'excitation en séparant par v, et en mettant en évidence les différences O et P"""
    marqueur=['^','o']#Para = pair = 0 et ortho = impair = 1
    fill = ['full', 'none', 'left', 'right', 'bottom', 'top']
    Y_tot = []
    Raie_tot = []
    Line_Tot = []
    Err =[]
    V_final = []
    for a in range (len(Observation_list)) : 
        Raies =  Observation_list[a].get_observable_names()
        I_Obs =  Observation_list[a].get_observable_vector()
        E_list = Observation_list[a].get_error_vector()

                
        #Comparaison avec la data base du code
        main_root="/Users/aurelien/Desktop/Stage_M2/"
        rootbase=main_root+"PDR/1.7/"
        namebase="all_h2_lines_info"
        doc_base=open(rootbase+namebase+".dat","r")
        file_base=doc_base.read();  ligne_base=file_base.split("\n")
             
        Data_base=[]
        for i in range (len(Raies)) :
            temp=i
            nom_raie=Raies[i]
            while nom_raie!=ligne_base[temp].split(',')[0] and temp<len(ligne_base):
                temp+=1
            if nom_raie==ligne_base[temp].split(',')[0] : 
                Data_base.append(ligne_base[temp])
            else : 
                print('Error : Not in database, please check if nomenclature is correct.')
                
        #Récupération des paramètres nécessaires     
        Wave_list,Eul_list,A_list,Eu_list,gu_list,ju_list=[],[],[],[],[],[]    
        for i in range (len(Data_base)):
            raie_sel=Data_base[i].split(',')
            Wave_list.append(float(raie_sel[11].split('\t')[-1]))
            A_list.append(float(raie_sel[10].split('\t')[-1]))
            Eul_list.append(float(raie_sel[9].split('\t')[-1]))
            Eu_list.append(float(raie_sel[7].split('\t')[-1]))
            gu_list.append(float(raie_sel[5].split('\t')[-1]))
            ju_list.append(float(raie_sel[2].split('\t')[-1]))
        
        #Calcul de la colonne densité pour les observations dans Nu_Obs
        Nu_Obs=[4*np.pi * (Wave_list[i] * 1e-4)   *I_Obs[i]/(h*clum*A_list[i]) for i in range (len(A_list))] #Attnetion lambda data en micron -> cm 
        Nu_Err=[4*np.pi * (Wave_list[i] * 1e-4)  *(0.3*I_Obs[i])/(h*clum*A_list[i]) for i in range (len(A_list))]
    
        Y_Obs=[(Nu_Obs[i]/gu_list[i]) for i in range (len(Nu_Obs))]
        Y_Err=[(Nu_Err[i]/gu_list[i]) for i in range (len(Nu_Obs))]
    
    
        Lines_Vibrationnel=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
        Y_Obs_vibrationnel=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
        Error_Vib=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
        E_vibrationnel=[[] for i in range(int(Raies[-1].split('-')[0])+1)]
        
        for i in range (len(Raies)): 
            Vh=int(Raies[i].split("-")[0])
            Lines_Vibrationnel[Vh].append(Raies[i])
            Y_Obs_vibrationnel[Vh].append(Y_Obs[i])
            Error_Vib[Vh].append(Y_Err[i])
            E_vibrationnel[Vh].append(Eu_list[i])
    
    
        if Level==None : 
            maxV=len(Lines_Vibrationnel)
        elif Level!= None : 
            maxV=Level
        Line_Tot.append(Lines_Vibrationnel)
        Y_tot.append(Y_Obs_vibrationnel)
        Raie_tot.append(E_vibrationnel)
        Err.append(Error_Vib)
        V_final.append(maxV)
        
        
        
        #-------------- Graphique----------------------------------
    if Full==True : #Tracé de tous les niveaux sur le meme diagramme 
        plt.figure(figsize=(21,9))
        for a in range (len(Y_tot)) : 
            plt.scatter(Raie_tot[a][1][0],Y_tot[a][1][0] ,c=palette[1],
                         marker=marqueur[0], label= "Para levels")
            
            plt.scatter(Raie_tot[1][0],Y_tot[1][0] ,c=palette[1],
                         marker=marqueur[1], label= "Ortho levels")
    for a in range (len(Y_tot)):
        for i in range (1,V_final[a]): #Par V
            if Full==False : #Tracé par v séparé
                plt.figure(figsize=(16,9)) # for v separated
            

            for j in range (len(Raie_tot[a][i])):# On trace les observations par niveau vibrationnel
                OP=int((Line_Tot[a][i][j].split(',')[0].split(' ')[1]).split(')')[0])%2
                line, caps, bars = plt.errorbar(Raie_tot[a][i][j],Y_tot[a][i][j] , yerr=Err[a][i][j],c=palette[i],
                             markersize=15,marker=marqueur[OP],markeredgecolor='blue',
                             capsize=3, elinewidth= 2 , ecolor=palette[a],zorder=1,fillstyle=fill[a])
                
                # Set the alpha for the error bars
                for bar in bars:
                    bar.set_alpha(0.3)
                
                # Set the alpha for the error bar caps (optional, but good practice for consistency)
                for cap in caps:
                    cap.set_alpha(0.3)
                
                yerr = np.array([np.array([0.5 * Y_tot[a][i][j]]), np.array([1 * Y_tot[a][i][j]]) ])
                #yerr = np.array([np.array([Y_Obs_vibrationnel[i][j]])])
    
                plt.errorbar(Raie_tot[a][i][j],Y_tot[a][i][j] , yerr=yerr,c=palette[i],
                             fmt="none", capsize=1, elinewidth= 2 , ecolor=palette[i],zorder=1,alpha=0.2)
            
 
            
        xmin,xmax,ymin,ymax=np.min(E_list),np.max(E_list),np.min(Y_Obs),np.max(Y_Obs)
    
        if Annotation==True :             #Annotation niveaux 
            for j in range (len(E_vibrationnel[i])): 
                if Full==True : 
                    ymax=np.max(Y_Obs_vibrationnel[1])
                plt.axvline(x=Raie_tot[a][i][j], color='black', linewidth=1, ymin=0.85,ymax=0.9)
                plt.annotate(Line_Tot[a][i][j],
                             xy=( Raie_tot[a][i][j] , (ymax)*1.25 ), 
                             xytext=(Raie_tot[a][i][j] + 20 ,  (ymax)*1.3 ),
                             fontsize=10,rotation=90)

        plt.xlabel(r"$E_u$ [K]",fontsize=25)
        #plt.ylabel(r'$log_{10}(N_u/g_u)$',fontsize=20) 
        plt.ylabel(r'$N_u / g_u$ $[cm^{-2}]$',fontsize=25)
        plt.grid(linestyle='dotted', alpha=0.7)
        plt.semilogy()
        plt.ylim(ymin/2,ymax*2.5)
        plt.legend(fontsize=20)
        #plt.text(5500 , ymax/2 , "\^: Para \n \o : Ortho", fontsize = 15)
        plt.tight_layout()

        
        
        
        
        
# =============================================================================
#                       Contour Plot  
# =============================================================================    

#Contourplot X2 
#Contour Transition H H2 




# =============================================================================
#                       Stellar/ISRF spectras   
# =============================================================================    



# =============================================================================
#                       Extinction curve 
# =============================================================================    



def fit_quality_contour_plots_singlepair2(grid_approximator,observation,parameter_name_couple,ref_model,with_scaling_factor=True,cost_function="reduced_chi2",scaling_factor_bounds = [0.1,10.],n_contours=10,uncertainty_contour=False,use_true_Dchi2_threshold=False,cmap="viridis",chi2_bounds=[None,None],cur_axis=None,couleur="tab:orange",RefPoint=None,RefPoint2=None,Title=None,LabelName=None, GoLim=None):
   
   # prepare chi2 thresholds
   if use_true_Dchi2_threshold :
      thresholds = [1., 2.3, 3.53, 4.72, 5.89, 7.04] # delta chi2 defining the 68.27% uncertainty domain depending on number of degrees of freedom, in a normal approximation (from Numerical Recipes, 15.6.5) - Probably wrong, check again
   else :
      thresholds = [1., 1., 1., 1., 1., 1.] # just always show the contour with Delta chi2 = 1
   
   # plot parameters
   font_size = 35
   n_points = 50
   
   # translate string for cost function if given
   if cost_function == "chi2":
      my_cost_function = pyismtools.utils.chi2
   elif cost_function == "reduced_chi2":
      my_cost_function = pyismtools.utils.reduced_chi2
   elif cost_function == "chi2_ISMFIT":
      my_cost_function = pyismtools.utils.chi2_ISMFIT
   
   # best model
   best_chi2 = my_cost_function(ref_model.get_observable_vector(),observation.get_observable_vector(),observation.get_error_vector(),observation.get_error_type_vector(),ref_model.get_N_parameters())
   ref_parameters = np.array( ([ref_model.get_scaling_factor()] if with_scaling_factor else []) + list(ref_model.get_parameter_vector()) )
   
   # prepare grids of parameter values for the plots
   nb_params = grid_approximator.get_N_parameters() + with_scaling_factor
   param_grids = []
   for param_name in parameter_name_couple :
      if param_name == "Scaling factor" :
         param_grids.append(np.logspace(np.log10(scaling_factor_bounds[0]),np.log10(scaling_factor_bounds[1]),n_points))
      else :
         i_param = grid_approximator.get_parameter_names().index(param_name)
         param_grids.append(np.logspace(np.log10(grid_approximator.get_parameter_mins()[i_param]),np.log10(grid_approximator.get_parameter_maxs()[i_param]),n_points))
   
   param_name_1, param_name_2 = parameter_name_couple
   if LabelName==None :
       Par_name1,Par_name2 = parameter_name_couple
   elif LabelName!= None :
       Par_name1,Par_name2 = LabelName
       
       
   if (cur_axis is None):  # if no axes provided, create a new figure and get the axes, otherwise we will work on the provided axes
      fig = plt.figure(figsize=(12,9))
      cur_axis = plt.gca()  
   
   # if the two parameters are the same :  1D cuts of the cost function vs one parameter
   if param_name_1==param_name_2 :
      i_param = grid_approximator.get_parameter_names().index(param_name_1)+1 if param_name_1 != "Scaling factor" else 0
      temp_param_grid = np.tile(ref_parameters,(n_points,1))
      temp_param_grid[:,i_param] = param_grids[0]
      cost_grid = my_cost_function(temp_param_grid[:,0][:,np.newaxis]*grid_approximator.vectorized_call(temp_param_grid[:,1:]),observation.get_observable_vector(),observation.get_error_vector(),observation.get_error_type_vector(),grid_approximator.get_N_parameters())
      divider = make_axes_locatable(cur_axis)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      cax.axis('off')
      cur_axis.plot(param_grids[0],cost_grid)
      cur_axis.set_xlabel(param_name_1,fontsize=font_size)
      cur_axis.set_ylabel(cost_function,fontsize=font_size)
      cur_axis.set_xscale('log')
      cur_axis.set_yscale('log')
      cur_axis.axvline(ref_parameters[i_param],linestyle="--",color='r')
   
   # otherwise : 2D contour plots for the cost function vs two parameters
   else :
      # compute the cost function on the 2D grid
      i_param1 = grid_approximator.get_parameter_names().index(param_name_1)+1 if param_name_1 != "Scaling factor" else 0
      i_param2 = grid_approximator.get_parameter_names().index(param_name_2)+1 if param_name_2 != "Scaling factor" else 0
      temp_param_grid = np.tile(ref_parameters,(n_points,n_points,1))
      temp_param_grid[:,:,i_param1] = param_grids[0][:,np.newaxis]
      temp_param_grid[:,:,i_param2] = param_grids[1][np.newaxis,:]
      flattened_param_grid = np.reshape(temp_param_grid,(-1,temp_param_grid.shape[-1])) # turn a (k,l,m) array into a (k*l,m) array (flattening the first two axis only, so that we have a list of parameter vectors)
      flattened_cost_map = my_cost_function(flattened_param_grid[:,0][:,np.newaxis]*grid_approximator.vectorized_call(flattened_param_grid[:,1:]),observation.get_observable_vector(),observation.get_error_vector(),observation.get_error_type_vector(),grid_approximator.get_N_parameters())
      cost_map = np.reshape(flattened_cost_map,(n_points,n_points)) # reshape into a map
      # make the plot
      divider = make_axes_locatable(cur_axis)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      cur_axis.set_xscale('log')
      cur_axis.set_yscale('log')
      vmin = np.nanmin(cost_map[np.nonzero(cost_map)])
      vmax = np.nanmax(cost_map[np.nonzero(cost_map)])
      if(chi2_bounds[0] is not None) :
         vmin = chi2_bounds[0]
      if(chi2_bounds[1] is not None) :
         vmax = chi2_bounds[1]
      if ( floor(np.log10(vmax)) - floor(np.log10(vmin)) - 1 < 1 ):
         vmin = 10.**floor(np.log10(vmin))
         vmax = 10.**floor(np.log10(vmax)+1)
      elif ( floor(np.log10(vmax)) - floor(np.log10(vmin)) - 1 < 2 ):
         vmin = 10.**floor(np.log10(vmin))
      cs = cur_axis.contourf(param_grids[1],param_grids[0],cost_map,norm = LogNorm(),locator=ticker.LogLocator(),vmin=vmin,vmax=vmax,cmap=cmap,levels=np.logspace(np.log10(vmin),np.log10(vmax),200),zorder=0)
      cur_axis.contour(param_grids[1],param_grids[0],cost_map,norm = LogNorm(),locator=ticker.LogLocator(),vmin=vmin,vmax=vmax,cmap=cmap,levels=np.logspace(np.log10(vmin),np.log10(vmax),200),zorder=0)
      cs2 = cur_axis.contour(param_grids[1],param_grids[0],cost_map,norm = LogNorm(),locator=ticker.LogLocator(),vmin=vmin,vmax=vmax,colors="w",levels=np.logspace(np.log10(vmin),np.log10(vmax),n_contours),zorder=1,linestyles=":")
      labels = cur_axis.clabel(cs2, cs2.levels, fontsize=20, inline = True, inline_spacing = 2)
      for l in labels:
         l.set_rotation(0)
      cur_axis.scatter(ref_parameters[i_param2],ref_parameters[i_param1],c=couleur,s=1000,zorder=3,edgecolors="white")
      
      
      
      if RefPoint != None : 
          cur_axis.scatter(RefPoint[0],RefPoint[1],c=RefPoint[2],s=1000,zorder=3,edgecolors="white")
          
      if RefPoint2 != None : 
          cur_axis.scatter(RefPoint2[0],RefPoint2[1],c=RefPoint2[2],s=1000,zorder=3,edgecolors="white")
          
      if GoLim!=None :
          indexGo = parameter_name_couple.index("Go")
          indexPth = parameter_name_couple.index("Pgas")
          GoGrid=[np.min(param_grids[indexPth]), np.max(param_grids[indexPth])]
          if indexGo==1 : 
              cur_axis.plot([GoLim,GoLim],GoGrid , marker="",linestyle="--",color="red")
          elif indexGo==0 : 
              cur_axis.plot(GoGrid , [GoLim,GoLim], marker="",linestyle="--",color="red")          
          
          
      if Title != None:
          plt.title(Title)
      cbar = plt.colorbar(cs,cax=cax, orientation='vertical')
      cbar.ax.set_yscale('log')
      cbar.ax.set_ylabel(r"$\chi^2$",fontsize=35,rotation=0)
      cur_axis.set_ylabel(Par_name1,fontsize=font_size,rotation=0)
      cur_axis.set_xlabel(Par_name2,fontsize=font_size)

      if nb_params<7 and cost_function == "reduced_chi2" and uncertainty_contour :
            cur_axis.contour(param_grids[1],param_grids[0],y,levels=[best_chi2 + thresholds[nb_params-1]],colors=['orange'],zorder=2)
   
   if (cur_axis is None):
      plt.tight_layout()
   plt.tight_layout()
   return cur_axis








# =============================================================================
#                      OPR   
# =============================================================================    




def df_OPR_errorbar(raies, opr):
    """
    Trace un graphique avec les valeurs moyennes de OPR et les barres d'erreur,
    en conservant l'ordre d'apparition des chaînes de caractères dans la liste Raies.

    Args:
        raies (list): Liste des valeurs de Raies.
        opr (list): Liste des valeurs de OPR.
    """

    # Créer un DataFrame pandas
    df = pd.DataFrame({'Raies': raies, 'OPR': opr})

    # Convertir la colonne Raies en catégorie ordonnée
    df['Raies'] = pd.Categorical(df['Raies'], categories=df['Raies'].unique(), ordered=True)

    # Regrouper les données par Raies et calculer les statistiques
    stats = df.groupby('Raies')['OPR'].agg(['mean', 'min', 'max'])

    # Calculer les erreurs (différences entre max/min et la moyenne)
    stats['erreur_basse'] = stats['mean'] - stats['min']
    stats['erreur_haute'] = stats['max'] - stats['mean']
    return(stats)


def marker_By_V(raies, opr, palette):
    """
    Trace les marqueurs centraux de chaque groupe "v1_J2" en utilisant les couleurs de la palette.

    Args:
        raies (list): Liste des valeurs de Raies.
        opr (list): Liste des valeurs de OPR.
        palette (list): Liste des couleurs de la palette.
    """

    # Créer un DataFrame pandas
    df = pd.DataFrame({'Raies': raies, 'OPR': opr})

    # Extraire les valeurs numériques et les couleurs associées
    valeurs_numeriques = []
    couleurs = []
    for raie in raies:
        valeur_numerique = int(raie.split('-')[0][1:])  # Extraire la valeur numérique après le "v"
        valeurs_numeriques.append(valeur_numerique)
        couleurs.append(palette[valeur_numerique - 1])  # Utiliser la valeur numérique comme indice de la palette

    df['Valeur_Numerique'] = valeurs_numeriques
    df['Couleur'] = couleurs

    # Regrouper les données par Raies et calculer la moyenne
    moyennes = df.groupby('Raies')['OPR'].mean()

    return(moyennes,df)









# =============================================================================
#                      Stat Df   
# =============================================================================   



def df_stats_and_histograms(df, columns=['Pth', 'Go', 'Omega']):
    """
    Calculates descriptive statistics (mean, standard deviation at 1, 2, and 3 sigma)
    and plots histograms for multiple specified columns of a DataFrame.
    The statistical values are returned in scientific format.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list, optional): A list of column names to analyze.
                                   Defaults to ['Pth', 'Go', 'Omega'].
    """
    num_cols = len(columns)
    plt.figure(figsize=(15, 5 * num_cols // 2))  # Adjust figure size based on the number of columns

    for i, col in enumerate(columns):
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame.")
            continue

        # Calculate the mean
        mean_val = df[col].mean()
        print(f"Mean of {col}: {mean_val:.2e}")

        # Calculate the standard deviation
        std_val = df[col].std()
        print(f"Standard Deviation of {col}: {std_val:.2e}")

        # Calculate values at 1, 2, and 3 sigma
        print(f"Values at different sigmas for {col}:")
        print(f"  +/- 1 Sigma: [{mean_val - std_val:.2e}, {mean_val + std_val:.2e}]")
        print(f"  +/- 2 Sigma: [{mean_val - 2 * std_val:.2e}, {mean_val + 2 * std_val:.2e}]")
        print(f"  +/- 3 Sigma: [{mean_val - 3 * std_val:.2e}, {mean_val + 3 * std_val:.2e}]")
        print("-" * 30)

        # Plot histogram
        plt.subplot(num_cols // 2 + num_cols % 2, 2, i + 1)
        df[col].hist(bins=20, edgecolor='black')
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    
    
# =============================================================================
#VIDEO 
# =============================================================================  

def get_model(approximator,param1_val, param2_val, obs_data,Model="SigAa",scaling="auto"):
    """
    Generates and fits a model using the interpolator.

    Args:
        param1_val (float): Value of the first parameter.
        param2_val (float): Value of the second parameter.
        obs_data: Observational data for fitting.

    Returns:
        A fitted model object.
    """
    # Call the interpolator
    model_choice = approximator[Model]([param1_val, param2_val])
    # Fit the model
    if scaling == "auto":
        fitted_model = ism.fit_tools.find_best_scaling_factor_single_model(obs_data, model_choice,scaling_factor_bounds=[0,20])
    else : 
        model_choice.set_scaling_factor(scaling)
        fitted_model = model_choice
        
    return fitted_model



def create_animation_frame(models_to_plot, obs_data, frame_path, name_list, colors, styles, fig_title=None,genre="Excitation",Level=None):
    """
    Generates and saves a single plot frame using viz.Excitation_Diagram.
    It assumes viz.Excitation_Diagram either saves the plot and closes it, or plots on current figure.
    If it plots on current figure, this function handles saving and closing.

    Args:
        models_to_plot (list): List of models to plot.
        obs_data: Observational data.
        frame_path (str): Full path (including filename) to save the frame.
        name_list (list): List of names for the legend.
        colors (list): List of colors for the models.
        styles (list): List of line styles for the models.
        fig_title (str, optional): Title for the figure.
    """
    # It's good practice to ensure the figure is clear if viz.Excitation_Diagram reuses figures
    # plt.clf() # Clear current figure - uncomment if needed.
    if genre == "Excitation" :
        Excitation_Diagram(models_to_plot,
                                 obs_data,
                                 NameList=name_list,
                                 Name=fig_title, # For plot title, if applicable
                                 color=colors,
                                 style=styles,
                                 threshold=True,
                                 Reg=False,
                                 Save=False, # Set Save to False if we save manually with plt.savefig
                                 Path=None,  # Path is not used if Save=False
                                 Annotation=False,
                                 Full=True)
        
        # Manual save is often more reliable for precise filename control
        plt.savefig(frame_path)
        plt.close() # Close the plot to free up memory
        
    elif genre =="Intensity" : 
        To_plot= [obs_data,models_to_plot[0]]
        fig = ism.ap_tools.plot_intensity_diagram2( To_plot ,
                                                       ylabel="Intensity [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]",
                                                       title=fig_title,
                                                       Noms=["",""],
                                                       couleur=["black","blue"],
                                                       Size=(21,9))
        plt.grid(linestyle='dotted', alpha=0.4)
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close() # Close the plot to free up memory
        
    elif genre == "Excitation_v" :
        if Level==None : 
            print ("Please mention Level variable to know which levels you want to plot")
        Excitation_Diagram(models_to_plot,
                                 obs_data,
                                 NameList=name_list,
                                 Name=fig_title, # For plot title, if applicable
                                 color=colors,
                                 style=styles,
                                 threshold=True,
                                 Reg=False,
                                 Save=False, # Set Save to False if we save manually with plt.savefig
                                 Path=None,  # Path is not used if Save=False
                                 Annotation=False,
                                 Full=False, Level=Level)
        
        # Manual save is often more reliable for precise filename control
        plt.savefig(frame_path)
        plt.close() # Close the plot to free up memory
        
        
def generate_excitation_animation(
    obs_data,
    approximator,
    param_to_vary_idx,
    fixed_param_val,
    varying_param_range,
    num_frames,
    output_filename_base="excitation_animation",
    animation_mode="single", # "single" or "cumulative"
    temp_frame_dir="/Users/aurelien/Desktop/Stage_M2/Traitement/temp_frames",
    duration_per_frame=0.5,
    fps=2,
    Model="SigAa",
    scaling="auto",
    Title="",
    genre ="Excitation",Level=None
):
    """
    Generates an animation (GIF and MP4) of the Excitation Diagram by varying one parameter.

    Args:
        obs_data: Observational data.
        param_to_vary_idx (int): Index of the parameter to vary (0 or 1).
        fixed_param_val (float): Value of the parameter that remains fixed.
        varying_param_range (tuple): (min_val, max_val) for the varying parameter.
        num_frames (int): Number of frames in the animation.
        output_filename_base (str): Base name for the output GIF and MP4 files (without extension).
        animation_mode (str): 'single' for one curve per frame, 'cumulative' for accumulating curves.
        temp_frame_dir (str): Directory to store temporary frame images.
        duration_per_frame (float): Duration of each frame in the GIF (seconds).
        fps (int): Frames per second for the MP4 video.
    """
    varying_param_values = np.linspace(varying_param_range[0], varying_param_range[1], num_frames)
    
    # Create temporary directory for frames
    if os.path.exists(temp_frame_dir):
        shutil.rmtree(temp_frame_dir) # Clean up if it exists
    os.makedirs(temp_frame_dir)

    frame_files = []
    
    # For cumulative mode
    all_models_cumulative = []
    all_names_cumulative = []
    all_colors_cumulative = []
    all_styles_cumulative = []
    
    # Define colors for cumulative mode (e.g., from a colormap)
    colormap = plt.cm.viridis # or plt.cm.jet, plt.cm.plasma, etc.
    cumulative_plot_colors = [colormap(i) for i in np.linspace(0, 0.9, num_frames)] # Avoid very light colors if at the end of map

    for i, val in enumerate(varying_param_values):
        current_params = [0.0, 0.0]
        current_params[param_to_vary_idx] = val
        current_params[1 - param_to_vary_idx] = fixed_param_val
        
        param1 = current_params[0]
        param2 = current_params[1]

        # Generate the current model
        current_model = get_model(approximator,param1, param2, obs_data,Model,scaling=scaling)
        
        frame_filepath = os.path.join(temp_frame_dir, f"frame_{i:03d}.png")
        frame_files.append(frame_filepath)
        
        fig_title_text = Title+f"  Pth={param1:.2e}, Go={param2:.2e}"
        current_model_legend_name = ""#f"Model (P{param_to_vary_idx+1}={val:.2e})"



        if animation_mode == "single":
            create_animation_frame(
                models_to_plot=[current_model],
                obs_data=obs_data,
                frame_path=frame_filepath,
                name_list=[current_model_legend_name],
                colors=["blue"], # Single color for the single line
                styles=["-"],    # Single style
                fig_title=fig_title_text,
                genre=genre,
                Level=Level
            )
            
            
            
        elif animation_mode == "cumulative":
            all_models_cumulative.append(current_model)
            all_names_cumulative.append(current_model_legend_name)
            all_colors_cumulative.append(cumulative_plot_colors[i])
            all_styles_cumulative.append("-") # Or cycle through styles: [":", "--", "-."][i % 3]
            
            create_animation_frame(
                models_to_plot=all_models_cumulative,
                obs_data=obs_data,
                frame_path=frame_filepath,
                name_list=all_names_cumulative,
                colors=all_colors_cumulative,
                styles=all_styles_cumulative,
                fig_title=fig_title_text,
                genre=genre,
                Level=Level
            )
            
        else:
            raise ValueError("animation_mode must be 'single' or 'cumulative'")
        
        
        
        print(f"Generated frame {i+1}/{num_frames}")

    # Create GIF and Video
    images = []
    for frame_file in frame_files:
        images.append(imageio.imread(frame_file))
    
    gif_path = f"{output_filename_base}.gif"
    mp4_path = f"{output_filename_base}.mp4"

    #imageio.mimsave(gif_path, images, duration=duration_per_frame * 1000) # imageio duration is in ms for GIFv2
    imageio.mimsave(mp4_path, images, fps=fps)
    
    print(f"\nAnimation saved as {gif_path} and {mp4_path}")

    # Clean up temporary frames
    shutil.rmtree(temp_frame_dir)
    print(f"Temporary frame directory {temp_frame_dir} removed.")
    











def animate_scatter_on_contour_2_paramsV2(contour_function, approx_dic, obs, bestmod_dic, x_scatter_values, y_scatter_values, output_path, base_filename="scatter_frame", contour_color="green", ref_point=None, title="", label_names=None, golim=None, scatter_size=750, scatter_color="blue", scatter_edgecolor='white'):
    """
    Generates contour plots with a moving scatter point (both x and y varying), saving each frame as a PNG.

    Args:
        contour_function (callable): The function used to generate the contour plot.
        approx_dic (dict): Dictionary containing approximation data.
        obs: Observation data.
        bestmod_dic (dict): Dictionary containing best model data.
        x_scatter_values (list or array-like): List of x-coordinates for the scatter point.
        y_scatter_values (list or array-like): List of y-coordinates for the scatter point.
        output_path (str): The directory where the generated PNG files will be saved.
        base_filename (str, optional): The base name for the saved PNG files. Defaults to "scatter_frame".
        contour_color (str, optional): Color of the contour. Defaults to "green".
        ref_point (list, optional): A fixed reference point. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "".
        label_names (list, optional): List of axis labels. Defaults to None.
        golim (float, optional): Limit for the Go axis. Defaults to None.
        scatter_size (float, optional): Size of the scatter point. Defaults to 750.
        scatter_color (str, optional): Color of the scatter point. Defaults to "blue".
        scatter_edgecolor (str, optional): Color of the scatter point edge. Defaults to 'white'.
    """
    os.makedirs(output_path, exist_ok=True)
    num_frames = min(len(x_scatter_values), len(y_scatter_values))
    for i in range(num_frames):
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel(label_names[0] if label_names else "X-axis")
        ax.set_ylabel(label_names[1] if label_names else "Y-axis")
        ax.set_title(title)
        if golim is not None and label_names and label_names[0] == r"$G_0$":
            ax.set_xlim(0, golim)

        contour_function(approx_dic["IC63"], obs, ("Go", "Pgas"),
                         bestmod_dic["IC63"], couleur=contour_color,
                         Title="", LabelName=label_names, GoLim=golim, ax=ax)

        current_x = x_scatter_values[i]
        current_y = y_scatter_values[i]
        ax.scatter(current_x, current_y, s=scatter_size, c=scatter_color, edgecolors=scatter_edgecolor)
        if ref_point:
            ax.plot(ref_point[0], ref_point[1], marker='x', color='blue', label="Fixed Point")
            ax.legend()

        filename = os.path.join(output_path, f"{base_filename}_{i+1:03d}.png")
        plt.savefig(filename)
        plt.close()
    print(f"Saved {num_frames} frames to: {output_path}")

def create_video_from_frames(image_folder, output_video_path, frame_rate=1):
    """
    Creates a video from a sequence of image files in a folder and then deletes the folder.

    Args:
        image_folder (str): The path to the folder containing the image files.
        output_video_path (str): The path where the output video file will be saved (e.g., "animation.mp4").
        frame_rate (int, optional): The frame rate of the video in frames per second. Defaults to 1.
    """
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not images:
        print(f"No PNG images found in: {image_folder}")
        return

    with imageio.get_writer(output_video_path, fps=frame_rate) as writer:
        for filename in images:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"Video created at: {output_video_path}")

    try:
        shutil.rmtree(image_folder)
        print(f"Temporary folder '{image_folder}' deleted.")
    except OSError as e:
        print(f"Error deleting temporary folder '{image_folder}': {e}")








#---Contour 
def animate_scatter_on_contour(obs_data,approximator,bestmodel,grid, x_values, y_values, output_path, 
                               base_filename="contour_frame",col="blue",
                               ref_point=None, title="", GoLim=None, scatter_size=750, scatter_color="blue", scatter_edgecolor='white'):
    """
    Generates contour plots with a moving scatter point, saving each frame as a PNG.

    Args:
        contour_function (callable): The function used to generate the contour plot.
        approx_dic (dict): Dictionary containing approximation data.
        obs: Observation data.
        bestmod_dic (dict): Dictionary containing best model data.
        x_values (list or array-like): List of x-coordinates for the scatter point.
        y_values (list or array-like): List of y-coordinates for the scatter point.
        output_path (str): The directory where the generated PNG files will be saved.
        base_filename (str, optional): The base name for the saved PNG files. Defaults to "scatter_frame".
        ref_point (list, optional): A fixed reference point. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "".
        label_names (list, optional): List of axis labels. Defaults to None.
        golim (float, optional): Limit for the Go axis. Defaults to None.
        scatter_size (float, optional): Size of the scatter point. Defaults to 750.
        scatter_color (str, optional): Color of the scatter point. Defaults to "blue".
        scatter_edgecolor (str, optional): Color of the scatter point edge. Defaults to 'white'.
    """
    os.makedirs(output_path, exist_ok=True)
    num_frames = min(len(x_values), len(y_values))
    for i in range(num_frames):


        contour=ism.ap_tools.fit_quality_contour_plots_singlepair2(approximator[grid], obs_data, ("Go", "Pgas"),
                         bestmodel[grid], couleur=col,
                         #RefPoint=ref_point, # RefPoint is commented out in your example
                         #RefPoint2=point_test, # point_test is not varied here
                         Title="", LabelName=[r"$G_0$",r"$P_{th}$"], GoLim=GoLim) # Pass the axes object

        current_x = x_values[i]
        current_y = y_values[i]
        contour.scatter(current_x, current_y, s=scatter_size, c=scatter_color, edgecolors=scatter_edgecolor)


        filename = os.path.join(output_path, f"{base_filename}_{i+1:03d}.png")
        plt.savefig(filename)
        plt.close()
    print(f"Saved {num_frames} frames to: {output_path}")

def create_video_from_frames(image_folder, output_video_path, frame_rate=1):
    """
    Creates a video from a sequence of image files in a folder and then deletes the folder.

    Args:
        image_folder (str): The path to the folder containing the image files.
        output_video_path (str): The path where the output video file will be saved (e.g., "animation.mp4").
        frame_rate (int, optional): The frame rate of the video in frames per second. Defaults to 1.
    """
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not images:
        print(f"No PNG images found in: {image_folder}")
        return

    with imageio.get_writer(output_video_path, fps=frame_rate) as writer:
        for filename in images:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"Video created at: {output_video_path}")

    try:
        shutil.rmtree(image_folder)
        print(f"Temporary folder '{image_folder}' deleted.")
    except OSError as e:
        print(f"Error deleting temporary folder '{image_folder}': {e}")
        
        
def plot_df_comparison_and_error(df1: pd.DataFrame, 
                                  df2: pd.DataFrame, 
                                  col_name: str, 
                                  epsilon: float = 1e-9,
                                  df1_label: str = 'df_co', 
                                  df2_label: str = 'df_combined (with override)'):
    """
    Generates a two-panel plot to compare a specified column from two DataFrames
    and show their relative error.

    Args:
        df1 (pd.DataFrame): The first DataFrame (e.g., df_co).
        df2 (pd.DataFrame): The second DataFrame (e.g., df_combined_sorted).
        col_name (str): The name of the column to plot from both DataFrames 
                        (e.g., 'Aij_s-1').
        epsilon (float): A small value added to the denominator for relative error 
                         calculation to prevent division by zero. Defaults to 1e-9.
        df1_label (str): Label for df1 in the plot legend.
        df2_label (str): Label for df2 in the plot legend.
    """

    # Ensure both dataframes have the specified column
    if col_name not in df1.columns or col_name not in df2.columns:
        raise ValueError(f"The column '{col_name}' must exist in both DataFrames.")
    
    # Ensure indices are aligned for error calculation
    # We will assume df1 and df2 are already sorted and their indices
    # represent the same transitions in the same order as in your example.
    # If not, you might need to merge or align them based on common identifiers.
    
    # Calculate relative error
    # Using df2 for numerator and df1 for denominator as per your example
    error_relative_common = np.abs(df2[col_name] - df1[col_name]) / \
                            (df1[col_name] + epsilon) * 100

    # --- Plotting ---
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(21, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Top Panel: Scatter plot of the specified column from both DataFrames
    ax.scatter(df1.index, df1[col_name],
               color='red', marker='o', s=20, label=f'{col_name} from {df1_label}')
    ax.scatter(df2.index, df2[col_name],
               color='blue', marker='x', s=20, alpha=0.5, label=f'{col_name} from {df2_label}')

    # Configure top panel axes and title
    ax.set_ylabel(f'{col_name} (s-1)')
    ax.set_title(f'Comparison of {col_name} Coefficients: {df1_label} vs. {df2_label}')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.7)
    ax.legend()

    # Bottom Panel: Relative Error Plot
    ax2.plot(df1.index, error_relative_common, 
             label=r'Relative Error ($\%$)', color='green', marker="o", linestyle=":")
    ax2.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    
    # ax.set_xscale('log') 
    # ax2.set_xscale('log') 
    #ax2.set_xscale('log') # Doit être en log pour suivre ax1
    ax2.set_xlim([-2,5e3])

    ax2.set_ylim([-2,100])
    #ax2.set_yscale('log') # Log scale for relative error, as per your example
    ax2.set_xlabel('Transition Index (sorted by vu, Ju, vl, Jl)', fontsize=25)
    ax2.set_ylabel(r' Relative Error [$\%$]', fontsize=15)
    ax2.grid(True, which="both", ls="-", alpha=0.6)
    ax2.legend() # Add legend for error plot

    plt.tight_layout()
    plt.show()
    
    
    
def plot_df_comparison_collisions(df1: pd.DataFrame, 
                                  df2: pd.DataFrame, 
                                  col_name1 , 
                                  col_name2, 
                                  T_common,
                                  epsilon: float = 1e-9,
                                  df1_label: str = 'df_co', 
                                  df2_label: str = 'df_combined (with override)',
                                  Numup = 6,
                                  ylim=100
                                  ):
    
    df_temp1 = df1[df1['UP'] == Numup]
    df_temp2 = df2[df2['UP'] == Numup]
    assert len(df_temp1)==len(df_temp2)
    # --- Plotting ---
    
    
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(21, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Top Panel: Scatter plot of the specified column from both DataFrames
    
    for i in range (len(df_temp1)):
        line1 = df_temp1.iloc[i]
        line2 = df_temp2.iloc[i]


        col1 = line1[col_name1]
        col2 = line2[col_name2]
        #col1_values = line1[col_names_for_rates].values # Get the actual NumPy array of values
        #col2_values = line2[col_names_for_rates].values # Get the actual NumPy array of values

        ax.plot(T_common, col1, color=palette[i], 
               marker='o', linestyle ="-",
               label=f"0-0 J{Numup-1}_{int(line1['LOW'])-1} from {df1_label}")
        ax.plot(T_common, col2, color=palette[i],
                   marker='x', alpha=0.9, linestyle=":",
                   label=f"0-0 {Numup-1}_{int(line2['LOW'])-1} from {df2_label}")



        # Bottom Panel: Relative Error Plot
        
        error_relative_common = np.abs(col1.values - col2.values) / \
                                (col1.values + epsilon) * 100
                                
        ax2.plot(T_common, error_relative_common, 
                 label=f"{Numup-1}_{int(line2['LOW'])-1} ", color=palette[i], marker="o", linestyle="--")
        
    # Configure top panel axes and title
    ax.set_ylabel(r'Collision Rates [$cm^3 \, s^{-1}$]')
    ax.set_title(f'Comparison of Collision rates Coefficients: {df1_label} vs. {df2_label}')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.7)
    ax.legend()
    
    ax2.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    
    # ax.set_xscale('log') 
    # ax2.set_xscale('log') 
    ax2.set_xscale('log') # Doit être en log pour suivre ax1
    ax2.set_xlim([T_common[0]-5,T_common[-1]+5])

    ax2.set_ylim([-0.5,ylim])

    #ax2.set_yscale('log') # Log scale for relative error, as per your example
    ax2.set_xlabel('Temperature [K]', fontsize=25)
    ax2.set_ylabel(r' Relative Error [$\%$]', fontsize=15)
    ax2.grid(True, which="both", ls="-", alpha=0.6)
    ax2.legend() # Add legend for error plot

    plt.tight_layout()
    plt.show()
    
    
    
    # =============================================================================
    #                            Nouvelle versipn Excitaiton  
    # =============================================================================    



def PyISM_to_df (el): 

    Lines = el.get_observable_names()
    Int= el.get_observable_vector()
    
    if isinstance(el,Observation): 
        Err = el.get_error_vector()
        Genre = "Observation"
        
    elif isinstance(el, Model): 
        Err = el.get_scaling_factor()
        Genre = "Model"
            
    data = {'Lines': Lines, 'Intensity': Int, 'Err/Scaling': Err,'Genre':Genre}
    df_temp=pd.DataFrame(data)
    df_temp2 = pd.merge(df_temp, df_info_co, on='Lines', how='left')
    return df_temp2


def add_Nu (df,err=None):
    df["Nu"] =(  4 * np.pi * df["Wavelenght[microns]"] * 1e-4 * df["Intensity"] ) / (h * clum * df["Aul[s-1]"])
    if df["Genre"][0] =="Observation" :
        df["Nu_err"] = (4 * np.pi * df["Wavelenght[microns]"] * 1e-4 * df["Err/Scaling"] ) / (h * clum * df["Aul[s-1]"])
        if err!=None : 
            df["Nu_err"] = (4 * np.pi * df["Wavelenght[microns]"] * 1e-4 * err * df["Intensity"] ) /( h * clum * df["Aul[s-1]"])
    return df 



def Excitation_Diagram_Ax(Model, Observation, NameList,style, Name=None,color="blue", Reg=False, Save=False, Path="",
                       Annotation=False, Full=False, Level=None,FillIdx =None, threshold=True,
                       Front="Model",Details=True,matchcolor=False,mol="h2",OP_visible=True,
                       ObsCol=None, ax=None) : # <--- AJOUT DE ax EN DERNIER
    """Crée un diagramme d'excitation en séparant par v, et en mettant en évidence les différences O et P"""
    palette=["g","r","c","m","y","k","tab:orange","tab:brown","tab:grey","salmon","olive","gold","orchid"]#"b",

    marqueur=['^','o']#Para = pair = 0 et ortho = impair = 1

    Raies=Observation.get_observable_names()
    I_Obs=Observation.get_observable_vector()
    E_list=Observation.get_error_vector()

    scaling_list,I_Model=[],[]
    if Model!=None:
        for i in range(len(Model)):
            
            scaling_list.append(Model[i].get_scaling_factor())
            I_Model.append(Model[i].get_observable_vector())
            # Correction: la boucle suivante ne modifiait pas réellement I_Model[i], 
            # mais nous conservons la logique originale pour ne modifier QUE l'appel.
            for el in I_Model[i] : 
                el=el*scaling_list[i]
    
    NameList2= []
    for i in range (len(Model)) : 
        Params = Model[i].get_parameter_vector()
        echelle = Model[i].get_scaling_factor()
        if Details == True :
            NameList2.append(r": P$_{\text{th}}$" + f" = {Params[0]:.2e} K.$cm^{{-3}}$, $G_0$ = {Params[1]:.2e}, $\Omega$={echelle:.1e}")
        elif Details == False : 
            NameList2.append("")
            
    #Comparaison avec la data base du code
    main_root="/Users/aurelien/Desktop/Stage_M2/"
    rootbase=main_root+"PDR/1.7/"
    namebase=f"all_{mol}_lines_info"
    
    # Gestion des fichiers pour éviter l'erreur si le fichier n'est pas là
    try:
        with open(rootbase+namebase+".dat","r") as doc_base:
            file_base=doc_base.read();  ligne_base=file_base.split("\n");  header_base=ligne_base[0:2]
    except FileNotFoundError:
        print(f"Erreur: Fichier de base de données non trouvé à {rootbase+namebase}.dat")
        return

    Data_base=[]
    for i in range (len(Raies)) :
        temp=i
        nom_raie=Raies[i]

        ligne_sans_espaces = ligne_base[temp].strip()
        Line = ligne_sans_espaces.split(',', 1)[0]
        while nom_raie!=Line and temp<len(ligne_base):
            temp+=1
            ligne_sans_espaces = ligne_base[temp].strip()
            Line = ligne_sans_espaces.split(',', 1)[0]
        if nom_raie==ligne_base[temp].split(',')[0] : 
            Data_base.append(ligne_base[temp])
        else : 
            print('Error : Not in database, please check if nomenclature is correct.')
            
    #Récupération des paramètres nécessaires      
    Wave_list,Eul_list,A_list,Eu_list,gu_list,ju_list=[],[],[],[],[],[]     
    for i in range (len(Data_base)):
        raie_sel=Data_base[i].split(',')
        try:
            Wave_list.append(float(raie_sel[11].split('\t')[-1]))
            A_list.append(float(raie_sel[10].split('\t')[-1]))
            Eul_list.append(float(raie_sel[9].split('\t')[-1]))
            Eu_list.append(float(raie_sel[7].split('\t')[-1]))
            gu_list.append(float(raie_sel[5].split('\t')[-1]))
            ju_list.append(float(raie_sel[2].split('\t')[-1]))
        except IndexError:
            print(f"Erreur de formatage dans la ligne de la base de données: {Data_base[i]}")
            continue
            
    #Calcul de la colonne densité pour les observations dans Nu_Obs
    Nu_Obs=[4*np.pi * (Wave_list[i] * 1e-4)    * I_Obs[i]/(h*clum*A_list[i]) for i in range (len(A_list))] #Attnetion lambda data en micron -> cm 
    if threshold ==True : 
        Nu_Err=[4*np.pi * (Wave_list[i] * 1e-4)  * (0.3*I_Obs[i])/(h*clum*A_list[i]) for i in range (len(A_list))]
    elif threshold ==False : 
        Nu_Err=[4*np.pi * (Wave_list[i] * 1e-4)  * (E_list[i])/(h*clum*A_list[i]) for i in range (len(A_list))]
    
    Y_Obs=[(Nu_Obs[i]/gu_list[i]) for i in range (len(Nu_Obs))]
    Y_Err=[(Nu_Err[i]/gu_list[i]) for i in range (len(Nu_Obs))]
    
    Nu_Model,Y_Model=[],[]
    for k in range (len(Model)):
        Nu_Model.append( [4*np.pi * (Wave_list[i] * 1e-4) *I_Model[k][i]/(h*clum*A_list[i]) for i in range (len(A_list))] )
        Y_Model.append([(Nu_Model[k][i]/gu_list[i]) for i in range (len(Nu_Obs))] )
    
    #Répartition par niveau vibrationnel 
    try:
        max_v_index = int(Raies[-1].split('-')[0])+1
    except (IndexError, ValueError):
        max_v_index = 1
        
    Lines_Vibrationnel=[[] for i in range(max_v_index)]
    Y_Obs_vibrationnel=[[] for i in range(max_v_index)]
    Error_Vib=[[] for i in range(max_v_index)]
    E_vibrationnel=[[] for i in range(max_v_index)]
    Y_model_vibrationnel=[ [ [] for i in range(max_v_index) ] for k in range (len(Model))]
    
    for i in range (len(Raies)): 
        try:
            Vh=int(Raies[i].split("-")[0])
            Lines_Vibrationnel[Vh].append(Raies[i])
            Y_Obs_vibrationnel[Vh].append(Y_Obs[i])
            Error_Vib[Vh].append(Y_Err[i])
            E_vibrationnel[Vh].append(Eu_list[i])
            for k in range(len(Model)):
                Y_model_vibrationnel[k][Vh].append(Y_Model[k][i])
        except (IndexError, ValueError):
            continue
            
    if E_vibrationnel[0]== [] : #Check si rotationel ou pas 
        E_vibrationnel = E_vibrationnel[1:]   
        Y_Obs_vibrationnel = Y_Obs_vibrationnel[1:]
        Lines_Vibrationnel = Lines_Vibrationnel[1:]
        Error_Vib = Error_Vib[1:]
        for k in range (len(Model)):
            Y_model_vibrationnel[k]=Y_model_vibrationnel[k][1:]
            
            
    resid_list=[]
    
    if Level==None : 
        minV = 0
        maxV=len(Lines_Vibrationnel)
        
    elif Level!= None : 
        minV=Level[0]
        maxV=Level[1]
    
    
    if Front=="Model": 
        order =[1,2]
    elif Front=="Obs":
        order=[2,1]
        
    #-------------- Graphique----------------------------------
    if ObsCol!=None : 
        palette=[ObsCol for i in range (20)]
    
    # --- DÉCISION DE LA CRÉATION DE L'AXE ---
    # Si ax n'est pas fourni, nous conservons le comportement original de création de figure
    ax_created = False
    if ax is None:
        if Full==True : #Tracé de tous les niveaux sur le meme diagramme 
            fig, ax = plt.subplots(figsize=(13,11))
            ax_created = True
        # Note: Si Full=False, la figure/axe est créé(e) dans la boucle for i in range(minV, maxV):
            
    
    if Full==True : #Tracé de tous les niveaux sur le meme diagramme 
        
        ax.scatter(E_vibrationnel[0][0],Y_Obs_vibrationnel[0][0] ,c=palette[0],
                      marker=marqueur[0],)# label= "Para levels")
        
        ax.scatter(E_vibrationnel[0][0],Y_Obs_vibrationnel[0][0] ,c=palette[0],
                      marker=marqueur[1], )#label= "Ortho levels")
        
        for k in range(len(Model)):
            ax.plot(E_vibrationnel[0],Y_model_vibrationnel[k][0],linestyle=style[k],
                      color=color[k],alpha=0.9,label={NameList[k]+NameList2[k]},marker="+")
            
        if FillIdx!=None :
             # Code de remplissage omis pour concision
             pass
        
    for i in range (minV,maxV): #Par V
        if Full==False : #Tracé par v séparé
            # Si ax n'est pas fourni, on crée une nouvelle figure/axe pour chaque niveau V
            if ax is None or ax_created is False: 
                fig, ax = plt.subplots(figsize=(16,9))
                # Note: ax_created reste False car on crée un nouvel axe à chaque itération ici
        
    
        for j in range (len(Lines_Vibrationnel[i])):# On trace les observations par niveau vibrationnel
            if OP_visible ==True :
                # Mise à jour: Tentative de gestion d'erreur minimaliste pour OP_visible
                try:
                    line_name_part = Lines_Vibrationnel[i][j].split(',')[0].strip()
                    # On cherche le J à l'intérieur des parenthèses ou après le dernier espace (si le format est 'X(J)')
                    j_value_str = line_name_part.split('(')[-1].split(')')[0]
                    OP=int(j_value_str)%2
                except (IndexError, ValueError):
                    # Si l'analyse échoue, on suppose Para (OP=0)
                    OP=0
            else :
                OP=0
            
            ax.errorbar(E_vibrationnel[i][j],Y_Obs_vibrationnel[i][j] , yerr=Error_Vib[i][j],c=palette[i],
                          markersize=10,marker=marqueur[OP],markeredgecolor='white',
                          capsize=3, elinewidth= 2 , ecolor=(0, 0, 0, 0.7),zorder=order[0])
            
            yerr = np.array([np.array([0.5 * Y_Obs_vibrationnel[i][j]]), np.array([1 * Y_Obs_vibrationnel[i][j]]) ])
            
            ax.errorbar(E_vibrationnel[i][j],Y_Obs_vibrationnel[i][j] , yerr=yerr,c=palette[i],
                          fmt="none", capsize=1, elinewidth= 2 , ecolor=palette[i],zorder=order[0],alpha=0.2)
            
            ax.annotate(text=f"v={i+1}", xy=(E_vibrationnel[i][0],Y_Obs_vibrationnel[i][0]),
                          xytext=(E_vibrationnel[i][0]+500,Y_Obs_vibrationnel[i][0]*1.8),
                          color=palette[i],fontweight='bold')
            
        for k in range(len(Model)) :    
            if matchcolor == True : 
                color = palette
                idxcolor = i
            elif matchcolor ==False : 
                idxcolor = k
                
            if Full==False : 
                ax.plot(E_vibrationnel[i],Y_model_vibrationnel[k][i],linestyle=style[k],color=color[idxcolor],alpha=0.9,
                          label={NameList[k]+NameList2[k]},zorder=order[1],marker="x")
                
            elif Full==True : #Que pour le label 
                ax.plot(E_vibrationnel[i],Y_model_vibrationnel[k][i],linestyle=style[k],
                          color=color[idxcolor],alpha=0.9,zorder=order[1],marker="x",markersize=6)    
                
            
        if Reg==True and len(E_vibrationnel[i])>2:#Si on ajoute la régression
            X=np.array(E_vibrationnel[i]).reshape(-1,1)
            model = LinearRegression();model.fit(X,np.log10(Y_Obs_vibrationnel[i]))
            coef = format(model.coef_[0],'.2e');r2 =int( model.score(X, np.log10(Y_Obs_vibrationnel[i])*100))
            y_pred = 10**model.predict(X)
            residuals=(np.log10(abs(Y_Obs_vibrationnel[i] - y_pred)))
            
            resid_list.append(residuals)
            ax.plot(X, y_pred, color='red',linestyle=":",label=f"Vup={i},T={-int(1/float(coef))} K ")
            
        xmin,xmax,ymin,ymax=np.min(E_list),np.max(E_list),np.min(Y_Obs)/2,np.max(Y_Obs)*3
    
        if Annotation==True :               #Annotation niveaux 
            for j in range (len(E_vibrationnel[i])): 
                if Full==True : 
                    ymax=np.max(Y_Obs_vibrationnel[0])
                ax.axvline(x=E_vibrationnel[i][j], color='black', linewidth=1, ymin=0.85,ymax=0.9)
                ax.annotate(Lines_Vibrationnel[i][j],
                              xy=( E_vibrationnel[i][j] , (ymax)*1.25 ), 
                              xytext=(E_vibrationnel[i][j] + 20 ,  (ymax)*1.3 ),
                              fontsize=10,rotation=90)
    
        ax.grid(linestyle='dotted', alpha=0.7)
        ax.semilogy()
        ax.set_ylim(ymin,ymax*1.2)
    
    
            
        if Full==False : 
            ax.set_title(f"{Name} Excitation Vup={i}")
            ax.legend(fontsize=20)
            mpl.rcParams['xtick.labelsize']=35
            mpl.rcParams['ytick.labelsize']=35
            ax.set_ylabel(r'$N_u / g_u$ $[cm^{-2}]$',fontsize=20)
            ax.set_xlabel(r"$E_u$ [K]",fontsize=30)
    
    
    
        elif Full ==True : 
            if Name !=None : 
                plt.title(f"{Name} Excitation Diagram all v")
            ax.legend(fontsize=12)
            ax.set_ylabel(r'$N_u / g_u$ $[cm^{-2}]$',fontsize=20)
            #ax.set_xlabel(r"$E_u$ [K]",fontsize=25)
    
        plt.tight_layout()
        
        if Save==True : 
            if Full==True : 
                plt.savefig(Path+Name+f"_excitation_diagram.pdf",dpi=200)
            elif  Full==False : 
                plt.savefig(Path+Name+f"_excitation_diagram_v{i}.pdf",dpi=200)       
    
    if Reg==True :     
        residus=[]
        for el in resid_list:
            for el2 in el :
                residus.append(el2)
        print("Moyenne, Max, Longueur des résidus écart (obs-LinReg)")
        print(np.mean(residus),np.max(residus),len(residus))









def plot_pr_pairs(
    df_lines: pd.DataFrame,
    df_spectrum: pd.DataFrame,
    col_jup: str = "J_up",
    col_band: str = "Band",
    col_branch: str = "Branch",
    col_wave_line: str = "wavelength",
    col_name: str = "Line",
    col_wave_spec: str = "Wavelength",
    col_int_spec: str = "Int",
    window: float = 0.05
):
    """
    Identifies and plots P/R branch pairs sharing the same upper level J_up.

    Args:
        df_lines (pd.DataFrame): Dataframe containing line properties (J_up, Branch, Wavelength).
        df_spectrum (pd.DataFrame): Dataframe containing the spectral data (Wavelength, Intensity).
        col_jup (str): Column name for the upper quantum number J in df_lines.
        col_band (str): Column name for the vibrational band in df_lines.
        col_branch (str): Column name for the branch type ('P' or 'R') in df_lines.
        col_wave_line (str): Column name for the theoretical wavelength in df_lines.
        col_name (str): Column name for the line label/name in df_lines.
        col_wave_spec (str): Column name for wavelength in df_spectrum.
        col_int_spec (str): Column name for intensity in df_spectrum.
        window (float): Wavelength window (+/-) around the center to plot.

    Returns:
        tuple: (fig, axes) Matplotlib figure and axes objects. Returns (None, None) if no pairs found.
    """
    # 1. Pair Identification
    pairs = []
    grouped = df_lines.groupby([col_band, col_jup])

    for (band, j_up), group in grouped:
        row_p = group[group[col_branch] == 'P']
        row_r = group[group[col_branch] == 'R']

        if not row_p.empty and not row_r.empty:
            pairs.append({
                'J_up': int(j_up),
                'Band': band,
                'P_data': row_p.iloc[0],
                'R_data': row_r.iloc[0]
            })

    # Sort by Band then J_up
    pairs.sort(key=lambda x: (x['Band'], x['J_up']))
    
    n_rows = len(pairs)
    if n_rows == 0:
        print("No P/R pairs found.")
        return None, None

    # 2. Figure Setup
    fig, axs = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows), sharey=False)
    fig.suptitle(f"Comparison of transitions from same upper level ($J_{{up}}$)", fontsize=16)
    
    # Ensure axs is always a 2D array even for 1 row
    if n_rows == 1:
        axs = np.array([axs])

    # 3. Plotting Loop
    for i, pair in enumerate(pairs):
        for col_idx, branch_key, color, side in [(0, 'P_data', 'darkred', 'P'), (1, 'R_data', 'darkgreen', 'R')]:
            ax = axs[i][col_idx]
            line_data = pair[branch_key]
            center_wav = line_data[col_wave_line]
            
            # Masking data (Standard method)
            mask = (df_spectrum[col_wave_spec] >= center_wav - window) & \
                   (df_spectrum[col_wave_spec] <= center_wav + window)
            spec_chunk = df_spectrum[mask]

            # Plotting
            ax.plot(spec_chunk[col_wave_spec], spec_chunk[col_int_spec], 'k-', lw=1.5)
            ax.axvline(center_wav, color=color, linestyle='--', alpha=0.7, lw=1.5)

            # Styling
            ax.set_title(f"{line_data[col_name]} ($J_{{up}}={pair['J_up']}$)", fontweight='bold', color='darkblue')
            ax.set_xlim([center_wav - window, center_wav + window])
            ax.grid(True, linestyle=':', alpha=0.5)
            
            # Annotation
            ax.text(0.05, 0.85, f"$\lambda_{{theo}} = {center_wav:.4f} \mu m$", 
                    transform=ax.transAxes, fontsize=10, color=color,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            if i == n_rows - 1:
                ax.set_xlabel("Wavelength [$\mu m$]", size=10)
            if col_idx == 0:
                ax.set_ylabel("Intensity [MJy/sr]", size=10)

    plt.tight_layout()
    return fig, axs
""" EXEMPLE D'UTILISATION DANS SPECTRUM.JWST.py
fig, axs = plot_pr_pairs(
    df_lines=df_work,             
    df_spectrum=df_spectrum,      
    col_jup='J_up',              
    col_band='Band',             
    col_branch='Branch',         
    col_wave_line='wavelength',  
    col_name='Line',              
    col_wave_spec='Wavelength',   
    col_int_spec='Int',           
    window=0.05                   
)
"""








def put_annotation(ax, df, color="darkred", col_wave='wavelength', col_name="Line", BaseName='',alpha=0.5,alpha_text=1,style="--", TextInfo=True):
    """
    Ajoute des lignes d'identification et des annotations sans perturber le layout.
    Utilise une transformation mixte (x=data, y=axes) pour ancrer le texte au cadre.
    """
    # Transformation mixte : x en coordonnées données, y en coordonnées axes (0 à 1)
    trans = ax.get_xaxis_transform()
    
    # On itère sur les lignes
    for x_val, label in zip(df[col_wave], df[col_name]):
        # Ligne verticale
        ax.axvline(
            x=x_val,
            color=color,
            linestyle=style,
            linewidth=1,
            alpha=alpha,
            #label='Lignes du DF' if x_val == df[col_wave].iloc[0] else None
        )
        if TextInfo == True : 
        # Texte d'annotation
            ax.text(
                x=x_val,
                y=0.95,          # Position Y fixe relative à la hauteur de l'axe (95% du haut)
                s=BaseName + f'{label}',
                transform=trans, # Applique la transfo mixte
                rotation=90,
                verticalalignment='top',   # Ancrage vertical
                horizontalalignment='right', # Ancrage horizontal
                fontsize=10,
                color=color,
                clip_on=True,     # Empêche le texte de forcer le redimensionnement de la figure
                alpha=alpha_text
            )
        
        




def Excitation_Diagram_V2(Model, Observation, NameList, style, Name=None, color="blue", Reg=False, Save=False, Path="",
                          Annotation=False, Full=False, Level=None, FillIdx=None, threshold=True,
                          Front="Model", Details=True, matchcolor=False, Size=(21,7), coeff_Y=1.2, 
                          thick=2, sizepoint=5, mol="h2", OP_visible=True, ObsCol=None):
    """
    Diagramme d'excitation V2 : Version robuste et optimisée.
    Sépare les niveaux par v, met en évidence les différences Ortho/Para.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.linear_model import LinearRegression

    # --- Constantes Physiques (CGS) ---
    clum = 2.99792458e10
    h = 6.62607015e-27
    
    palette = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9","red","blue","green","black","dimgrey"]
    marqueur = ['^', 'o'] # Pair (0) = Para (^), Impair (1) = Ortho (o)

    # --- 1. Extraction des Observations et Modèles ---
    Raies_brutes = Observation.get_observable_names()
    I_Obs_brut = Observation.get_observable_vector()
    E_list_brut = Observation.get_error_vector()

    scaling_list, I_Model_brut = [], []
    if Model is not None:
        for i in range(len(Model)):
            scaling_list.append(Model[i].get_scaling_factor())
            # Application directe du scaling
            I_Model_brut.append([val  for val in Model[i].get_observable_vector()])#* scaling_list[i] # SCLAING DEJA DANS INTENSITE 
    
    # Création des labels de légende avec paramètres
    NameList2 = []
    if Model is not None:
        for i in range(len(Model)):
            if Details:
                Params = Model[i].get_parameter_vector()
                echelle = Model[i].get_scaling_factor()
                NameList2.append(rf": P$_{{\text{{th}}}}$ = {Params[0]:.2e} K.$cm^{{-3}}$, $G_0$ = {Params[1]:.2e}, $\Omega$={echelle:.1e}")
            else:
                NameList2.append("")

    # --- 2. Lecture Robuste de la Base de Données (Dictionnaire) ---
    main_root = "/Users/aurelien/Desktop/Stage_M2/"
    rootbase = main_root + "PDR/1.7/"
    namebase = f"all_{mol}_lines_info"
    
    if mol == "co_new": 
        rootbase = "/Users/aurelien/Desktop/PhD/Data/"
        namebase = "co_lines_info_HITRAN"
        
    db_lookup = {}
    try:
        with open(rootbase + namebase + ".dat", "r") as doc_base:
            for ligne in doc_base.readlines()[2:]: # On passe le header
                if ligne.strip() and not ligne.strip().startswith('#'):
                    parts = ligne.split(',')
                    nom_raie = parts[0].strip()
                    db_lookup[nom_raie] = parts
    except FileNotFoundError:
        print(f"Erreur : Base de données introuvable à {rootbase}{namebase}.dat")
        return

    # --- 3. Match des Raies et Calculs Physiques ---
    Wave_list, A_list, Eu_list, gu_list = [], [], [], []
    valid_indices = []

    Wave_list, A_list, Eu_list, gu_list = [], [], [], []
    valid_indices = []

    for i, raie in enumerate(Raies_brutes):
        if raie in db_lookup:
            parts = db_lookup[raie]
            try:
                # Si on demande co_new MAIS que la ligne n'a pas 13 colonnes, on se rabat sur l'index 11
                idx_wave = 12 if (mol == "co_new" and len(parts) > 12) else 11
                
                Wave_list.append(float(parts[idx_wave].split('\t')[-1]))
                A_list.append(float(parts[10].split('\t')[-1]))
                Eu_list.append(float(parts[7].split('\t')[-1]))
                gu_list.append(float(parts[5].split('\t')[-1]))
                valid_indices.append(i)
            except Exception as e:
                # J'ajoute len(parts) dans l'erreur pour qu'on sache combien de colonnes il a vraiment lu
                print(f"Erreur de lecture pour {raie} (Colonnes lues: {len(parts)}): {e}")
        else:
            print(f"Avertissement : {raie} absente de la base de données. Ignorée.")
            """
    for i, raie in enumerate(Raies_brutes):
        if raie in db_lookup:
            parts = db_lookup[raie]
            try:
                # Index 12 pour co_new, 11 pour le reste (basé sur l'ancien code)
                idx_wave = 12 if mol == "co_new" else 11
                Wave_list.append(float(parts[idx_wave].split('\t')[-1]))
                A_list.append(float(parts[10].split('\t')[-1]))
                Eu_list.append(float(parts[7].split('\t')[-1]))
                gu_list.append(float(parts[5].split('\t')[-1]))
                valid_indices.append(i)
            except Exception as e:
                print(f"Erreur de lecture des paramètres pour {raie}: {e}")
        else:
            print(f"Avertissement : {raie} absente de la base de données. Ignorée.")
            """
    # Filtrage des vecteurs pour ne garder que les raies valides
    Raies = [Raies_brutes[i] for i in valid_indices]
    I_Obs = [I_Obs_brut[i] for i in valid_indices]
    E_list = [E_list_brut[i] for i in valid_indices]
    I_Model = [[mod_vec[i] for i in valid_indices] for mod_vec in I_Model_brut] if Model else []

    if not Raies:
        print("Aucune raie valide n'a pu être traitée. Arrêt du tracé.")
        return

    # --- Calculs Nu / gu ---
    Nu_Obs, Nu_Err, Y_Obs, Y_Err = [], [], [], []
    for i in range(len(A_list)):
        facteur_conversion = 4 * np.pi * (Wave_list[i] * 1e-4) / (h * clum * A_list[i])
        Nu_Obs.append(facteur_conversion * I_Obs[i])
        
        err_val = (0.3 * I_Obs[i]) if threshold else E_list[i]
        Nu_Err.append(facteur_conversion * err_val)
        
        Y_Obs.append(Nu_Obs[i] / gu_list[i])
        Y_Err.append(Nu_Err[i] / gu_list[i])

    Y_Model = []
    if Model is not None:
        for k in range(len(Model)):
            Y_Model.append([(4 * np.pi * (Wave_list[i] * 1e-4) * I_Model[k][i] / (h * clum * A_list[i])) / gu_list[i] for i in range(len(A_list))])

    # --- 4. Répartition Robuste par Niveau Vibrationnel (v) ---
    # Recherche du v maximum pour dimensionner les listes
    max_v = 0
    for raie in Raies:
        try:
            v_act = int(raie.split("-")[0])
            max_v = max(max_v, v_act)
        except:
            pass
            
    max_v_index = max_v + 1
    
    Lines_Vib = [[] for _ in range(max_v_index)]
    Y_Obs_Vib = [[] for _ in range(max_v_index)]
    Err_Vib = [[] for _ in range(max_v_index)]
    Eu_Vib = [[] for _ in range(max_v_index)]
    Y_Mod_Vib = [[[] for _ in range(max_v_index)] for _ in range(len(Model))] if Model else []

    for i, raie in enumerate(Raies):
        try:
            v_act = int(raie.split("-")[0])
            Lines_Vib[v_act].append(raie)
            Y_Obs_Vib[v_act].append(Y_Obs[i])
            Err_Vib[v_act].append(Y_Err[i])
            Eu_Vib[v_act].append(Eu_list[i])
            if Model:
                for k in range(len(Model)):
                    Y_Mod_Vib[k][v_act].append(Y_Model[k][i])
        except Exception:
            pass

    # --- 5. Tracé (Plotting) ---
    minV = Level[0] if Level else 0
    maxV = Level[1] if Level else max_v_index
    
    order = [1, 2] if Front == "Model" else [2, 1]
    if ObsCol is not None:
        palette = [ObsCol] * 20

    resid_list = []

    # Figure Unique (Full=True)
    if Full:
        fig, ax = plt.subplots(figsize=Size)
        # Création des handles de légende O/P (points invisibles)
        if OP_visible:
            ax.scatter([], [], c='black', marker=marqueur[0], label="Para levels")
            ax.scatter([], [], c='black', marker=marqueur[1], label="Ortho levels")

        if Model:
            for k in range(len(Model)):
                # Tracé d'une ligne globale invisible pour la légende
                ax.plot([], [], linestyle=style[k], color=color[k], alpha=0.9, 
                        label=NameList[k] + NameList2[k], marker="X", linewidth=thick, markersize=sizepoint)

    # Boucle sur les niveaux vibrationnels
    for v_idx in range(minV, maxV):
        if v_idx >= len(Lines_Vib) or not Lines_Vib[v_idx]: 
            continue # On saute les niveaux v qui n'ont aucune raie

        if not Full:
            fig, ax = plt.subplots(figsize=Size)

        # Tracé des observations
        for j in range(len(Lines_Vib[v_idx])):
            # Détection Ortho/Para Robuste
            OP = 0
            if OP_visible:
                try:
                    # Ex: "1-0_R( 10)" -> Extrait "10"
                    j_str = Lines_Vib[v_idx][j].split('(')[1].split(')')[0].strip()
                    OP = int(j_str) % 2
                except:
                    OP = 0
            
            c_obs = palette[v_idx % len(palette)]
            
            # Point et Barre d'erreur principale
            ax.errorbar(Eu_Vib[v_idx][j], Y_Obs_Vib[v_idx][j], yerr=Err_Vib[v_idx][j], c=c_obs,
                        markersize=15, marker=marqueur[OP], markeredgecolor='white',
                        capsize=3, elinewidth=2, ecolor=(0, 0, 0, 0.7), zorder=order[0])
            
            # Barres d'erreur transparentes (design)
            yerr_transp = np.array([[0.5 * Y_Obs_Vib[v_idx][j]], [1.0 * Y_Obs_Vib[v_idx][j]]])
            ax.errorbar(Eu_Vib[v_idx][j], Y_Obs_Vib[v_idx][j], yerr=yerr_transp, c=c_obs,
                        fmt="none", capsize=1, elinewidth=2, ecolor=c_obs, zorder=order[0], alpha=0.2)
        
        # Annotation textuelle du niveau v
        if Lines_Vib[v_idx]:
            ax.annotate(text=f"v={v_idx}", xy=(Eu_Vib[v_idx][0], Y_Obs_Vib[v_idx][0]),
                        xytext=(Eu_Vib[v_idx][0] + 100, Y_Obs_Vib[v_idx][0] * 1.8),
                        color=palette[v_idx % len(palette)], fontweight='bold', fontsize=25)

        # Tracé des modèles
        if Model:
            for k in range(len(Model)):
                idxcolor = v_idx if matchcolor else k
                c_mod = palette[idxcolor % len(palette)] if matchcolor else color[k]
                
                if not Full:
                    ax.plot(Eu_Vib[v_idx], Y_Mod_Vib[k][v_idx], linestyle=style[k], color=c_mod, alpha=0.9,
                            label=NameList[k] + NameList2[k], zorder=order[1], marker="+", linewidth=thick, markersize=sizepoint)
                else:
                    ax.plot(Eu_Vib[v_idx], Y_Mod_Vib[k][v_idx], linestyle=style[k], color=c_mod, alpha=0.9, 
                            zorder=order[1], marker="+", linewidth=thick, markersize=sizepoint)

        # Régression Linéaire
        if Reg and len(Eu_Vib[v_idx]) > 2:
            X = np.array(Eu_Vib[v_idx]).reshape(-1, 1)
            model_reg = LinearRegression()
            model_reg.fit(X, np.log10(Y_Obs_Vib[v_idx]))
            coef = float(model_reg.coef_[0])
            T_ex = -int(1 / coef) if coef != 0 else 0
            
            y_pred = 10 ** model_reg.predict(X)
            residuals = np.log10(np.abs(Y_Obs_Vib[v_idx] - y_pred))
            resid_list.extend(residuals)
            
            ax.plot(X, y_pred, color='red', linestyle=":", label=f"Vup={v_idx}, T={T_ex} K")

        # Mise en forme (Limites, annotations)
        if Y_Obs:
            ymin, ymax = np.min(Y_Obs) / 2, np.max(Y_Obs) * 3
        else:
            ymin, ymax = 1e-5, 1e5

        if Annotation:
            for j in range(len(Eu_Vib[v_idx])):
                ann_ymax = np.max(Y_Obs_Vib[0]) if (Full and Y_Obs_Vib[0]) else ymax
                ax.axvline(x=Eu_Vib[v_idx][j], color='black', linewidth=1, ymin=0.85, ymax=0.9)
                ax.annotate(Lines_Vib[v_idx][j], xy=(Eu_Vib[v_idx][j], ann_ymax * 1.25),
                            xytext=(Eu_Vib[v_idx][j] + 20, ann_ymax * 1.3),
                            fontsize=10, rotation=90)

        ax.grid(linestyle='dotted', alpha=0.7)
        ax.semilogy()
        ax.set_ylim(ymin / coeff_Y, ymax * coeff_Y)
        
        if not Full:
            if Name: plt.annotate(f"{Name} Excitation Vup={v_idx}", xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=20)
            ax.legend(fontsize=20)
            ax.set_ylabel(r'$N_u / g_u$ $[cm^{-2}]$', fontsize=25)
            ax.set_xlabel(r"$E_u$ [K]", fontsize=25)
            plt.tight_layout()
            if Save: plt.savefig(f"{Path}{Name}_excitation_diagram_v{v_idx}.pdf", dpi=200)

    # Finitions du graph Full
    if Full:
        if Name: plt.title(f"{Name} Excitation Diagram all v")
        ax.legend(fontsize=15)
        ax.set_ylabel(r'$N_u / g_u$ $[cm^{-2}]$', fontsize=25)
        ax.set_xlabel(r"$E_u$ [K]", fontsize=25)
        plt.tight_layout()
        if Save: plt.savefig(f"{Path}{Name}_excitation_diagram.pdf", dpi=200)

    # Affichage des résidus
    if Reg and resid_list:
        print("Moyenne, Max, Longueur des résidus (écart obs-LinReg log10) :")
        print(f"Moyenne: {np.mean(resid_list):.3f} | Max: {np.max(resid_list):.3f} | Nb de points: {len(resid_list)}")
    
    plt.show()
    
    
    
    
def Excitation_Diagram_V3(Model, Observation, NameList, style, Name=None, color="blue", Reg=False, Save=False, Path="",
                          ShowLine=False, Full=False, Level=None, FillIdx=None, threshold=True,
                          Front="Model", Details=True, matchcolor=False, Size=(21,7), coeff_Y=1.2, 
                          thick=2, sizepoint=5, mol="h2", OP_visible=True, ObsCol=None, showerr_color=False,PalettePerso = None):
    """
    Diagramme d'excitation V3 : Version robuste et optimisée.
    Sépare les niveaux par v, met en évidence les différences Ortho/Para.
    Gère intelligemment les erreurs.
    ShowLine permet d'afficher les traits à 95% de l'axe Y avec décalage anti-chevauchement.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from sklearn.linear_model import LinearRegression

    # --- Constantes Physiques (CGS) ---
    clum = 2.99792458e10
    h = 6.62607015e-27
    if PalettePerso == None : 
        palette = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9","red","blue","green","black","dimgrey"]
    else : 
        palette = PalettePerso
    marqueur = ['^', 'o'] # Pair (0) = Para (^), Impair (1) = Ortho (o)

    # --- 1. Extraction des Observations et Modèles ---
    Raies_brutes = Observation.get_observable_names()
    I_Obs_brut = Observation.get_observable_vector()
    E_list_brut = Observation.get_error_vector()
    
    try:
        E_type_brut = Observation.get_error_type_vector()
    except AttributeError:
        E_type_brut = ["normal"] * len(Raies_brutes)

    scaling_list, I_Model_brut = [], []
    if Model is not None:
        for i in range(len(Model)):
            scaling_list.append(Model[i].get_scaling_factor())
            I_Model_brut.append([val  for val in Model[i].get_observable_vector()]) #* scaling_list[i] #LE SCALING EST DEJA APPLIQU2 DANS LES INTENSITES DONNEES 
    
    NameList2 = []
    if Model is not None:
        for i in range(len(Model)):
            if Details:
                Params = Model[i].get_parameter_vector()
                echelle = Model[i].get_scaling_factor()
                NameList2.append(rf": P$_{{\text{{th}}}}$ = {Params[0]:.2e} K.$cm^{{-3}}$, $G_0$ = {Params[1]:.2e}, $\Omega$={echelle:.1e}")
            else:
                NameList2.append("")

    # --- 2. Lecture Robuste de la Base de Données ---
    main_root = "/Users/aurelien/Desktop/Stage_M2/"
    rootbase = main_root + "PDR/1.7/"
    namebase = f"all_{mol}_lines_info"
    
    if mol == "co_new": 
        #rootbase = "/Users/aurelien/Desktop/PhD/Data/"
        #namebase = "co_lines_info_HITRAN"
        rootbase = "/Users/aurelien/Desktop/PhD/Results/CO_Update/"
        namebase = "all_co_info_V2"
        
    db_lookup = {}
    try:
        with open(rootbase + namebase + ".dat", "r") as doc_base:
            for ligne in doc_base.readlines()[2:]: 
                if ligne.strip() and not ligne.strip().startswith('#'):
                    parts = ligne.split(',')
                    nom_raie = parts[0].strip()
                    db_lookup[nom_raie] = parts
    except FileNotFoundError:
        print(f"Erreur : Base de données introuvable à {rootbase}{namebase}.dat")
        return

    # --- 3. Match des Raies et Calculs Physiques ---
    Wave_list, A_list, Eu_list, gu_list = [], [], [], []
    valid_indices = []

    for i, raie in enumerate(Raies_brutes):
        if raie in db_lookup:
            parts = db_lookup[raie]
            try:
                idx_wave = 12 if (mol == "co_new" and len(parts) > 12) else 11
                Wave_list.append(float(parts[idx_wave].split('\t')[-1]))
                A_list.append(float(parts[10].split('\t')[-1]))
                Eu_list.append(float(parts[7].split('\t')[-1]))
                gu_list.append(float(parts[5].split('\t')[-1]))
                valid_indices.append(i)
            except Exception as e:
                print(f"Erreur de lecture pour {raie} (Colonnes lues: {len(parts)}): {e}")
        else:
            print(f"Avertissement : {raie} absente de la base de données. Ignorée.")
            
    Raies = [Raies_brutes[i] for i in valid_indices]
    I_Obs = [I_Obs_brut[i] for i in valid_indices]
    E_list = [E_list_brut[i] for i in valid_indices]
    E_type = [E_type_brut[i] for i in valid_indices]
    I_Model = [[mod_vec[i] for i in valid_indices] for mod_vec in I_Model_brut] if Model else []

    if not Raies:
        print("Aucune raie valide n'a pu être traitée. Arrêt du tracé.")
        return

    Nu_Obs, Nu_Err, Y_Obs, Y_Err = [], [], [], []
    is_30_percent_err = [] 
    
    for i in range(len(A_list)):
        facteur_conversion = 4 * np.pi * (Wave_list[i] * 1e-4) / (h * clum * A_list[i])
        Nu_Obs.append(facteur_conversion * I_Obs[i])
        
        if threshold:
            err_val = 0.3 * I_Obs[i]
            is_30_percent = True
        else:
            if E_type[i].lower() == 'lognormal' or E_list[i] > I_Obs[i]: 
                 err_val = 0.3 * I_Obs[i]
                 is_30_percent = True
            else:
                 err_val = E_list[i]
                 is_30_percent = False
                 
        Nu_Err.append(facteur_conversion * err_val)
        is_30_percent_err.append(is_30_percent)
        Y_Obs.append(Nu_Obs[i] / gu_list[i])
        Y_Err.append(Nu_Err[i] / gu_list[i])

    Y_Model = []
    if Model is not None:
        for k in range(len(Model)):
            Y_Model.append([(4 * np.pi * (Wave_list[i] * 1e-4) * I_Model[k][i] / (h * clum * A_list[i])) / gu_list[i] for i in range(len(A_list))])

    # --- 4. Répartition Robuste par Niveau Vibrationnel ---
    max_v = 0
    for raie in Raies:
        try:
            v_act = int(raie.split("-")[0])
            max_v = max(max_v, v_act)
        except:
            pass
            
    max_v_index = max_v + 1
    Lines_Vib = [[] for _ in range(max_v_index)]
    Y_Obs_Vib = [[] for _ in range(max_v_index)]
    Err_Vib = [[] for _ in range(max_v_index)]
    Eu_Vib = [[] for _ in range(max_v_index)]
    Err_Type_Vib = [[] for _ in range(max_v_index)]
    Y_Mod_Vib = [[[] for _ in range(max_v_index)] for _ in range(len(Model))] if Model else []

    for i, raie in enumerate(Raies):
        try:
            v_act = int(raie.split("-")[0])
            Lines_Vib[v_act].append(raie)
            Y_Obs_Vib[v_act].append(Y_Obs[i])
            Err_Vib[v_act].append(Y_Err[i])
            Eu_Vib[v_act].append(Eu_list[i])
            Err_Type_Vib[v_act].append(is_30_percent_err[i])
            if Model:
                for k in range(len(Model)):
                    Y_Mod_Vib[k][v_act].append(Y_Model[k][i])
        except Exception:
            pass

    # --- 5. Tracé ---
    minV = Level[0] if Level else 0
    maxV = Level[1] if Level else max_v_index
    order = [1, 2] if Front == "Model" else [2, 1]
    if ObsCol is not None:
        palette = [ObsCol] * 20

    resid_list = []

    if Full:
        fig, ax = plt.subplots(figsize=Size)
        if OP_visible:
            ax.scatter([], [], c='black', marker=marqueur[0], label="Para levels")
            ax.scatter([], [], c='black', marker=marqueur[1], label="Ortho levels")
            
        if showerr_color and not threshold:
            ax.plot([], [], color='red', marker='|', linestyle='None')
            ax.plot([], [], color='black', marker='|', linestyle='None')

        if Model:
            for k in range(len(Model)):
                ax.plot([], [], linestyle=style[k], color=color[k], alpha=0.9, 
                        label=NameList[k] + NameList2[k], marker="X", linewidth=thick, markersize=sizepoint)

    for v_idx in range(minV, maxV):
        if v_idx >= len(Lines_Vib) or not Lines_Vib[v_idx]: 
            continue

        if not Full:
            fig, ax = plt.subplots(figsize=Size)

        for j in range(len(Lines_Vib[v_idx])):
            OP = 0
            if OP_visible:
                try:
                    j_str = Lines_Vib[v_idx][j].split('(')[1].split(')')[0].strip()
                    OP = int(j_str) % 2
                except:
                    OP = 0
            
            c_obs = palette[v_idx % len(palette)]
            e_color = 'red' if (showerr_color and not threshold and Err_Type_Vib[v_idx][j]) else (0, 0, 0, 0.7)
            
            ax.errorbar(Eu_Vib[v_idx][j], Y_Obs_Vib[v_idx][j], yerr=Err_Vib[v_idx][j], c=c_obs,
                        markersize=15, marker=marqueur[OP], markeredgecolor='white',
                        capsize=3, elinewidth=2, ecolor=e_color, zorder=order[0])
            
            yerr_transp = np.array([[0.5 * Y_Obs_Vib[v_idx][j]], [1.0 * Y_Obs_Vib[v_idx][j]]])
            ax.errorbar(Eu_Vib[v_idx][j], Y_Obs_Vib[v_idx][j], yerr=yerr_transp, c=c_obs,
                        fmt="none", capsize=1, elinewidth=2, ecolor=c_obs, zorder=order[0], alpha=0.2)
        
        if Lines_Vib[v_idx]:
            ax.annotate(text=f"v={v_idx}", xy=(Eu_Vib[v_idx][0], Y_Obs_Vib[v_idx][0]),
                        xytext=(Eu_Vib[v_idx][0] + 100, Y_Obs_Vib[v_idx][0] * 1.8),
                        color=palette[v_idx % len(palette)], fontweight='bold', fontsize=25)

        if Model:
            for k in range(len(Model)):
                idxcolor = v_idx if matchcolor else k
                c_mod = palette[idxcolor % len(palette)] if matchcolor else color[k]
                
                if not Full:
                    ax.plot(Eu_Vib[v_idx], Y_Mod_Vib[k][v_idx], linestyle=style[k], color=c_mod, alpha=0.9,
                            label=NameList[k] + NameList2[k], zorder=order[1], marker="+", linewidth=thick, markersize=sizepoint)
                else:
                    ax.plot(Eu_Vib[v_idx], Y_Mod_Vib[k][v_idx], linestyle=style[k], color=c_mod, alpha=0.9, 
                            zorder=order[1], marker="+", linewidth=thick, markersize=sizepoint)

        if Reg and len(Eu_Vib[v_idx]) > 2:
            X = np.array(Eu_Vib[v_idx]).reshape(-1, 1)
            model_reg = LinearRegression()
            model_reg.fit(X, np.log10(Y_Obs_Vib[v_idx])) # On garde log10 pour le graphique
            coef = float(model_reg.coef_[0])
            
            # --- LA CORRECTION EST ICI ---
            # On multiplie le coef par ln(10) pour convertir la pente de la base 10 à la base e
            T_ex = -1.0 / (coef * np.log(10)) if coef != 0 else 0
            
            y_pred = 10 ** model_reg.predict(X)
            residuals = np.log10(np.abs(Y_Obs_Vib[v_idx] - y_pred))
            resid_list.extend(residuals)
            
            # On affiche avec un arrondi propre (ex: :.1f ou :.0f) au lieu de int()
            ax.plot(X, y_pred, color='red', linestyle=":", label=f"Vup={v_idx}, T={T_ex:.1f} K")

        # --- NOUVEAU BLOC: ShowLine avec anti-chevauchement ---
        if ShowLine:
            eu_counts = {}
            for j in range(len(Eu_Vib[v_idx])):
                # Arrondi pour grouper les raies ayant presque le même Eu
                eu_val = round(Eu_Vib[v_idx][j], 1)
                count = eu_counts.get(eu_val, 0)
                
                # Le décalage vertical s'incrémente de 8% pour chaque raie superposée
                y_text_pos = 0.96 + (0.08 * count) 
                
                # Dessine le trait de 90% à 95% de l'axe (indépendant de l'échelle logarithmique)
                ax.plot([Eu_Vib[v_idx][j], Eu_Vib[v_idx][j]], [0.90, 0.95], 
                        transform=ax.get_xaxis_transform(), color='black', linewidth=1, alpha=0.5)
                
                # Écrit le nom de la raie
                ax.text(Eu_Vib[v_idx][j], y_text_pos, Lines_Vib[v_idx][j], 
                        transform=ax.get_xaxis_transform(), rotation=90, 
                        va='bottom', ha='center', fontsize=12, color=palette[v_idx % len(palette)],
                        clip_on=False) # clip_on=False empêche le texte d'être coupé s'il sort du cadre
                
                eu_counts[eu_val] = count + 1
        # --------------------------------------------------------

        if Y_Obs:
            ymin, ymax = np.min(Y_Obs) / 2, np.max(Y_Obs) * 3
        else:
            ymin, ymax = 1e-5, 1e5

        #ax.grid(linestyle='dotted', alpha=0.7)
        ax.semilogy()
        ax.set_ylim(ymin / coeff_Y, ymax * coeff_Y)
        
        if not Full:
            if Name: plt.annotate(f"{Name} Excitation Vup={v_idx}", xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=20)
            
            if showerr_color and not threshold:
                handles, labels = ax.get_legend_handles_labels()
                import matplotlib.lines as mlines
                assumed_err_line = mlines.Line2D([], [], color='red', marker='|', linestyle='None', markersize=10, label='30% Assumption Error')
                obs_err_line = mlines.Line2D([], [], color='black', marker='|', linestyle='None', markersize=10, label='Observational Error')
                handles.extend([assumed_err_line, obs_err_line])
                ax.legend(handles=handles, fontsize=20)
            else:
                ax.legend(fontsize=20)
                
            ax.set_ylabel(r'$N_u / g_u$ $[cm^{-2}]$', fontsize=25)
            ax.set_xlabel(r"$E_u$ [K]", fontsize=25)
            
            # Ajustement pour laisser de la place au texte en haut
            if ShowLine:
                plt.subplots_adjust(top=0.85)
            else:
                plt.tight_layout()
                
            if Save: plt.savefig(f"{Path}{Name}_excitation_diagram_v{v_idx}.pdf", dpi=200)

    if Full:
        if Name: plt.title(f"{Name} Excitation Diagram all v")
        ax.legend(fontsize=15)
        ax.set_ylabel(r'$N_u / g_u$ $[cm^{-2}]$', fontsize=25)
        ax.set_xlabel(r"$E_u$ [K]", fontsize=25)
        
        if ShowLine:
            plt.subplots_adjust(top=0.85)
        else:
            plt.tight_layout()
            
        if Save: plt.savefig(f"{Path}{Name}_excitation_diagram.pdf", dpi=200)

    if Reg and resid_list:
        print("Moyenne, Max, Longueur des résidus (écart obs-LinReg log10) :")
        print(f"Moyenne: {np.mean(resid_list):.3f} | Max: {np.max(resid_list):.3f} | Nb de points: {len(resid_list)}")
    
    plt.show()
    
def Excitation_Diagram_V4(Model, Observation, NameList, style, Name=None, color="blue", Reg=False, Save=False, Path="",
                          ShowLine=False, Full=False, Level=None, threshold=True,
                          Front="Model", Details=True, matchcolor=False, Size=(21,7), coeff_Y=1.2, 
                          thick=2, sizepoint=5, mol="co_new", OP_visible=True, ObsCol=None, showerr_color=False, PalettePerso=None):
    """
    Diagramme d'excitation V4 : Standard Scientifique & Analyse de Sensibilité.
    - Labels horizontaux anti-collision sur les points.
    - Gestion intelligente des erreurs (Normal, Lognormal, 30% threshold).
    - Tri physique par Eu et répartition par v_up.
    - Correction intégrée : Factorisation de la conversion Nu/gu pour éviter les IndexError.
    - Correction scaling : Le get_observable_vector() contient déjà le scaling factor.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import re
    from sklearn.linear_model import LinearRegression

    # --- Constantes Physiques (CGS) ---
    clum, h = 2.99792458e10, 6.62607015e-27
    if PalettePerso == None : 
        palette = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9","red","blue","green","black","dimgrey"]
    else : 
        palette = PalettePerso
    marqueur = ['^', 'o'] if OP_visible else ['o', 'o']

    # --- 1. Extraction des données ---
    Raies_brutes = Observation.get_observable_names()
    I_Obs_brut = Observation.get_observable_vector()
    E_list_brut = Observation.get_error_vector()
    
    try:
        E_type_brut = Observation.get_error_type_vector()
    except:
        E_type_brut = ["normal"] * len(Raies_brutes)

    I_Model_brut = []
    NameList2 = []
    if Model is not None:
        for i in range(len(Model)):
            # On ne re-multiplie PAS par le scaling factor ici !
            I_Model_brut.append([val for val in Model[i].get_observable_vector()]) 
            
            # Récupération des paramètres pour la légende
            if Details:
                Params = Model[i].get_parameter_vector()
                echelle = Model[i].get_scaling_factor()
                NameList2.append(rf" : P$_{{\text{{th}}}}$ = {Params[0]:.2e} K.$cm^{{-3}}$, $G_0$ = {Params[1]:.2e}, $\Omega$={echelle:.1e}")
            else:
                NameList2.append("")

    # --- 2. Lecture Base de Données ---
    # --- 2. Lecture Robuste de la Base de Données ---
    main_root = "/Users/aurelien/Desktop/Stage_M2/"
    rootbase = main_root + "PDR/1.7/"
    namebase = f"all_{mol}_lines_info"
    
    if mol == "co_new": 
        #rootbase = "/Users/aurelien/Desktop/PhD/Data/"
        #namebase = "co_lines_info_HITRAN"
        rootbase = "/Users/aurelien/Desktop/PhD/Results/CO_Update/"
        namebase = "all_co_info_V2"
        
    db_lookup = {}
    with open(rootbase + namebase + ".dat", "r") as f:
        for ligne in f.readlines()[2:]:
            if ligne.strip() and not ligne.strip().startswith('#'):
                parts = ligne.split(',')
                db_lookup[parts[0].strip()] = parts

    # --- 3. Calculs Physiques ---
    data_points = []
    for i, raie in enumerate(Raies_brutes):
        if raie in db_lookup:
            p = db_lookup[raie]
            # Sécurité robuste pour l'index de la longueur d'onde
            idx_w = 12 if (mol == "co_new" and len(p) > 12) else 11
            wave, Aul, Eu, gu = float(p[idx_w]), float(p[10]), float(p[7]), float(p[5])
            
            # Conversion Intensité -> Nu/gu
            fact = 4 * np.pi * (wave * 1e-4) / (h * clum * Aul)
            y_val = (fact * I_Obs_brut[i]) / gu
            
            # Stockage du facteur de conversion (évite de le recalculer et prévient l'IndexError)
            conv_factor = fact / gu
            
            # Gestion Erreur
            is_assumed = False
            if threshold or E_type_brut[i] == 'lognormal' or E_list_brut[i] > I_Obs_brut[i]:
                err_y = 0.3 * y_val
                is_assumed = True
            else:
                err_y = (fact * E_list_brut[i]) / gu
            
            # Extraction v_up
            v_up = int(raie.split("-")[0]) if "-" in raie else 0
            
            data_points.append({
                'name': raie, 'v': v_up, 'Eu': Eu, 'y': y_val, 'err': err_y, 
                'assumed': is_assumed, 'idx': i, 'conv_factor': conv_factor
            })

    if not data_points: 
        print("Erreur: Aucune raie valide trouvée en base de données.")
        return
    
    # --- 4. Tracé ---
    fig, ax = plt.subplots(figsize=Size)
    ax.set_yscale('log')
    
    # Tri par Eu pour le tracé fluide des lignes de modèle
    data_points.sort(key=lambda x: x['Eu'])
    
    # Gestion des niveaux v demandés
    v_list = sorted(list(set(d['v'] for d in data_points)))
    if Level: 
        v_list = [v for v in v_list if Level[0] <= v <= Level[1]]

    pos_tracker = [] # Pour éviter l'overlap des noms de raies

    # --- Handles invisibles pour une légende propre ---
    if OP_visible:
        ax.scatter([], [], c='black', marker=marqueur[0], label="Para levels")
        ax.scatter([], [], c='black', marker=marqueur[1], label="Ortho levels")

    if Model is not None:
        for k in range(len(Model)):
            ax.plot([], [], linestyle=style[k], color=color[k], alpha=0.8, 
                    label=NameList[k] + NameList2[k], linewidth=thick)
    
    # --- Tracé des points ---
    for v in v_list:
        pts = [d for d in data_points if d['v'] == v]
        c = palette[v % len(palette)]
        
        for p in pts:
            # Ortho/Para
            op = int(re.search(r'\( *(\d+)\)', p['name']).group(1)) % 2 if '(' in p['name'] else 0
            
            # Erreur couleur
            ecol = 'red' if (showerr_color and p['assumed']) else (0,0,0,0.6)
            
            ax.errorbar(p['Eu'], p['y'], yerr=p['err'], fmt='none', ecolor=ecol, capsize=3, zorder=2)
            ax.scatter(p['Eu'], p['y'], marker=marqueur[op], s=sizepoint*20, c=c, edgecolors='white', zorder=3)

            # --- Annotation des raies (Smart ShowLine) ---
            if ShowLine:
                off_x, off_y = 5, 0
                for (px, py) in pos_tracker:
                    if abs(px - p['Eu']) < 50 and abs(np.log10(py) - np.log10(p['y'])) < 0.15:
                        off_y -= 15 # Décalage vertical si trop proche
                
                ax.annotate(p['name'], xy=(p['Eu'], p['y']), xytext=(off_x, off_y), 
                            textcoords="offset points", va='center', fontsize=10, color=c,
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
                pos_tracker.append((p['Eu'], p['y']))

        # Modèles
        if Model:
            for k in range(len(Model)):
                # Calcul direct et propre grâce au conv_factor stocké
                y_mod = [I_Model_brut[k][p['idx']] * p['conv_factor'] for p in pts]
                ax.plot([p['Eu'] for p in pts], y_mod, style[k], color=c if matchcolor else color[k], alpha=0.8, linewidth=thick)

    # --- 5. Fix des limites de l'axe Y ---
    y_vals = [p['y'] for p in data_points]
    if y_vals:
        ymin, ymax = np.min(y_vals), np.max(y_vals)
        ax.set_ylim(ymin / (coeff_Y * 2), ymax * (coeff_Y * 2))

    # --- Cosmétique ---
    ax.set_xlabel(r"$E_u$ [K]", fontsize=20)
    ax.set_ylabel(r"$N_u / g_u$ [cm$^{-2}$]", fontsize=20)
    # ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=12, loc='best')
    if Name: ax.set_title(f"Diagramme d'excitation - {Name}", fontsize=22)
    
    plt.tight_layout()
    if Save: plt.savefig(f"{Path}{Name}_Excitation.pdf")
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def Excitation_Diagram_V5(Obsv1, Obsv2, Model=None, style=None, color=None, matchcolor=False, 
                          coeff_Y=1.5, thick=1.5, plot_texcit=True, mol="co_new"):
    """
    Excitation Diagram V5 - Clean Indentation, Uniform CO Markers & Split Box Layout.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import re
    from sklearn.linear_model import LinearRegression

    clum, h = 2.99792458e10, 6.62607015e-27
    
    # Taille forcée à (18, 10) à la demande de l'utilisateur
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Lecture de la base de données spectroscopique
    rootbase = "/Users/aurelien/Desktop/PhD/Data/" if mol == "co_new" else "/Users/aurelien/Desktop/Stage_M2/PDR/1.7/"
    namebase = "co_lines_info_HITRAN" if mol == "co_new" else f"all_{mol}_lines_info"
    
    db_lookup = {}
    with open(rootbase + namebase + ".dat", "r") as f:
        for ligne in f.readlines()[2:]:
            if ligne.strip() and not ligne.strip().startswith('#'):
                parts = ligne.split(',')
                db_lookup[parts[0].strip()] = parts

    v_groups = {1: [], 2: []}
    
    # Reconstruction des points physiques
    for vu, obs_obj in [(1, Obsv1), (2, Obsv2)]:
        if obs_obj is None: 
            continue
        names = obs_obj.get_observable_names()
        flux = obs_obj.get_observable_vector()
        
        for idx, raie in enumerate(names):
            if raie in db_lookup:
                p = db_lookup[raie]
                idx_w = 12 if (mol == "co_new" and len(p) > 12) else 11
                wave, Aul, Eu, gu = float(p[idx_w]), float(p[10]), float(p[7]), float(p[5])
                
                fact = 4 * np.pi * (wave * 1e-4) / (h * clum * Aul)
                y_val = (fact * flux[idx]) / gu
                
                ju_match = re.search(r'\( *(\d+)\)', raie)
                ju = int(ju_match.group(1)) if ju_match else 0
                
                v_groups[vu].append({'Eu': Eu, 'y': y_val, 'Ju': ju, 'idx': idx, 'conv_factor': fact / gu})

    v_colors = {1: '#1f77b4', 2: '#ff7f0e'}
    
    # Tracé des points et calcul de Trot
    for vu in [1, 2]:
        pts = v_groups[vu]
        if not pts: 
            continue
        c = color[vu] if (color and vu < len(color)) else v_colors.get(vu, 'black')
        
        xu = [p['Eu'] for p in pts]
        yu = [p['y'] for p in pts]
        
        sort_idx = np.argsort(xu)
        xu, yu = np.array(xu)[sort_idx], np.array(yu)[sort_idx]
        
        # Marqueur unique 'o' pour le CO (Pas de tri Ortho/Para à 0)
        for p in pts:
            ax.scatter(p['Eu'], p['y'], marker='o', s=120, c=c, edgecolors='white', zorder=5)
            
        if plot_texcit and len(pts) > 1:
            X_fit = xu.reshape(-1, 1)
            Y_fit = np.log(yu)
            reg = LinearRegression().fit(X_fit, Y_fit)
            slope = reg.coef_[0]
            if slope < 0:
                trot = -1.0 / slope
                ax.plot(xu, np.exp(reg.predict(X_fit)), color=c, linestyle='--', alpha=0.7, 
                        label=f'v={vu} Fit (Trot = {abs(trot):.0f} K)')

    # Calcul de Tvib entre v=1 et v=2 pour le même Ju
    tvib_lines = []
    if plot_texcit and v_groups[1] and v_groups[2]:
        v1_dict = {p['Ju']: p for p in v_groups[1]}
        v2_dict = {p['Ju']: p for p in v_groups[2]}
        common_Ju = set(v1_dict.keys()).intersection(v2_dict.keys())
        
        for ju in sorted(common_Ju):
            p1, p2 = v1_dict[ju], v2_dict[ju]
            dE = p2['Eu'] - p1['Eu']
            dlnN = np.log(p1['y']) - np.log(p2['y'])
            if dlnN > 0:
                tvib = dE / dlnN
                tvib_lines.append(f"Tvib (Ju={ju}) = {tvib:.0f} K")
                ax.plot([p1['Eu'], p2['Eu']], [p1['y'], p2['y']], color='gray', linestyle=':', alpha=0.3)

    # Cosmétique de l'affichage scientifique
    ax.set_yscale('log')
    ax.set_xlabel('Upper Level Energy Eu (K)', fontsize=15)
    ax.set_ylabel('Nu / gu (cm-2)', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.legend(loc='upper right', fontsize=12)
    
    # BOÎTE 1 : Équations fondamentales agrandies (Bas gauche)
    eq_text = (
        r"$\mathbf{Physics\ Equations:}$" "\n\n"
        r"$\ln(N_u/g_u) = -\frac{E_u}{T_{\mathrm{rot}}} + \mathrm{Cst}$" "\n\n"
        r"$T_{\mathrm{vib}} = \frac{E_{u,2} - E_{u,1}}{\ln(N_u/g_u)_1 - \ln(N_u/g_u)_2}$"
    )
    ax.text(0.02, 0.04, eq_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="gray", alpha=0.9))
    
    # BOÎTE 2 : Résultats numériques Tvib séparés (Juste au-dessus ou décalés)
    if tvib_lines:
        results_text = r"$\mathbf{Calculated\ T_{\mathrm{vib}}:}$" + "\n" + "\n".join(tvib_lines)
        ax.text(0.32, 0.04, results_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.5", fc="#f9f9f9", ec="gray", alpha=0.9))

    plt.tight_layout()
    return fig, ax