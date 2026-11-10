#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:27:28 2025

@author: aurelien
"""

import numpy as np
from tqdm import tqdm
import pyismtools as ism
import math 
import pandas as pd 
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings('ignore')

import sys
chemin_modules = "/Users/aurelien/Desktop/PhD/Codes/Package/Perso/"
sys.path.append(chemin_modules)
import read as read 
import viz as viz


# =============================================================================
#                             General tools 
# =============================================================================

def get_scaling_simple(liste1, liste2):
    #Donne le meilleur facteur d'échelle entre les deux listes en miniisant la diff simplement
    liste1 = np.array(liste1)
    liste2 = np.array(liste2)
    # Calcul du facteur d'échelle optimal
    facteur_echelle = np.sum(liste1 * liste2) / np.sum(liste1**2)
    return facteur_echelle


def dilution (R,d):
    """Retourne la dilution en extinction géométrique"""
    return ((R/d)**2)   # Facteur de dilution en extinciton géométrique 



# =============================================================================
#                       Processing data pour PyISM Tools
# =============================================================================


def Processing_to_ISM(name,obs_path,Obs_names,model_path,Grid_names,Grid_surnames,Model_names=None,Model_surnames=None,
                      threshold=0.3,limscale_low = 1e-2,limscale=10,Process_OP=False,incertitude=True,mol="h2") :
    
    assert len(Grid_names)==len(Grid_surnames)
    
    Obs_Data=[]
    for el in Obs_names : #Processing Observations
        Obs_Data.append(ism.Observation.from_ASCII_file(obs_path+el,el))  
        if incertitude==True : 
            Obs_Data[-1].set_multiplicative_noise_threshold(threshold=threshold);print(f"{threshold*100}% uncertainties added")       

    
    Grid_Data=[]       #Processing Grids
    for el in Grid_names:
        Grid_Data.append(ism.Model_grid.from_ASCII_file(model_path+el+".dat",3,el))
    
    if Model_names!= None : #Processing Solo Models
        Model_Data=[]
        for el in Model_names : 
            Model_Data.append(ism.Model_grid.from_ASCII_file(model_path+el+".dat",3,el))
            #-------------------------------------------------------------------
    for i in range (len(Obs_Data)) : #Translation of nomenclature between models and observations
        Line_Names = list(Obs_Data[i].get_observable_names())
        new_name_dict = {};dict_orto={};dict_para={}
        for x in Line_Names:
            vup,temp= x.split('-')
            vlow,temp2=temp.split('_')
            trtype,temp3=temp2.split('(')
            Jlow=temp3.split(')')[0].split(' ')[1]
            if trtype == 'S' :
                Jup = str(int(Jlow)+2)
            if trtype == 'R' :
                Jup = str(int(Jlow)+1)
            elif trtype == 'P' :
                Jup = str(int(Jlow)-1)
            elif trtype == 'O' :
                Jup = str(int(Jlow)-2)
            elif trtype == 'Q' :
                Jup = Jlow
            new_name_dict[x] = f"inta_00_{mol}_v{vup}_j{str(Jup)}__v{vlow}_j{Jlow}"
            if int(Jlow)%2==0 : 
                dict_para[x]=f"inta_00_{mol}_v{vup}_j{str(Jup)}__v{vlow}_j{Jlow}"
            elif int(Jlow)%2==1 : 
                dict_orto[x]=f"inta_00_{mol}_v{vup}_j{str(Jup)}__v{vlow}_j{Jlow}"
            
        name_dictionnaire=f"Traduction_{name}"
        my_translation = ism.Translation(name_dictionnaire,new_name_dict)
        quantity_names = my_translation.get_alternate_names()
        
        trans_o = ism.Translation(name_dictionnaire,dict_orto)
        trans_p = ism.Translation(name_dictionnaire,dict_para)
        names_o,names_p=trans_o.get_alternate_names(),trans_p.get_alternate_names()
        
        Grid_Dic={}

        if len(Obs_Data)==1 : #Si que 1 observation
        
            for i in range (len(Grid_Data)) : #For grids of model
            
                grid_o,grid_p=Grid_Data[i].copy(),Grid_Data[i].copy()#partie si on sépare les niveaux ortho et para
                grid_o.reduce_observables( names_o);grid_o.translate_names(trans_o)
                grid_p.reduce_observables(names_p);grid_p.translate_names(trans_p)
                
                Grid_Data[i].reduce_observables(quantity_names)#pour les grilles complètes
                Grid_Data[i].translate_names(my_translation)
                
                Grid_Dic[Grid_surnames[i]]=Grid_Data[i]
                if Process_OP==True :
                    Grid_Dic[Grid_surnames[i]+"_O"]=grid_o
                    Grid_Dic[Grid_surnames[i]+"_P"]=grid_p
                
            if Model_names!= None : 
                for el in Model_Data :  #for solo models
                    el.reduce_observables(quantity_names)
                    el.translate_names(my_translation)                
                
        elif len(Obs_Data)>1 : # Si plusieurs observations, alors plusieurs traductions
            print("Attention, il est recommandé de faire un Processing par jeu d'observation et les comparer ensuite")
            
    #--------------------------------------------------------------------------
    Approx_Dic={}
    BestMod_Dic={}
    BestChi_Dic={}
    Obs_Dic={}
    Obs_Dic[Obs_names[0]]=Obs_Data[0]
    for i in range (len(Grid_Data)):
        # Création des interpolation linéaire et BestFit
        approximator=ism.Grid_approximator.interpolate_from_grid(Grid_Data[i])
        best_model,best_chi2=ism.fit_tools.find_best_interpolated_model(Obs_Data[0],
                                                                        approximator,N_repetitions=20,
                                                                        scaling_factor_bounds=[limscale_low,limscale],
                                                                        cost_function="reduced_chi2")
        print(f"Grid named {Grid_surnames[i]} is processed")
        Approx_Dic[Grid_surnames[i]]=approximator
        BestMod_Dic[Grid_surnames[i]]=best_model
        BestChi_Dic[Grid_surnames[i]]=best_chi2   
        
        if Process_OP==True : #Si on veut séparer les ortho et para et comparer séparément
            approximator_ortho=ism.Grid_approximator.interpolate_from_grid(grid_o)
            approximator_para=ism.Grid_approximator.interpolate_from_grid(grid_p)
            obs_o, obs_p=Obs_Data[0].copy(),Obs_Data[0].copy()
            obs_o.reduce_observables(trans_o.get_names());obs_p.reduce_observables(trans_p.get_names())

            best_model_O, best_chi2_O = ism.fit_tools.find_best_interpolated_model(obs_o,
                                                        approximator_ortho,N_repetitions=20,scaling_factor_bounds=[0,limscale],cost_function="reduced_chi2")
            best_model_P, best_chi2_P = ism.fit_tools.find_best_interpolated_model(obs_p,
                                                        approximator_para,N_repetitions=20,scaling_factor_bounds=[0,limscale],cost_function="reduced_chi2")
            
            Approx_Dic[Grid_surnames[i]+"_O"]=approximator_ortho
            Approx_Dic[Grid_surnames[i]+"_P"]=approximator_para
            BestMod_Dic[Grid_surnames[i]+"_O"]=best_model_O
            BestMod_Dic[Grid_surnames[i]+"_P"]=best_model_P
            BestChi_Dic[Grid_surnames[i]+"_O"]=best_chi2_O 
            BestChi_Dic[Grid_surnames[i]+"_P"]=best_chi2_P 
            
            Obs_Dic[Grid_surnames[i]+"_O"]=obs_o 
            Obs_Dic[Grid_surnames[i]+"_P"]=obs_p 

            print(f"Grid named {Grid_surnames[i]}"+"_O and {Grid_surnames[i]}_P are processed")
    Model_Dic={}        
    if Model_names!= None : 
        for i in range (len(Model_Data)):
            # Création des interpolation liénaires et BestFit
            best_model,best_chi2=ism.fit_tools.find_best_model_in_grid(Obs_Data[0],Model_Data[i],
                                                                            scaling_factor_bounds=[0,limscale])        
            print(f"Model named {Model_surnames[i]} is processed")
            Model_Dic[Model_surnames[i]]=best_model
    return(Obs_Data,Approx_Dic,BestMod_Dic,BestChi_Dic,Model_Dic,Obs_Dic,Grid_Dic)



def Processing_to_ISM_V2(name,obs_path,Obs_names,model_path,Grid_names,Grid_surnames,Model_names=None,Model_surnames=None,
                      threshold=0.3,limscale=10,Process_OP=False,incertitude=True,mol="h2") :
    
    assert len(Grid_names)==len(Grid_surnames)
    
    Obs_Data=[]
    for el in Obs_names : #Processing Observations
        Obs_Data.append(ism.Observation.from_ASCII_file(obs_path+el,el))  
        if incertitude==True : 
            Obs_Data[-1].set_multiplicative_noise_threshold(threshold=threshold);print(f"{threshold*100}% uncertainties added")       

    Grid_Data=[]       #Processing Grids
    for el in Grid_names:
        Grid_Data.append(ism.Model_grid.from_ASCII_file(model_path+el+".dat",3,el))
    
    Model_Data=[]
    if Model_names!= None : #Processing Solo Models
        for el in Model_names : 
            Model_Data.append(ism.Model_grid.from_ASCII_file(model_path+el+".dat",3,el))
            
    # --- SOUS-FONCTION DE TRADUCTION ---
    def get_theo_name(raie_str, molecule):
        vup, temp = raie_str.split('-')
        vlow, temp2 = temp.split('_')
        trtype, temp3 = temp2.split('(')
        Jlow = temp3.split(')')[0].split(' ')[1]
        if trtype == 'S' : Jup = str(int(Jlow)+2)
        elif trtype == 'R' : Jup = str(int(Jlow)+1)
        elif trtype == 'P' : Jup = str(int(Jlow)-1)
        elif trtype == 'O' : Jup = str(int(Jlow)-2)
        elif trtype == 'Q' : Jup = Jlow
        return f"inta_00_{molecule}_v{vup}_j{str(Jup)}__v{vlow}_j{Jlow}", int(Jlow)

    #-------------------------------------------------------------------
    for i in range (len(Obs_Data)) : #Translation of nomenclature between models and observations
        Line_Names = list(Obs_Data[i].get_observable_names())
        new_name_dict = {};dict_orto={};dict_para={}
        blends_to_add = {} # Dictionnaire pour stocker les sommes à créer
        
        for x in Line_Names:
            # 1. GESTION DES BLENDS (Les "Raie1+Raie2")
            if "+" in x:
                raie1, raie2 = x.split('+')
                theo1, Jlow1 = get_theo_name(raie1, mol)
                theo2, Jlow2 = get_theo_name(raie2, mol)
                
                theo_sum = f"SUM_{x}" # Nom temporaire pour la grille
                new_name_dict[x] = theo_sum
                blends_to_add[theo_sum] = (theo1, theo2)
                # On ignore l'ortho/para pour les blends CO pour ne pas complexifier
                
            # 2. GESTION DES RAIES CLASSIQUES
            else:
                theo_name, Jlow = get_theo_name(x, mol)
                new_name_dict[x] = theo_name
                if Jlow % 2 == 0 : 
                    dict_para[x] = theo_name
                elif Jlow % 2 == 1 : 
                    dict_orto[x] = theo_name
        
        # --- INJECTION DES SOMMES DANS LES GRILLES THEORIQUES ---
        if blends_to_add:
            # On passe sur toutes les grilles et modèles chargés
            for grid in Grid_Data + Model_Data:
                grid_names_list = list(grid.get_observable_names())
                grid_array = grid.get_observable_array()
                
                new_cols = []
                new_names = []
                for sum_name, (t1, t2) in blends_to_add.items():
                    if t1 in grid_names_list and t2 in grid_names_list:
                        idx1 = grid_names_list.index(t1)
                        idx2 = grid_names_list.index(t2)
                        # On additionne les deux colonnes théoriques
                        new_cols.append(grid_array[:, idx1] + grid_array[:, idx2])
                        new_names.append(sum_name)
                
                # Ajout dynamique dans l'objet grid pyismtools via Introspection
                if new_cols:
                    new_cols_array = np.column_stack(new_cols)
                    try:
                        # Si jamais la méthode propre existe finalement
                        for nom, col in zip(new_names, new_cols):
                            grid.add_observable(nom, col)
                    except AttributeError:
                        # --- MODE INTROSPECTION DYNAMIQUE ---
                        names_attr = None
                        array_attr = None
                        
                        # On fouille dans tous les attributs de l'objet
                        for attr in dir(grid):
                            # On ignore les méthodes (fonctions) et les attributs magiques
                            if callable(getattr(grid, attr)) or attr.startswith('__'):
                                continue
                                
                            val = getattr(grid, attr)
                            
                            # On cherche l'attribut qui contient exactement la liste des noms
                            if isinstance(val, (list, tuple, np.ndarray)) and len(val) == len(grid_names_list):
                                if list(val) == list(grid_names_list):
                                    names_attr = attr
                                    
                            # On cherche l'attribut qui contient le tableau numpy
                            elif isinstance(val, np.ndarray) and val.shape == grid_array.shape:
                                # On vérifie que c'est bien le bon tableau
                                if np.allclose(val, grid_array, equal_nan=True):
                                    array_attr = attr
                        
                        # Si on a trouvé les cachettes, on injecte nos blends !
                        if names_attr and array_attr:
                            current_names = list(getattr(grid, names_attr))
                            setattr(grid, names_attr, current_names + new_names)
                            
                            current_array = getattr(grid, array_attr)
                            setattr(grid, array_attr, np.hstack((current_array, new_cols_array)))
                        else:
                            raise ValueError(f"Impossible de trouver les variables internes de la grille ! Noms: {names_attr}, Data: {array_attr}")
        
        # --- APPLICATION DE LA TRADUCTION ---
        name_dictionnaire=f"Traduction_{name}"
        my_translation = ism.Translation(name_dictionnaire,new_name_dict)
        quantity_names = my_translation.get_alternate_names()
        
        trans_o = ism.Translation(name_dictionnaire,dict_orto)
        trans_p = ism.Translation(name_dictionnaire,dict_para)
        names_o,names_p=trans_o.get_alternate_names(),trans_p.get_alternate_names()
        
        Grid_Dic={}

        if len(Obs_Data)==1 : #Si que 1 observation
            for i_grid in range (len(Grid_Data)) : #For grids of model
                grid_o,grid_p=Grid_Data[i_grid].copy(),Grid_Data[i_grid].copy()#partie si on sépare les niveaux ortho et para
                grid_o.reduce_observables( names_o);grid_o.translate_names(trans_o)
                grid_p.reduce_observables(names_p);grid_p.translate_names(trans_p)
                
                Grid_Data[i_grid].reduce_observables(quantity_names)#pour les grilles complètes
                Grid_Data[i_grid].translate_names(my_translation)
                
                Grid_Dic[Grid_surnames[i_grid]]=Grid_Data[i_grid]
                if Process_OP==True :
                    Grid_Dic[Grid_surnames[i_grid]+"_O"]=grid_o
                    Grid_Dic[Grid_surnames[i_grid]+"_P"]=grid_p
                
            if Model_names!= None : 
                for el in Model_Data :  #for solo models
                    el.reduce_observables(quantity_names)
                    el.translate_names(my_translation)                
                
        elif len(Obs_Data)>1 : # Si plusieurs observations, alors plusieurs traductions
            print("Attention, il est recommandé de faire un Processing par jeu d'observation et les comparer ensuite")
            
    #--------------------------------------------------------------------------
    Approx_Dic={}
    BestMod_Dic={}
    BestChi_Dic={}
    Obs_Dic={}
    Obs_Dic[Obs_names[0]]=Obs_Data[0]
    
    for i_grid in range (len(Grid_Data)):
        # Création des interpolation linéaire et BestFit
        approximator=ism.Grid_approximator.interpolate_from_grid(Grid_Data[i_grid])
        best_model,best_chi2=ism.fit_tools.find_best_interpolated_model(Obs_Data[0],
                                                                        approximator,N_repetitions=20,
                                                                        scaling_factor_bounds=[0,limscale],
                                                                        cost_function="reduced_chi2")
        print(f"Grid named {Grid_surnames[i_grid]} is processed")
        Approx_Dic[Grid_surnames[i_grid]]=approximator
        BestMod_Dic[Grid_surnames[i_grid]]=best_model
        BestChi_Dic[Grid_surnames[i_grid]]=best_chi2   
        
        if Process_OP==True : #Si on veut séparer les ortho et para et comparer séparément
            approximator_ortho=ism.Grid_approximator.interpolate_from_grid(grid_o)
            approximator_para=ism.Grid_approximator.interpolate_from_grid(grid_p)
            obs_o, obs_p=Obs_Data[0].copy(),Obs_Data[0].copy()
            obs_o.reduce_observables(trans_o.get_names());obs_p.reduce_observables(trans_p.get_names())

            best_model_O, best_chi2_O = ism.fit_tools.find_best_interpolated_model(obs_o,
                                                        approximator_ortho,N_repetitions=20,scaling_factor_bounds=[0,limscale],cost_function="reduced_chi2")
            best_model_P, best_chi2_P = ism.fit_tools.find_best_interpolated_model(obs_p,
                                                        approximator_para,N_repetitions=20,scaling_factor_bounds=[0,limscale],cost_function="reduced_chi2")
            
            Approx_Dic[Grid_surnames[i_grid]+"_O"]=approximator_ortho
            Approx_Dic[Grid_surnames[i_grid]+"_P"]=approximator_para
            BestMod_Dic[Grid_surnames[i_grid]+"_O"]=best_model_O
            BestMod_Dic[Grid_surnames[i_grid]+"_P"]=best_model_P
            BestChi_Dic[Grid_surnames[i_grid]+"_O"]=best_chi2_O 
            BestChi_Dic[Grid_surnames[i_grid]+"_P"]=best_chi2_P 
            
            Obs_Dic[Grid_surnames[i_grid]+"_O"]=obs_o 
            Obs_Dic[Grid_surnames[i_grid]+"_P"]=obs_p 

            print(f"Grid named {Grid_surnames[i_grid]}_O and {Grid_surnames[i_grid]}_P are processed")

    Model_Dic={}        
    if Model_names!= None : 
        for i_mod in range (len(Model_Data)):
            # Création des interpolation liénaires et BestFit
            best_model,best_chi2=ism.fit_tools.find_best_model_in_grid(Obs_Data[0],Model_Data[i_mod],
                                                                        scaling_factor_bounds=[0,limscale])        
            print(f"Model named {Model_surnames[i_mod]} is processed")
            Model_Dic[Model_surnames[i_mod]]=best_model
            
    return(Obs_Data,Approx_Dic,BestMod_Dic,BestChi_Dic,Model_Dic,Obs_Dic,Grid_Dic)


#------------------------------------------------------------------------------
#                       Rapport ortho Para 
#------------------------------------------------------------------------------
def df_to_T(df,colE,colN,colg,index_T):
    idx1 = index_T[0]
    idx3 = index_T[2]
    #donne - (1/kb)facultatif * Ej+1*Ej-1/log10(Nup*gdow/(Ndow*gup)) voir p 13 Bron 2016
    return ( - (df.iloc[idx3][colE]-df.iloc[idx1][colE]) / 
            np.log( (df.iloc[idx3][colN]*df.iloc[idx1][colg]) / 
                     (df.iloc[idx1][colN]*df.iloc[idx3][colg]) )  )  

def Boltzmann (T,E_level,g_level): 
    #Renvoit la distibution de Boltzmann de tous les niveaux de H2 pour une température donnée 
    N_level=[]
    Z = np.sum (np.array(g_level) * np.exp( -(np.array(E_level))/T ) )
    #N_i = g_i * exp(-E_i / (k_B * T))
    for i in range(len(E_level)): 
        N_level.append( g_level[i]*np.exp (-E_level[i]/T) /Z)
    return (N_level)#


def df_Boltzmann_distrib (df,T,colE,colg,Ntot): 
    Z = (df[colg] * np.exp( -(df[colE])/T )).sum()
    df["N_Boltz"] = Ntot * df[colg] * np.exp( -(df[colE]) /T ) / Z
    
def OPR_LTE (N_level, is_P_level):
    #Calcule la somme des O sur les P de tous les niveaux de H2 en passant par la différence O et P 
    Sum_O,Sum_P=0,0
    for i in range(len(N_level)) : 
        if is_P_level[i]==True : 
            Sum_P+=N_level[i]
        else : 
            Sum_O+=N_level[i]
    return (Sum_O/Sum_P)



def df_OPR_triplet (df,colE,colN,colg,Temp,index_T,OP_LTE) :
    idx1 = index_T[0]
    idx2 = index_T[1]
    """Calcule le rapport ortho para entre trois raies selon Bron 2016"""
    
    if df.iloc[idx2]["o"] == "p" : #Si j pair         
        OP_t = OP_LTE * ( (df.iloc[idx1][colN] * df.iloc[idx2][colg]) 
                      / (  df.iloc[idx2][colN] * df.iloc[idx1][colg]) )  *np.exp (
            - (df.iloc[idx2][colE] - df.iloc[idx1][colE] ) / Temp )
                         
    elif df.iloc[idx2]["o"] == "o": #Si j impair:         
        OP_t = OP_LTE * ( (df.iloc[idx2][colN] * df.iloc[idx1][colg]) 
        / (df.iloc[idx1][colN] * df.iloc[idx2][colg]) )  *  np.exp (
             (df.iloc[idx2][colE] - df.iloc[idx1][colE] ) / Temp )
        
    return(OP_t)



    
    
def calculate_OPR_Triplet(df, col_name,E_level,g_level,is_P_level,end_level=5,vJ_names =["vu","Ju"],rotationnal=False):
    """Complex function derived from Bron et al 2016 to calculate OPR with a triplet of consecutives lines in same vu Ju
    Arguments : 
        - df with vu, Ju, mean_N , E(K), g for every level of interest 
        - col_name of the mean_N column density to use (df can have multiple N_mean in columns)
        - E_level, g_level, is_P_level : list of Energy (K) and degeneracy for ALL level of the molecules to calculate the Bolzmann distribution
        - end_level = N° of v level where the calculation stops 
    Returns : 
        List_OPR, List_OPR_LTE, Level_OPR, List_T, Is_P 
    """
    masks_v=read.mask_by_v(df,vJ_names[0])
    
    List_OPR=[]
    List_OPR_LTE=[]
    Level_OPR=[]
    List_T=[]
    Is_P=[]
    if rotationnal == True : 
        start = 0
        end_level = 1
    else : 
        start = 1
    for i in range (start,end_level) : 
        df_v=df[masks_v[i]]

        indices = read.detect_3consecutifs(df_v, vJ_names[1])
        print(indices)

        for j in range (len(indices)) : 
            Is_P.append(df_v.iloc[j][vJ_names[1]]%2)
            
            Level_OPR.append(f"v{df_v.iloc[indices[j][1]][vJ_names[0]]}-J{df_v.iloc[indices[j][1]][vJ_names[1]]}")
            
            T = df_to_T(df_v,"En(K)",col_name,"g",indices[j])
            List_T.append(T)
            
            Distrib_Boltzmann = Boltzmann(T, E_level, g_level)
            
            OP_LTE=OPR_LTE(Distrib_Boltzmann,is_P_level)
            List_OPR_LTE.append(OP_LTE)
            
            OP_triplet = df_OPR_triplet(df_v, "En(K)", col_name, "g", T, indices[j], OP_LTE)
            List_OPR.append(OP_triplet)    
            
    return(List_OPR, List_OPR_LTE, Level_OPR, List_T, Is_P ) 




def calculate_OPR_Triplet_V2(df, col_name,E_level,g_level,is_P_level,end_level=13,vJ_names =["vu","Ju"],rotationnal=False):
    """Complex function derived from Bron et al 2016 to calculate OPR with a triplet of consecutives lines in same vu Ju
    Arguments : 
        - df with vu, Ju, mean_N , E(K), g for every level of interest 
        - col_name of the mean_N column density to use (df can have multiple N_mean in columns)
        - E_level, g_level, is_P_level : list of Energy (K) and degeneracy for ALL level of the molecules to calculate the Bolzmann distribution
        - end_level = N° of v level where the calculation stops 
    Returns : 
        List_OPR, List_OPR_LTE, Level_OPR, List_T, Is_P 
    """
    masks_v=read.mask_by_v(df,vJ_names[0])
    
    List_OPR=[]
    List_OPR_LTE=[]
    Level_OPR=[]
    List_T=[]
    Is_P=[]

    if rotationnal == True : 
        start = 0
        end_level = 1
    else : 
        start = 1
        
    for i in range (start,end_level) : 
        df_v=df[masks_v[i]]
        indices = read.detect_3consecutifs_V2(df_v, vJ_names[1])
        
        for j in range (len(indices)) : 
            Ju  = indices[j][1]
            
            Is_P.append(df_v.iloc[Ju][vJ_names[1]]%2)
            Level_OPR.append(f"v{df_v.iloc[Ju][vJ_names[0]]}-J{df_v.iloc[Ju][vJ_names[1]]}")
  
            T = df_to_T(df_v,"En(K)",col_name,"g",indices[j])
            List_T.append(T)

            Distrib_Boltzmann = Boltzmann(T, E_level, g_level)
            
            OP_LTE=OPR_LTE(Distrib_Boltzmann,is_P_level)
            List_OPR_LTE.append(OP_LTE)
            
            OP_triplet = df_OPR_triplet(df_v, "En(K)", col_name, "g", T, indices[j], OP_LTE)
            List_OPR.append(OP_triplet)    

    return(List_OPR, List_OPR_LTE, Level_OPR, List_T, Is_P ) 


#------------------------------------------------------------------------------
#                       Correction d'extinction 
#------------------------------------------------------------------------------

def correct_extinction_Gordon23_Al_Av(lamb,RV):
    """Extrait de Gordon 2023 
    renvoit Alambda/Av dans la formule de correction d'extinction avec lamb en microns !
    A(lambda) est l'extinction absolue à la longueur d'onde donnée en magnitude
    A(V) est l'extinction dans la bande V en magnitude. -> Courbe d'extinction normalisée
    """
    def W(l,lb,d):
        """Needed in correction Gordon"""
        z = (l - (lb - 0.5*d))/d
        if z < 0:
            return 0
        elif 0 <= z <= 1:
            return 3.*z*z - 2.*z*z*z
        elif z >1:
            return 1


    def Dm(l,l0,g0,a):
        """Needed in correction Gordon"""
        g = 2.*g0 / (1. + np.exp(a*(l-l0)))
        return (g/l0)**2 / ( (l/l0 - l0/l)**2 + (g/l0)**2 )
    
    g1      = 0.38526
    alpha1  = 1.68467
    alpha2  = 0.78791
    lamb_b  = 4.30578
    delta   = 4.78338
    S1      = 0.06652
    lamb01  = 9.8434
    gamma01 = 2.21205
    a1      = -0.24703
    S2      = 0.0267
    lamb02  = 19.258294
    gamma02 = 17.0
    a2      = -0.27
    a = g1 * lamb**(-alpha1) * (1. - W(lamb,lamb_b,delta)) + g1*lamb_b**(-alpha1+alpha2)*lamb**(-alpha2)*W(lamb,lamb_b,delta) + S1*Dm(lamb,lamb01,gamma01,a1) + S2*Dm(lamb,lamb02,gamma02,a2)
    b = -1.01251 * lamb**(-1.06099)
 
    return a + b*(1./RV - 1./3.1)





def correct_extinction_Kaplan_powerlaw_Al (lamb , beta ):
    '''Renvoit Alambda selon la formule simple de Kaplan en loi de puissance
    '''
    return( (lamb**(-beta) ) )


#------------------------------------------------------------------------------
#                       Création de spectre Stellaire 
#------------------------------------------------------------------------------





#------------------------------------------------------------------------------
#                        OPR processing for 3 consecutive lines 
#------------------------------------------------------------------------------
#Detection of 3 lines 
#Texcit 
#Boltzman 


#------------------------------------------------------------------------------
#                       Passage de D à Go 
#------------------------------------------------------------------------------





def get_min_withCond(df, col_distance, col_condition, valeur_min, valeur_max):
    """
    Trouve l'indice minimal où la colonne 'col_distance' est minimale et la colonne 'col_condition' est comprise entre valeur_min et valeur_max.

    Args:
        df (pd.DataFrame): Le DataFrame pandas.
        col_distance (str): Le nom de la colonne à minimiser (par exemple, 'Distance').
        col_condition (str): Le nom de la colonne de condition.
        valeur_min (float): La valeur minimale de la condition.
        valeur_max (float): La valeur maximale de la condition.

    Returns:
        int: L'indice minimal où la distance est minimale et la condition est satisfaite, ou None si aucune valeur ne satisfait la condition.
    """

    indices_satisfaisants = df[(df[col_condition] >= valeur_min) & (df[col_condition] < valeur_max)].index.tolist()

    if not indices_satisfaisants:
        return None  # Aucune valeur ne satisfait la condition

    df_satisfaisant = df.loc[indices_satisfaisants]
    indice_min = df_satisfaisant[col_distance].idxmin()
    if indice_min==None : 
        print("Aucune valeur ne satisfait la condition")
    return indice_min






def filter_df_general_nearest(df_general, df_reference, threshold_dh2, pth_col_gen='Pth', go_col_gen='Go',
                              pth_col_ref='Pth', go_col_ref='Go', dh2_col_ref='D(H_H2)'):
    """
    Pour transition H_H2
    Filters df_general by finding the nearest (Pth, Go) point in df_reference
    and checking if its D(H_H2) value is below the threshold.

    Args:
        df_general (pd.DataFrame): The DataFrame to filter.
        df_reference (pd.DataFrame): The DataFrame to find the nearest points in.
        threshold_dh2 (float): The threshold value for D(H_H2).
        pth_col_gen (str, optional): The Pth column name in df_general. Defaults to 'Pth'.
        go_col_gen (str, optional): The Go column name in df_general. Defaults to 'Go'.
        pth_col_ref (str, optional): The Pth column name in df_reference. Defaults to 'Pth'.
        go_col_ref (str, optional): The Go column name in df_reference. Defaults to 'Go'.
        dh2_col_ref (str, optional): The D(H_H2) column name in df_reference. Defaults to 'D(H_H2)'.

    Returns:
        pd.DataFrame: The filtered df_general.
    """
    indices_to_keep = []
    for index_gen, row_gen in df_general.iterrows():
        pth_gen = row_gen[pth_col_gen]
        go_gen = row_gen[go_col_gen]

        # Calculate squared Euclidean distances
        distances_squared = (df_reference[pth_col_ref] - pth_gen)**2 + (df_reference[go_col_ref] - go_gen)**2

        # Find the index of the nearest point in df_reference
        nearest_index = distances_squared.idxmin()

        # Check if the D(H_H2) value of the nearest point is below the threshold
        if df_reference.loc[nearest_index, dh2_col_ref] <= threshold_dh2:
            indices_to_keep.append(index_gen)

    return df_general.loc[indices_to_keep].copy()



def calculate_slope(x_data, y_data, relationship_type='linear'):
    """
    Calculates the slope coefficient for a given relationship type between x and y data.

    Args:
        x_data (list or np.ndarray): A list or array of x values.
        y_data (list or np.ndarray): A list or array of y values.
        relationship_type (str): The type of relationship to fit.
                                 'linear' for y = m*x + c
                                 'power' for y = a*x^b (fits log(y) = log(a) + b*log(x))
                                 Defaults to 'linear'.

    Returns:
        float or tuple: The calculated slope coefficient.
                        For 'linear', returns the slope m.
                        For 'power', returns the exponent b (slope in log-log space).
                        Returns None if calculation is not possible (e.g., insufficient data, invalid type, log(0)).

    Raises:
        ValueError: If x_data and y_data have different lengths,
                    if relationship_type is invalid,
                    or if data is non-positive for 'power' relationship.
    """
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length.")

    n = len(x_data)
    if n < 2:
        # Need at least two points to determine a slope
        print("Warning: Need at least 2 data points to calculate a slope.")
        return None

    # --- Method 1: Simple (Manual Calculation) ---

    if relationship_type == 'linear':
        # Calculate sums for linear regression (y = m*x + c) using least squares formula
        # m = (n * sum(xy) - sum(x) * sum(y)) / (n * sum(x^2) - (sum(x))^2)
        sum_x = sum(x_data)
        sum_y = sum(y_data)
        sum_xy = sum(x * y for x, y in zip(x_data, y_data))
        sum_x_squared = sum(x**2 for x in x_data)

        denominator = n * sum_x_squared - sum_x**2
        if denominator == 0:
            # This happens if all x values are the same (vertical line)
            print("Warning: Cannot calculate slope, all x values are the same.")
            return None

        simple_slope = (n * sum_xy - sum_x * sum_y) / denominator
        # return simple_slope # Uncomment to return simple method result directly

    elif relationship_type == 'power':
        # Calculate sums for linear regression on log-transformed data (log(y) = b*log(x) + log(a))
        # The slope of log(y) vs log(x) is 'b'.
        # Need to handle non-positive values as log is undefined
        log_x_data = []
        log_y_data = []
        for x, y in zip(x_data, y_data):
            if x <= 0 or y <= 0:
                raise ValueError("For 'power' relationship, all x and y values must be positive.")
            log_x_data.append(math.log(x))
            log_y_data.append(math.log(y))

        sum_log_x = sum(log_x_data)
        sum_log_y = sum(log_y_data)
        sum_log_x_log_y = sum(lx * ly for lx, ly in zip(log_x_data, log_y_data))
        sum_log_x_squared = sum(lx**2 for lx in log_x_data)

        denominator = n * sum_log_x_squared - sum_log_x**2
        if denominator == 0:
             # This happens if all log(x) values are the same (all x values are the same)
            print("Warning: Cannot calculate power exponent, all log(x) values (or x values) are the same.")
            return None

        simple_slope = (n * sum_log_x_log_y - sum_log_x * sum_log_y) / denominator
        return simple_slope # Uncomment to return simple method result directly

    else:
        raise ValueError("Invalid relationship_type. Choose 'linear' or 'power'.")

    # --- Method 2: Developed (Using NumPy) ---
    # This method is generally preferred for accuracy and robustness

    x_np = np.array(x_data)
    y_np = np.array(y_data)

    if relationship_type == 'linear':
        # Use numpy's polyfit for linear regression (degree 1)
        # It returns coefficients [slope, intercept]
        developed_coeffs = np.polyfit(x_np, y_np, 1)
        developed_slope = developed_coeffs[0]
        # return developed_slope # Uncomment to return developed method result directly

    elif relationship_type == 'power':
        # Transform data using log and use numpy's polyfit
        # Check for non-positive values before logging
        if np.any(x_np <= 0) or np.any(y_np <= 0):
             # This check was already done in the simple method, but good to have here too
             raise ValueError("For 'power' relationship, all x and y values must be positive.")

        log_x_np = np.log(x_np)
        log_y_np = np.log(y_np)

        # Fit a linear model to the log-transformed data
        # The coefficients will be [exponent (b), log(a)]
        developed_coeffs = np.polyfit(log_x_np, log_y_np, 1)
        developed_slope = developed_coeffs[0] # This is the exponent 'b'
        # return developed_slope # Uncomment to return developed method result directly

    # By default, return the result from the more robust developed method
    return developed_slope




def curve_prolongation(x_data, y_data, x_new_points, method='flat', x_fit_start=1500):
    """
    Prolonge une courbe donnée en se basant sur son comportement après un certain point X.

    Args:
        x_data (array-like): Les coordonnées X des points de la courbe existante.
        y_data (array-like): Les coordonnées Y des points de la courbe existante.
        x_new_points (array-like): Les nouvelles coordonnées X pour lesquelles prolonger la courbe.
                                    Ces points devraient idéalement être supérieurs au max(x_data).
        method (str): La méthode de prolongation.
                      Peut être 'flat' (plate), 'linear' (linéaire), ou 'log' (loi logarithmique y = a*log(x) + b).
        x_fit_start (float): Le point X à partir duquel les données seront utilisées pour l'ajustement.
                             (Par défaut 2000, comme spécifié par l'utilisateur).

    Returns:
        numpy.ndarray: Les coordonnées Y prolongées pour x_new_points.
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    x_new_points = np.asarray(x_new_points)

    # Filtrer les données pour l'ajustement (x > x_fit_start)
    mask_fit = x_data >= x_fit_start
    x_fit = x_data[mask_fit]
    y_fit = y_data[mask_fit]


    if method == 'flat':
        # Prolonger de manière plate (dernier point connu)
        last_y = y_fit[-1] # Prendre le tout dernier point du dataset
        y_extrapolated = [last_y]*len(x_new_points)
        
        
    elif method == 'linear':
        # Prolonger de manière linéaire (y = ax + b)
        # Ajustement linéaire sur les points x_fit, y_fit
        a, b = np.polyfit(x_fit, y_fit, 1)
        y_extrapolated = a * x_new_points + b
        y_extrapolated = y_extrapolated.tolist()
    elif method == 'log':
        # Prolonger suivant une loi logarithmique (y = a * log(x) + b)
        # Nécessite que x_fit et x_new_points soient > 0
        if np.any(x_fit <= 0) or np.any(x_new_points <= 0):
            raise ValueError("La méthode 'log' ne peut être utilisée si les valeurs X sont inférieures ou égales à zéro.")

        # Transformer X en log(X) pour un ajustement linéaire
        log_x_fit = np.log(x_fit)
        
        # Ajustement linéaire sur (log(X), Y)
        a, b = np.polyfit(log_x_fit, y_fit, 1)
        
        # Extrapoler avec la fonction log
        y_extrapolated = a * np.log(x_new_points) + b
        y_extrapolated = y_extrapolated.tolist()

    else:
        raise ValueError("Méthode non reconnue. Choisissez parmi 'flat', 'linear', 'log'.")

    return y_extrapolated




def process_quantum_identifiers(df, extractor_func, col_jup='J_up'):
    """
    Enriches dataframe with quantum identifiers and enforces numeric types for J.

    Args:
        df (pd.DataFrame): Input dataframe.
        extractor_func (callable): Function returning (Band, J_up, Branch).
        col_jup (str): Name of the column containing J_up to force int conversion.

    Returns:
        pd.DataFrame: Processed dataframe sorted by Band and J_up.
    Utilisaion dans Spectrum_JWST.py
    """
    df_out = df.copy()
    
    # Apply extraction
    try:
        temp_data = df_out.apply(extractor_func, axis=1, result_type='expand')
        df_out[['Band', col_jup, 'Branch']] = temp_data
    except Exception as e:
        print(f"Error during extraction: {e}")
        return df_out

    # Force J_up to integer (handles strings like '1.0' or ' 1 ')
    # 'coerce' turns errors into NaN, then we drop them or keep valid ones
    df_out[col_jup] = pd.to_numeric(df_out[col_jup], errors='coerce')
    
    # Optional: Remove rows where J_up could not be determined
    df_out = df_out.dropna(subset=[col_jup])
    df_out[col_jup] = df_out[col_jup].astype(int)

    return df_out



import matplotlib.cm as cm
def get_safe_variation(val_base, pct_requested, val_min, val_max):
    """
    Calcule la valeur de test et le vrai pourcentage appliqué sans dépasser les bornes.
    """
    if val_min is None or val_max is None:
        val_test = val_base * (1.0 + pct_requested / 100.0)
        return val_test, pct_requested
        
    val_test = val_base * (1.0 + pct_requested / 100.0)
    val_test_clamped = np.clip(val_test, val_min, val_max)
    real_pct = ((val_test_clamped - val_base) / val_base) * 100.0
    return val_test_clamped, real_pct


def Sensitivity_Excitation_Gradient(approximator, observation, base_params, param_to_vary, variations_pct, viz_module,cmap="coolwarm"):
    """
    Génère un diagramme d'excitation en faisant varier un paramètre dans les limites de la grille.
    """
    models_list = []
    names_list = []
    colors_list = []
    
    # Récupération des bornes de la grille (Pth=index 0, G0=index 1)
    grid_mins = approximator.get_parameter_mins()
    grid_maxs = approximator.get_parameter_maxs()
    
    cmap = cm.get_cmap(cmap)
    norm = plt.Normalize(vmin=min(variations_pct), vmax=max(variations_pct))
    
    for pct in variations_pct:
        # Sécurisation des valeurs selon le paramètre
        if param_to_vary == 'Pth':
            P_test, real_pct = get_safe_variation(base_params['Pth'], pct, grid_mins[0], grid_maxs[0])
            G_test = base_params['G0']
            Om_test = base_params['Omega']
        elif param_to_vary == 'G0':
            P_test = base_params['Pth']
            G_test, real_pct = get_safe_variation(base_params['G0'], pct, grid_mins[1], grid_maxs[1])
            Om_test = base_params['Omega']
        elif param_to_vary == 'Omega': # Omega n'a pas de borne stricte dans la grille
            P_test = base_params['Pth']
            G_test = base_params['G0']
            Om_test, real_pct = get_safe_variation(base_params['Omega'], pct, 1e-4, 10.0)
        
        # Création du modèle
        mod = approximator([P_test, G_test])
        mod.set_scaling_factor(Om_test)
        models_list.append(mod)
        
        # Nommage intelligent
        signe = "+" if real_pct > 0 else ""
        label = rf"{param_to_vary} {signe}{real_pct:.1f} \%"
        if abs(real_pct) < abs(pct) and pct != 0:
            label += " (Lim)" # Indique qu'on a tapé la limite de la grille
            
        names_list.append(label)
        colors_list.append(cmap(norm(pct))) # On garde la couleur cible pour la lisibilité
        
    styles_list = ["-" if pct == 0 else "--" for pct in variations_pct]
    
    viz_module.Setting_Figure("Excitation")
    viz_module.Excitation_Diagram_V4(
        Model=models_list, 
        Observation=observation,
        NameList=names_list, 
        color=colors_list, style=styles_list,
        Details=True, threshold=False, ShowLine=True,
        Name=f"Sensibilité à {param_to_vary}",
        Full=True, mol="co_new", Size=(16, 10), OP_visible=False, showerr_color=False
    )

def Plot_Line_Sensitivity_Bar(approximator, base_params, param_to_vary, variation_pct=10.0):
    """
    Trace un Bar Plot sans faire crasher l'interpolateur si on touche le bord de la grille.
    """
    import matplotlib.pyplot as plt 
    grid_mins = approximator.get_parameter_mins()
    grid_maxs = approximator.get_parameter_maxs()
    
    # Modèle de base
    mod_base = approximator([base_params['Pth'], base_params['G0']])
    mod_base.set_scaling_factor(base_params['Omega'])
    raies = mod_base.get_observable_names()
    int_base = mod_base.get_observable_vector()
    
    # Modèle perturbé sécurisé
    if param_to_vary == 'Pth':
        P_test, real_pct = get_safe_variation(base_params['Pth'], variation_pct, grid_mins[0], grid_maxs[0])
        G_test = base_params['G0']
        Om_test = base_params['Omega']
    elif param_to_vary == 'G0':
        P_test = base_params['Pth']
        G_test, real_pct = get_safe_variation(base_params['G0'], variation_pct, grid_mins[1], grid_maxs[1])
        Om_test = base_params['Omega']
    elif param_to_vary == 'Omega':
        P_test = base_params['Pth']
        G_test = base_params['G0']
        Om_test, real_pct = get_safe_variation(base_params['Omega'], variation_pct, 1e-4, 10.0)
    
    # Si on ne peut pas du tout varier (on est pile sur le bord et on demande à aller plus loin)
    if real_pct == 0 and variation_pct != 0:
        print(f"⚠️ Impossible d'augmenter {param_to_vary} de {variation_pct}%, on est déjà au max de la grille !")
        return
        
    mod_var = approximator([P_test, G_test])
    mod_var.set_scaling_factor(Om_test)
    int_var = mod_var.get_observable_vector()
    
    # Calcul et Plot
    sensibilite = ((int_var - int_base) / int_base) * 100.0
    df_sens = pd.DataFrame({'Line': raies, 'Sensitivity_pct': sensibilite})
    df_sens['Abs_Sens'] = df_sens['Sensitivity_pct'].abs()
    df_sens = df_sens.sort_values('Abs_Sens', ascending=True)
    
    plt.figure(figsize=(10, max(6, len(raies)*0.3)))
    colors = ['firebrick' if val < 0 else 'forestgreen' for val in df_sens['Sensitivity_pct']]
    bars = plt.barh(df_sens['Line'], df_sens['Sensitivity_pct'], color=colors, edgecolor='black')
    
    plt.axvline(0, color='black', linewidth=1)
    
    titre = f"Sensibilité pour {param_to_vary} : +{real_pct:.1f}%"
    if real_pct != variation_pct:
        titre += " (Plafonné par la grille)"
        
    plt.xlabel("Variation de l'intensité de la raie [%]", fontsize=14)
    plt.title(titre, fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    
    
    
#########################CALCULS DANS LE CUBE DE DONNEES 

import scipy.optimize
import numpy as np
import copy

def fit_best_model_fixed_Omega(observation, approximator, fixed_Omega, method='L-BFGS-B'):
    """
    Trouve les meilleurs paramètres physiques en maintenant le Scaling Factor (Omega) fixe.
    L'observation originale n'est JAMAIS modifiée.
    
    Args:
        observation: Objet Observation de pyismtools.
        approximator: Objet Grid_approximator de pyismtools.
        fixed_Omega (float): La valeur du Scaling factor à forcer.
        method (str): La méthode d'optimisation de scipy.
    """
    
    # 1. On travaille sur des copies par sécurité (comme dans fit_tools.py)
    obs_cp = observation.copy()
    approx_cp = approximator.copy()
    
    # 2. Réduction aux observables communs
    obs_names = set(obs_cp.get_observable_names())
    mod_names = set(approx_cp.get_observable_names())
    intersection = list(mod_names.intersection(obs_names))
    obs_cp.reduce_observables(intersection)
    approx_cp.reduce_observables(intersection)
    
    # Préparation des masques pour ignorer les NaN dans les observations
    nan_mask = np.isnan(obs_cp.get_observable_vector())
    obs_vector = obs_cp.get_observable_vector()[~nan_mask]
    err_vector = obs_cp.get_error_vector()[~nan_mask]
    err_types = obs_cp.get_error_type_vector()[~nan_mask]
    N_params = approx_cp.get_N_parameters()

    # 3. Récupération des limites de la grille depuis l'approximateur
    # On optimise en log10 pour la stabilité
    l_mins = np.log10(approx_cp.get_parameter_mins())
    l_maxs = np.log10(approx_cp.get_parameter_maxs())
    bounds = [(l_mins[i], l_maxs[i]) for i in range(N_params)]
    
    # Point de départ initial (milieu de la grille)
    x0 = (l_mins + l_maxs) / 2.0

    # 4. Définition de la fonction de coût interne
    def cost_func(log_params):
        params = 10**log_params
        
        # Génération du modèle interpolé via l'approximateur
        # (allow_extrapolation=False par défaut, protégé par les bounds)
        test_model = approx_cp(params, allow_extrapolation=False)
        
        # ON APPLIQUE LE SCALING FORCÉ AU MODÈLE
        test_model.set_scaling_factor(fixed_Omega)
        
        # Récupération du vecteur de flux du modèle (avec le scaling appliqué !)
        mod_vector = test_model.get_observable_vector(scaled=True)[~nan_mask]
        
        # Calcul du chi2 réduit (appel à la fonction native de utils)
        return ism.utils.reduced_chi2(mod_vector, obs_vector, err_vector, err_types, N_params)

    # 5. Minimisation avec scipy
    res = scipy.optimize.minimize(
        cost_func, 
        x0,
        bounds=bounds,
        method=method
    )
    
    # 6. Reconstitution du meilleur modèle trouvé
    best_params = 10**res.x
    
    # On réutilise l'approximateur original (non réduit) pour avoir toutes les observables
    best_model = approximator(best_params)
    best_model.set_scaling_factor(fixed_Omega)
    
    return best_model, best_params, res.fun



def fit_fixed_Omega_and_Pth(observation, approximator, fixed_Omega, fixed_Pth, G0_bounds=(1e0, 1e6)):
    """
    Fixe Omega et Pth, et trouve le meilleur G0.
    """
    obs_cp = observation.copy()
    approx_cp = approximator.copy()
    
    # Réduction aux observables communs
    intersection = list(set(approx_cp.get_observable_names()).intersection(set(obs_cp.get_observable_names())))
    obs_cp.reduce_observables(intersection)
    approx_cp.reduce_observables(intersection)
    
    nan_mask = np.isnan(obs_cp.get_observable_vector())
    obs_vector = obs_cp.get_observable_vector()[~nan_mask]
    err_vector = obs_cp.get_error_vector()[~nan_mask]
    err_types = obs_cp.get_error_type_vector()[~nan_mask]
    N_params = approx_cp.get_N_parameters()

    def cost_func(log_G0):
        # On injecte le Pth fixe et le G0 variable
        test_model = approx_cp([fixed_Pth, 10**log_G0], allow_extrapolation=False)
        test_model.set_scaling_factor(fixed_Omega)
        
        mod_vector = test_model.get_observable_vector(scaled=True)[~nan_mask]
        return ism.utils.reduced_chi2(mod_vector, obs_vector, err_vector, err_types, N_params)

    # Minimisation 1D sur G0
    l_bounds = (np.log10(G0_bounds[0]), np.log10(G0_bounds[1]))
    res = scipy.optimize.minimize_scalar(cost_func, bounds=l_bounds, method='bounded')
    
    best_G0 = 10**res.x
    best_model = approximator([fixed_Pth, best_G0])
    best_model.set_scaling_factor(fixed_Omega)
    
    return best_model, best_G0, res.fun



def fit_with_constraints(observation, approximator, fixed_omega=None, fixed_physical_params=None):
    """
    Fonction universelle pour contraindre n'importe quel paramètre.
    
    Args:
        observation: L'objet Observation
        approximator: L'objet Grid_approximator
        fixed_omega (float, optionnel): Si renseigné, fixe le scaling factor.
                                        Sinon, on suppose Omega=1 (pas de scaling).
        fixed_physical_params (dict, optionnel): Dictionnaire des paramètres physiques à figer.
                                                 Ex: {"Pgas": 1e7} ou {"Pgas": 1e7, "Go": 5e3}
    """
    if fixed_physical_params is None:
        fixed_physical_params = {}
        
    obs_cp = observation.copy()
    approx_cp = approximator.copy()
    
    intersection = list(set(approx_cp.get_observable_names()).intersection(set(obs_cp.get_observable_names())))
    obs_cp.reduce_observables(intersection)
    approx_cp.reduce_observables(intersection)
    
    nan_mask = np.isnan(obs_cp.get_observable_vector())
    obs_vector = obs_cp.get_observable_vector()[~nan_mask]
    err_vector = obs_cp.get_error_vector()[~nan_mask]
    err_types = obs_cp.get_error_type_vector()[~nan_mask]
    
    # Analyse des paramètres de la grille
    param_names = approx_cp.get_parameter_names()
    N_params = approx_cp.get_N_parameters()
    
    # Identifier quels paramètres sont libres et lesquels sont fixés
    free_indices = []
    free_bounds = []
    l_mins = np.log10(approx_cp.get_parameter_mins())
    l_maxs = np.log10(approx_cp.get_parameter_maxs())
    
    for i, name in enumerate(param_names):
        if name not in fixed_physical_params:
            free_indices.append(i)
            free_bounds.append((l_mins[i], l_maxs[i]))
            
    # Si tout est fixé, on calcule juste le chi2 direct
    if len(free_indices) == 0:
        params_array = [fixed_physical_params[name] for name in param_names]
        test_model = approx_cp(params_array)
        if fixed_omega is not None: test_model.set_scaling_factor(fixed_omega)
        mod_vector = test_model.get_observable_vector(scaled=True)[~nan_mask]
        chi2 = ism.utils.reduced_chi2(mod_vector, obs_vector, err_vector, err_types, N_params)
        return test_model, params_array, chi2

    # Point de départ pour les paramètres libres
    x0 = [(b[0] + b[1])/2.0 for b in free_bounds]

    def cost_func(log_free_params):
        # 1. Reconstruire le vecteur complet de paramètres [Pgas, Go, ...]
        full_params = []
        free_counter = 0
        for i, name in enumerate(param_names):
            if name in fixed_physical_params:
                full_params.append(fixed_physical_params[name]) # On met la valeur fixe
            else:
                full_params.append(10**log_free_params[free_counter]) # On met la valeur testée
                free_counter += 1
                
        # 2. Appel au modèle
        test_model = approx_cp(full_params, allow_extrapolation=False)
        
        # 3. Application du scaling si demandé
        if fixed_omega is not None:
            test_model.set_scaling_factor(fixed_omega)
            
        # 4. Calcul du Chi2
        mod_vector = test_model.get_observable_vector(scaled=True)[~nan_mask]
        return ism.utils.reduced_chi2(mod_vector, obs_vector, err_vector, err_types, N_params)

    # Lancement de l'optimisation (seulement sur les paramètres libres !)
    if len(free_indices) == 1:
        res = scipy.optimize.minimize_scalar(cost_func, bounds=free_bounds[0], method='bounded')
        best_free_log_params = [res.x]
    else:
        res = scipy.optimize.minimize(cost_func, x0, bounds=free_bounds, method='L-BFGS-B')
        best_free_log_params = res.x

    # Reconstitution finale du meilleur modèle
    final_params = []
    free_counter = 0
    for name in param_names:
        if name in fixed_physical_params:
            final_params.append(fixed_physical_params[name])
        else:
            final_params.append(10**best_free_log_params[free_counter])
            free_counter += 1
            
    best_model = approximator(final_params)
    if fixed_omega is not None:
        best_model.set_scaling_factor(fixed_omega)

    # Retourne: Modèle, Liste des paramètres complets [Pgas, G0], Chi2
    return best_model, final_params, res.fun
