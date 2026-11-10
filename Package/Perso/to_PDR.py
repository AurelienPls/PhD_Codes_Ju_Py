#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:28:53 2025

@author: aurelien
"""

import pandas as pd 
import numpy as np 
from scipy.optimize import curve_fit
import re
import sys
chemin_modules = "/Users/aurelien/Desktop/PhD/Codes/Package/Perso/"
sys.path.append(chemin_modules)
import read as read 

import os 
# =============================================================================
#                                Input PDR 
# =============================================================================

#Create pdr.in
#Create template.in 
#Create file for grid 
#Creta grid from file and template 













# =============================================================================
#                               Read data files 
# =============================================================================


#Read collision file 
#Read Line file 
#Read Energy file 
#------------------------------------------------------------------------------
def read_level_to_df(element="H2",line_data=8,end_line_data=-5):
    rootbase="/Users/aurelien/Desktop/Stage_M2_LERMA/PDR/1.7/data/Levels/"
    namebase="level_h2"
    doc_base=open(rootbase+namebase+".dat","r")
    file_base=doc_base.read();  ligne_level=file_base.split("\n");  
    
    E_level,g_level,is_P_level=[],[],[]
    for i in range (line_data,len(ligne_level)+end_line_data): 
        temp=ligne_level[i].split(maxsplit=7)
        E_level.append(float(temp[2]))
        g_level.append(float(temp[1]))
        is_P_level.append(int(temp[5])%2==0)
        
    #Version avec dataframe 
    ColNames =['n','g','En(K)','o','v','J','comment']
    df_level = read.dat_to_pd(rootbase, namebase+".dat", 8,ColNames=ColNames,end_data=-5,tofloat=False)
    df_level[['n','v','J']] = df_level[['n','v','J']].astype(int)
    return df_level



def add_branch_label(df):
    def branch(row):
        delta_J = row["Ju"] - row["Jl"]
        if delta_J == 1:
            letter = "R"
        elif delta_J == 0:
            letter = "Q"
        elif delta_J == -1:
            letter = "P"
        elif delta_J == 2:
            letter = "S"
        elif delta_J == -2:
            letter = "O"
        else:
            letter = "?"
        return f"{row['vu']}-{row['vl']}_{letter}( {int(row['Jl'])})"
    df["Line"] = df.apply(branch, axis=1)
    return df




#------------------------------------------------------------------------------
def write_coll_times_scale(fichier_entree, fichier_sortie, echelle):
    """
    Copie le format d'un fichier, en multipliant uniquement les taux de collision par une échelle,
    en conservant le format des trois premières colonnes, de la ligne des températures, et en affichant
    les taux de collision avec 4 chiffres après la virgule.

    Args:
        fichier_entree (str): Chemin du fichier d'entrée.
        fichier_sortie (str): Chemin du fichier de sortie.
        echelle (float): Facteur d'échelle pour la multiplication.
    """

    try:
        with open(fichier_entree, 'r') as entree, open(fichier_sortie, 'w') as sortie:
            for ligne in entree:
                ligne_modifiee = ''
                mots = ligne.split()

                if ligne.startswith('!COLLISION TEMPERATURES'):
                    # Traitement spécial pour la ligne des températures
                    sortie.write(ligne)
                    continue

                if ligne.startswith('!NUM'):
                    sortie.write(ligne)
                    continue

                if len(mots) >= 4:  # Vérifier qu'il y a au moins 4 colonnes
                    # Traitement des trois premières colonnes (entiers)
                    try:
                        ligne_modifiee = f'{int(mots[0]):>4} {int(mots[1]):>4} {int(mots[2]):>4} '
                    except ValueError:
                        sortie.write(ligne)
                        continue

                    # Traitement des taux de collision (multiplication et formatage)
                    for i in range(3, len(mots)):
                        try:
                            nombre = float(mots[i])
                            nombre_modifie = nombre * echelle
                            ligne_modifiee += f'{nombre_modifie:.4E} '  # Formatage avec 4 chiffres après la virgule
                        except ValueError:
                            ligne_modifiee += mots[i] + ' '

                    sortie.write(ligne_modifiee.rstrip() + '\n')
                else:
                    sortie.write(ligne)

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{fichier_entree}' n'a pas été trouvé.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")




#------------------------------------------------------------------------------
def df_to_collision(df, fichier_sortie, num_energy_levels, num_collisions, colonnes, commentaire_fin=None):
    """
    Crée un fichier au format spécifié à partir d'un DataFrame, avec des noms de colonnes personnalisés,
    une ligne d'en-tête fixe et une option pour ajouter un commentaire à la fin.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        fichier_sortie (str): Chemin du fichier de sortie.
        num_energy_levels (int): Nombre de niveaux d'énergie.
        num_collisions (int): Nombre de transitions collisionnelles.
        colonnes (list): Liste des noms de colonnes (pour le DataFrame).
        commentaire_fin (str, optional): Chaîne de caractères à ajouter comme commentaire à la fin du fichier.
    """

    if os.path.exists(fichier_sortie):
        raise FileExistsError(f"Le fichier '{fichier_sortie}' existe déjà.")

    try:
        with open(fichier_sortie, 'w') as sortie:
            # Écriture des informations d'en-tête
            sortie.write(f'!NUMBER OF ENERGY  LEVELS\n{num_energy_levels}\n')
            sortie.write(f'!NUMBER OF COLLISIONAL  TRANSITIONS\n{num_collisions}\n')

            # Extraction des températures de collision à partir des noms de colonnes
            temperatures_colonnes = colonnes[3:]
            temperatures = [float(col.replace('K', '')) for col in temperatures_colonnes]

            sortie.write(f'!NUMBER OF COLLISION  TEMPERATURES\n{len(temperatures)}\n')

            # Écriture des températures de collision
            temp_line = '!COLLISION TEMPERATURES\n'
            for temp in temperatures:
                temp_line += f'{temp:10.1f} '
            sortie.write(temp_line.rstrip() + '\n')

            # Écriture de l'en-tête fixe
            sortie.write('!NUM    UP    LOW    DOWN_COLL_RATES [cm^3 s^-1]\n')

            # Écriture des données du DataFrame
            for index, row in df.iterrows():
                ligne = f'{int(row[colonnes[0]]):>4} {int(row[colonnes[1]]):>4} {int(row[colonnes[2]]):>4} '
                for temp_col in temperatures_colonnes:
                    ligne += f'{row[temp_col]:.4E} '
                sortie.write(ligne.rstrip() + '\n')

            # Ajout du commentaire à la fin (si fourni)
            if commentaire_fin:
                sortie.write(f'!{commentaire_fin}\n')

    except Exception as e:
        print(f"Une erreur s'est produite : {e}")


def Info_to_df(mol="h2"):
    """Returns a df with all info from the Info file 
    Works mainl for H2"""
    main_root="/Users/aurelien/Desktop/Stage_M2/"
    rootbase=main_root+"PDR/1.7/"
    namebase=f"all_{mol}_lines_info"
    doc_base=open(rootbase+namebase+".dat","r")
    file_base=doc_base.read();  ligne_base=file_base.split("\n")
    
    Col_info = ["Line","vu","Ju","vl","Jl","gu" ,"gl" ,"Eu(K)","El[K]","Eul[K]","Aul","Wavelenght"]
    #df_Info = read.dat_to_pd(root_base+ namebase, line_data=2, ColN,end_data=-1)
    df_Info=pd.DataFrame( [x.split(",") for x in ligne_base[2:-1]],columns=Col_info )
    df_Info[["vu","Ju","vl","Jl"]] = df_Info[["vu","Ju","vl","Jl" ]].astype(int)
    df_Info[["gu" ,"gl","Eu(K)","El[K]","Eul[K]","Aul","Wavelenght" ]] = df_Info[["gu" ,"gl","Eu(K)","El[K]","Eul[K]","Aul","Wavelenght" ]].astype(float)
    return df_Info



def Level_to_pd(mol="h2"):
    """Returns a df with all level lines from a specific molecule. 
    Works mainly for h2"""
    rootbase="/Users/aurelien/Desktop/Stage_M2/PDR/1.7/data/Levels/"
    namebase=f"level_{mol}"
    doc_base=open(rootbase+namebase+".dat","r")
    file_base=doc_base.read();  ligne_level=file_base.split("\n")
    
    E_level,g_level,is_P_level=[],[],[]
    for i in range (8,len(ligne_level)-5): 
        temp=ligne_level[i].split(maxsplit=7)
        E_level.append(float(temp[2]))
        g_level.append(float(temp[1]))
        is_P_level.append(int(temp[5])%2==0)
        
    #Version avec dataframe 
    ColNames =['n','g','En(K)','o','v','J','comment']
    df_level = read.dat_to_pd(rootbase, namebase+".dat", 8,ColNames=ColNames,end_data=-5,tofloat=False)
    df_level[['n','v','J']] = df_level[['n','v','J']].astype(int)
    df_level[["En(K)",'g']] = df_level[["En(K)",'g']].astype(float)
    return df_level







def collision_to_df( name: str,path=None) -> pd.DataFrame:
    """
    Lit un fichier .dat de taux de collision et stocke les données dans un DataFrame.

    Args:
        file_path (str): Le chemin complet vers le fichier .dat.
        molecule_name (str): Le nom de la molécule à prendre en compte (actuellement non utilisé
                             dans cette version car le fichier d'exemple ne contient pas
                             d'information sur la molécule par ligne de données,
                             mais peut être implémenté si la structure du fichier évolue).

    Returns:
        pd.DataFrame: Un DataFrame contenant les taux de collision.
                      Les premières colonnes sont des entiers (NUM, UP, LOW)
                      et les colonnes de taux de collision sont des floats,
                      nommées 'TEMP_PDR' (par exemple, '2.0K_PDR').
                      Renvoie un DataFrame vide si le fichier est introuvable ou si les données ne peuvent pas être lues.
    """
    rootbase="/Users/aurelien/Desktop/Stage_M2/PDR/1.7/data/Collisions/"

    if path!= None : 
        rootbase = path
    filename = f"coll_{name}.dat"
    file = rootbase+filename
    
    collision_temperatures = []
    data_lines = []
    start_data_reading = False

    try:
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()

                # Extraire les températures de collision
                if line.startswith('!COLLISION TEMPERATURES'):
                    # La ligne suivante contient les températures
                    line = next(f).strip()
                    # Utiliser re.findall pour capturer tous les nombres flottants dans la ligne
                    temperatures_str = re.findall(r'\d+\.?\d*', line)
                    collision_temperatures = [float(temp) for temp in temperatures_str]

                # Marquer le début des données
                if line.startswith('!NUM'):
                    start_data_reading = True
                    continue # Ne pas traiter cette ligne comme une ligne de données

                if start_data_reading and line:
                    data_lines.append(line)

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file}' n'a pas été trouvé.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture du fichier : {e}")
        return pd.DataFrame()

    if not collision_temperatures:
        print("Avertissement : Aucune température de collision trouvée dans le fichier. Impossible de nommer les colonnes de taux.")
        return pd.DataFrame()

    # Créer les noms de colonnes pour les taux de collision
    collision_rate_columns = [f"{int(temp) if temp == int(temp) else temp}K_PDR" for temp in collision_temperatures]

    # Définir les noms des colonnes initiales
    initial_columns = ['NUM', 'UP', 'LOW']

    # Combiner tous les noms de colonnes
    all_columns = initial_columns + collision_rate_columns

    # Traiter les lignes de données et créer le DataFrame
    df_data = []
    for line in data_lines:
        # Diviser la ligne en utilisant les espaces comme délimiteurs
        parts = line.split()
        if len(parts) >= len(all_columns): # S'assurer qu'il y a suffisamment de données
            # Les 3 premières parties sont des entiers
            num_up_low = [int(p) for p in parts[:3]]
            # Les parties restantes sont des taux de collision (floats)
            rates = [float(p) for p in parts[3:3+len(collision_rate_columns)]]
            df_data.append(num_up_low + rates)
        else:
            print(f"Avertissement: Ligne ignorée en raison d'un nombre insuffisant de colonnes: {line}")

    df = pd.DataFrame(df_data, columns=all_columns)

    return df


def merge_level_data(df_transitions: pd.DataFrame, 
                     df_level: pd.DataFrame, 
                     Col_transi,
                     Level_transi) -> pd.DataFrame:
    """
    Fusionne un DataFrame de transitions avec un DataFrame de niveaux pour ajouter
    les informations de v et J pour les niveaux supérieur (Nup) et inférieur (Nlow).

    Args:
        df_transitions (pd.DataFrame): Le DataFrame principal contenant les transitions
                                       (doit inclure 'Nup' et 'Nlow').
        df_level (pd.DataFrame): Le DataFrame des niveaux (doit inclure 'n', 'v', 'J').

    Returns:
        pd.DataFrame: Un nouveau DataFrame fusionné avec les colonnes 'vu', 'Ju', 'vl', 'Jl' ajoutées.
                      Retourne un DataFrame vide si les colonnes nécessaires sont manquantes.
    """
    # Vérification des colonnes nécessaires
    required_transitions_cols = Col_transi#['Nup', 'Nlow']
    required_level_cols = Level_transi#['n', 'v', 'J']

    if not all(col in df_transitions.columns for col in required_transitions_cols):
        print(f"Erreur: df_transitions doit contenir les colonnes {required_transitions_cols}.")
        return pd.DataFrame()

    if not all(col in df_level.columns for col in required_level_cols):
        print(f"Erreur: df_level doit contenir les colonnes {required_level_cols}.")
        return pd.DataFrame()

    # --- Étape 1: Joindre les informations pour le niveau SUPÉRIEUR (Nup) ---
    # Renommer les colonnes de df_level pour la jointure avec Nup
    df_level_for_Nup = df_level[Level_transi].rename(
        columns={'n': Col_transi[0], 'v': 'vu', 'J': 'Ju'}
    )

    # Fusionner sur Nup
    df_merged = pd.merge(
        df_transitions,
        df_level_for_Nup,
        on= Col_transi[0],
        how='left' # Utilise left join pour conserver toutes les transitions
    )

    # --- Étape 2: Joindre les informations pour le niveau INFÉRIEUR (Nlow) ---
    # Renommer les colonnes de df_level pour la jointure avec Nlow
    df_level_for_Nlow = df_level[Level_transi].rename(
        columns={'n': Col_transi[1], 'v': 'vl', 'J': 'Jl'}
    )

    # Fusionner sur Nlow avec le DataFrame déjà fusionné
    df_final = pd.merge(
        df_merged,
        df_level_for_Nlow,
        on= Col_transi[1],
        how='left' # Utilise left join pour conserver toutes les transitions
    )

    return df_final  



def Lines_to_pd(mol="h2",version="Old",Info=False):
    """
    Returns a DataFrame with all line transitions from a specific molecule.
    Works mainly for h2.

    Args:
        mol (str): The molecule name (e.g., "h2"). This determines the filename.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed line data.
    """
    if version == "Old":
        rootbase = "/Users/aurelien/Desktop/Stage_M2/PDR/1.7/data/Lines/"
    elif version =="HITRAN": 
        rootbase = "/Users/aurelien/Desktop/PhD/PDR/Test_UVCO/data_HITRAN/Lines/"
        #"/Users/aurelien/Desktop/PhD/PDR/Test_PDR7_HITRAN/data/Lines/"

    namebase = f"line_{mol}"
    file_path = os.path.join(rootbase, namebase + ".dat")

    data = []
    # Noms des colonnes pour les numéros quantiques (sans préfixe)
    base_column_names = ['n_trans', 'nu', 'nl', 'E_K', 'Aij_s-1', 'vu', 'Ju', 'vl', 'Jl']
    
    # Noms des colonnes spécifiques pour h2 avec Lambda
    h2_column_names = base_column_names + ['Description_text', 'Lambda_micrometres']
    
    # Noms des colonnes généraux pour les autres molécules
    other_mol_column_names = base_column_names + ['Description']

    with open(file_path, 'r') as f:
        # Saute les 4 premières lignes d'en-tête/commentaires
        for _ in range(4):
            next(f)
        
        for line in f:
            line = line.strip()
            if not line: # Saute les lignes vides
                continue
            
            parts = line.split()
            
            try:
                # Données numériques de base
                n_trans = int(parts[0])
                nu = int(parts[1])
                nl = int(parts[2])
                E_K = float(parts[3])
                Aij_s1 = float(parts[4])

                # Trouver les sections 'quant:' et 'info:'
                quant_idx = parts.index('quant:')
                info_idx = parts.index('info:')
                
                # Extraire les numéros quantiques, supprimer les points-virgules
                vu = int(parts[quant_idx + 1].replace(';', ''))
                Ju = int(parts[quant_idx + 2].replace(';', ''))
                vl = int(parts[quant_idx + 3].replace(';', ''))
                Jl = int(parts[quant_idx + 4].replace(';', ''))

                if mol == "h2":
                    # Pour h2, la section 'info:' a 'Lambda' puis une partie 'Description'
                    info_string = ' '.join(parts[info_idx + 1:])
                    
                    if ';' in info_string:
                        lambda_desc_parts = info_string.split(';', 1)
                        lambda_str = lambda_desc_parts[0].strip().replace('micrometres', '')
                        description_text = lambda_desc_parts[1].strip() if len(lambda_desc_parts) > 1 else ""
                        
                        try:
                            lambda_val = float(lambda_str)
                        except ValueError:
                            lambda_val = np.nan # Utilise np.nan (nécessite numpy) si la conversion échoue
                    else:
                        lambda_val = np.nan
                        description_text = info_string

                    data.append([n_trans, nu, nl, E_K, Aij_s1, vu, Ju, vl, Jl, description_text, lambda_val])
                else:
                    # Pour les autres molécules, la description est tout après 'info:'
                    description = ' '.join(parts[info_idx + 1:]).strip()
                    data.append([n_trans, nu, nl, E_K, Aij_s1, vu, Ju, vl, Jl, description])

            except (ValueError, IndexError) as e:
                if Info == True : 
                    print(f"Skipping malformed or incomplete line: {line} - Error: {e}")
                continue

    if mol == "h2":
        df_lines = pd.DataFrame(data, columns=h2_column_names)
    else:
        df_lines = pd.DataFrame(data, columns=other_mol_column_names)
    
    # Assurer que les colonnes numériques ont les bons types
    df_lines['n_trans'] = df_lines['n_trans'].astype(int)
    df_lines['nu'] = df_lines['nu'].astype(int)
    df_lines['nl'] = df_lines['nl'].astype(int)
    df_lines['E_K'] = df_lines['E_K'].astype(float)
    df_lines['Aij_s-1'] = df_lines['Aij_s-1'].astype(float)
    df_lines['vu'] = df_lines['vu'].astype(int)
    df_lines['Ju'] = df_lines['Ju'].astype(int)
    df_lines['vl'] = df_lines['vl'].astype(int)
    df_lines['Jl'] = df_lines['Jl'].astype(int)

    # Pour h2, s'assurer que Lambda est un float
    if mol == "h2":
        # pd.to_numeric avec errors='coerce' convertira les valeurs non numériques en NaN
        df_lines['Lambda_micrometres'] = pd.to_numeric(df_lines['Lambda_micrometres'], errors='coerce')
    df_lines = add_branch_label(df_lines)
    return df_lines



def Lines_to_pd2(mol="h2",path = "/Users/aurelien/Desktop/Stage_M2/PDR/1.7/data/Lines/"):
    """
    Returns a DataFrame with all line transitions from a specific molecule.
    Works mainly for h2.

    Args:
        mol (str): The molecule name (e.g., "h2"). This determines the filename.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed line data.
    """

    rootbase = path

    namebase = f"line_{mol}"
    file_path = os.path.join(rootbase, namebase + ".dat")

    data = []
    # Noms des colonnes de base pour les données numériques et les numéros quantiques
    base_column_names = ['n_trans', 'nu', 'nl', 'E_K', 'Aij_s-1', 'vu', 'Ju', 'vl', 'Jl']

    with open(file_path, 'r') as f:
        # Saute les 4 premières lignes d'en-tête/commentaires
        for _ in range(4):
            next(f)
        
        for line in f:
            line = line.strip()
            if not line: # Saute les lignes vides
                continue
            
            parts = line.split()
            
            try:
                # Données numériques de base
                n_trans = int(parts[0])
                nu = int(parts[1])
                nl = int(parts[2])
                E_K = float(parts[3])
                Aij_s1 = float(parts[4])

                # Trouver les sections 'quant:' et 'info:'
                quant_idx = parts.index('quant:')
                info_idx = parts.index('info:')
                
                # Extraire les numéros quantiques, supprimer les points-virgules
                vu = int(parts[quant_idx + 1].replace(';', ''))
                Ju = int(parts[quant_idx + 2].replace(';', ''))
                vl = int(parts[quant_idx + 3].replace(';', ''))
                Jl = int(parts[quant_idx + 4].replace(';', ''))

                if mol == "h2":
                    # Pour h2, la section 'info:' contient la Lambda et une description textuelle
                    info_string = ' '.join(parts[info_idx + 1:])
                    
                    lambda_val = np.nan
                    description_text = info_string # Valeur par défaut
                    
                    if ';' in info_string:
                        # Divise par le premier point-virgule pour séparer Lambda de la description
                        lambda_desc_parts = info_string.split(';', 1)
                        lambda_str = lambda_desc_parts[0].strip().replace('micrometres', '').strip()
                        
                        try:
                            lambda_val = float(lambda_str)
                        except ValueError:
                            lambda_val = np.nan # Si la conversion de Lambda échoue
                        
                        if len(lambda_desc_parts) > 1:
                            description_text = lambda_desc_parts[1].strip()
                    
                    data.append([n_trans, nu, nl, E_K, Aij_s1, vu, Ju, vl, Jl, description_text, lambda_val])
                else:
                    # Pour les autres molécules, la section 'info:' a une description et potentiellement une fréquence en GHz
                    description_parts = parts[info_idx + 1:]
                    
                    frequency_ghz = np.nan
                    description_text = ' '.join(description_parts).strip() # Valeur par défaut
                    
                    # Vérifier si la dernière partie ressemble à une fréquence en GHz
                    if description_parts and "GHz" in description_parts[-1]:
                        freq_str = description_parts[-1].replace("GHz", "").strip()
                        try:
                            frequency_ghz = float(freq_str)
                            # La description est tout sauf la dernière partie si c'est une fréquence
                            description_text = ' '.join(description_parts[:-1]).strip()
                        except ValueError:
                            # Si la conversion échoue, la dernière partie n'est pas une fréquence valide
                            pass # frequency_ghz reste np.nan, description_text reste l'intégralité
                    
                    data.append([n_trans, nu, nl, E_K, Aij_s1, vu, Ju, vl, Jl, description_text, frequency_ghz])

            except (ValueError, IndexError) as e:
                print(f"Skipping malformed or incomplete line: {line} - Error: {e}")
                continue

    if mol == "h2":
        # Noms des colonnes spécifiques pour h2
        final_column_names = base_column_names + ['Description_text', 'Lambda_micrometres']
    else:
        # Noms des colonnes pour les autres molécules avec fréquence
        final_column_names = base_column_names + ['Description_text', 'Frequency_GHz']
        
    df_lines = pd.DataFrame(data, columns=final_column_names)
    
    # Assurer que les colonnes numériques ont les bons types
    df_lines['n_trans'] = df_lines['n_trans'].astype(int)
    df_lines['nu'] = df_lines['nu'].astype(int)
    df_lines['nl'] = df_lines['nl'].astype(int)
    df_lines['E_K'] = df_lines['E_K'].astype(float)
    df_lines['Aij_s-1'] = df_lines['Aij_s-1'].astype(float)
    df_lines['vu'] = df_lines['vu'].astype(int)
    df_lines['Ju'] = df_lines['Ju'].astype(int)
    df_lines['vl'] = df_lines['vl'].astype(int)
    df_lines['Jl'] = df_lines['Jl'].astype(int)

    # Convertir la colonne spécifique au bon type (déjà géré par pd.DataFrame si np.nan est float)
    if mol == "h2":
        # pd.to_numeric avec errors='coerce' convertira les valeurs non numériques en NaN
        df_lines['Lambda_micrometres'] = pd.to_numeric(df_lines['Lambda_micrometres'], errors='coerce')
    else:
        df_lines['Frequency_GHz'] = pd.to_numeric(df_lines['Frequency_GHz'], errors='coerce')
        
    df_lines = add_branch_label(df_lines)


    return df_lines



#------------------------------------------------------------------------------
def prolonge_collision_inT(df1, colonnes,list_T,version="log"):
    """
    Prolonge les lignes d'un DataFrame avec une fonction logarithmique.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        colonnes (list): Liste des noms de colonnes.

    Returns:
        pd.DataFrame: DataFrame prolongé.
    """
    df = df1.copy()
    temperatures_colonnes = colonnes[3:]
    temperatures = [float(col.replace('K', '')) for col in temperatures_colonnes]
    
    #---------------------------------------------------
    def fonction_log(x, a, b, c):
        return a * np.log(x) + b * x + c

    def fonction_log10(x, a, b, c):
        return a * np.log10(x) + b * x + c    
    
    if list_T ==None : 
        nouvelles_temperatures = [5000.0, 7000.0, 10000.0, 15000.0, 20000.0]
    else : 
        nouvelles_temperatures=list_T
        
    nouvelles_colonnes = [str(temp) + 'K' for temp in nouvelles_temperatures]

    for nouvelle_colonne in nouvelles_colonnes:
        df[nouvelle_colonne] = np.nan

    for index, row in df.iterrows():
        y_data = row[temperatures_colonnes].astype(float)
        
        
        if version =="log" : 
            popt, _ = curve_fit(fonction_log, temperatures, y_data, p0=[1, 1, 1])

            for temp, nouvelle_colonne in zip(nouvelles_temperatures, nouvelles_colonnes):
                df.at[index, nouvelle_colonne] = fonction_log(temp, *popt)
                
        elif version =="log10" : 
            popt, _ = curve_fit(fonction_log10, temperatures, y_data, p0=[1, 1, 1])

            for temp, nouvelle_colonne in zip(nouvelles_temperatures, nouvelles_colonnes):
                df.at[index, nouvelle_colonne] = fonction_log10(temp, *popt)

    return df







# =============================================================================
#                               Read data files 
# =============================================================================


def mask_by_v(df,colv="vu",maxv=8):
    """Returns df masks by v or another column stated in colv"""
    masks_v=[]
    for i in range (df.iloc[-1][colv]+1): #(maxv): #(df.iloc[-1][colv]+1): 
        masks_v.append( df[colv]==i)
    return(masks_v)




# =============================================================================
#                               plot data files 
# =============================================================================

# Plot collision
#Plot line 










# =============================================================================
#                               Output PDR 
# =============================================================================

#Create Grid 
#Create model 

def write_Obs_file(df, nom_fichier, colLine, colInt, colErr):
    """
    Crée un fichier avec un nom spécifique et écrit les données de trois colonnes
    d'un DataFrame dans le format spécifié.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        nom_fichier (str): Le nom du fichier à créer.
        colLine (str): Le nom de la première colonne (Lines).
        colInt (str): Le nom de la deuxième colonne (Intensity).
        colErr (str): Le nom de la troisième colonne (Err).
    """
    if os.path.exists(nom_fichier):
        raise FileExistsError(f"Le fichier '{nom_fichier}' existe déjà.")
        
    with open(nom_fichier, 'w') as f:
        # Écriture de l'en-tête
        f.write("Lines, Intensity (erg cm-2 s-1 A-1), Err\n")

        # Écriture des données
        for index, row in df.iterrows():
            ligne = f"{row[colLine]}, {row[colInt]:.4e}, {row[colErr]:.4e}\n"
            f.write(ligne)