#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:26:42 2025

@author: aurelien
"""

import pandas as pd
import pickle
import pyismtools as ism
import re
import h5py
import numpy as np
import os as os 
"""
chemin_modules = "/Users/aurelien/Desktop/PhD/Codes/Package/Perso/"
sys.path.append(chemin_modules)
import read as read 

"""


def extraire_transition(transition_str):
    """
    Extrait les valeurs vu, Ju, vl et Jl d'une chaîne de transition,
    en tenant compte des branches O, P, Q, R et S.

    Args:
        transition_str (str): Chaîne de transition au format "vu-vl_Branche(Jdown!!)".

    Returns:
        tuple: Un tuple contenant (vu, Ju, vl, Jl).
    """

    # 1. Séparation de la chaîne en parties
    parties = transition_str.split('_')
    niveaux = parties[0].split('-')
    rotation = parties[1]

    # 2. Extraction des valeurs
    vu = int(niveaux[0])
    vl = int(niveaux[1])
    branche = rotation[:1]  # Extraction de la branche (O, P, Q, R, S)
    Jl = int(rotation[2:-1])  # Extraction de la valeur de Ju

    # 3. Calcul de Jl (selon la règle de sélection)
    if   branche == 'O':
        Ju = Jl - 2
    elif branche == 'P':
        Ju = Jl - 1
    elif branche == 'Q':
        Ju = Jl
    elif branche == 'R':
        Ju = Jl + 1
    elif branche == 'S':
        Ju = Jl + 2
    else:
        raise ValueError("Branche de transition invalide (O, P, Q, R, S)")

    return vu, Ju, vl, Jl

#df[['vu', 'Ju', 'vl', 'Jl']] = df['Lines'].apply(lambda x: pd.Series(read.extraire_transition(x)))


def supprimer_espace_apres_parenthese(chaine):
    """
    Supprime l'espace après la parenthèse ouvrante dans une chaîne de caractères.

    Args:
        chaine (str): La chaîne à modifier.

    Returns:
        str: La chaîne modifiée.
    """
    return chaine.replace('( ', '(')




def Nettoyage (x):
    return((x.split(',')[0]))
#------------------------------------------------------------------------------
#                                           Pandas 
#------------------------------------------------------------------------------
def dat_to_pd(path,name,line_data,end_data=None, ColNames=None, Col_Idx=None, tofloat=True, Info=False):
    """
    Generates a dataframe according to the .dat to be easily usable 
    Return a data frame with every columns in float if not specified 
    """
    #assert ColNames!=None and Col_Idx!=None
    doc=open(path+name,"r")
    file=doc.read();  ligne=file.split("\n");  header=ligne[0:line_data-1]
    

    data_file=ligne[line_data:end_data]
    
    if ColNames != None : 
        Column_name = ColNames
        if Info ==True : 
            print(Column_name)
    elif Col_Idx != None : 
        Temp = ligne[Col_Idx].split("  ")
        Column_name = [el.split("]")[1] for el in Temp]

    df=pd.DataFrame([x.split(maxsplit=len(Column_name)) for x in data_file], columns=Column_name)
    if tofloat==True : 
        df[Column_name] = df[Column_name].astype("float")
    
    return(df)



def mask_by_v(df,colv="vu",maxv=10):
    """Returns df masks by v or another column stated in colv"""
    masks_v=[]
    for i in range (int(df.iloc[-1][colv]+1)): #(maxv): #(df.iloc[-1][colv]+1): 
        masks_v.append( df[colv]==i)
    return(masks_v)


def mask_by_unique_val(df, nom_colonne):
    """Créé des masque par valeurs uniques
    Utilisé pour trouver les transitions  provenant d'un meme niveau d'énergie pour calculer les OPR 
    """
    masques = {}
    valeurs_uniques = df[nom_colonne].unique()

    for valeur in valeurs_uniques:
        masque = df[nom_colonne] == valeur
        if masque.sum() >= 2:  # Vérifie si le masque a au moins deux True
            masques[valeur] = masque
    return masques


def write_df_to_dat(df,nom_fichier, separator=",  "):
    with open(nom_fichier, 'w') as f:
        #En tete = noms colones 
        f.write("# " + ", ".join(df.columns) + "\n")

        for _, row in df.iterrows():
            f.write(separator.join(map(str, row.values)) + "\n")
            


    
#------------------------------------------------------------------------------
#                                OPR preparation                           
#------------------------------------------------------------------------------       

def IDAT_to_OPR_for_Av(df: pd.DataFrame, av_value: float, n_cols: list) -> pd.DataFrame:
    """
    Transforms a DataFrame for a given Av value, extracting v, J, and N_mod from column names (efficient).

    Args:
        df: The input DataFrame.
        av_value: The Av value to filter by.
        n_cols: A list of column names representing N(h) values (e.g., 'n(H2 v=1,J=2)').

    Returns:
        A new DataFrame with 'Line', 'vu', 'Ju', and 'N_mod' columns.
    """
    filtered_df = df[df['AV'] == av_value].copy()
    
    data = []
    for col in n_cols:
        match = re.search(r'v=(\d+),J=(\d+)', col)
        if match:
            data.append({
                'Line': match.group(0),
                'v': int(match.group(1)),
                'J': int(match.group(2)),
                'N_mod': filtered_df.at[filtered_df.index[0], col]  # Accessing single value efficiently
            })
        else:
            data.append({
                'Line': None,
                'vu': None,
                'Ju': None,
                'N_mod': None
            })
            
    new_df = pd.DataFrame(data)
    return new_df





def identifier_ortho_para(df,col="Ju"):
    """
    Ajoute une nouvelle colonne 'is_ortho_para' à un DataFrame
    indiquant si le niveau est ortho ou para, en détectant la colonne 'Ju'.

    Args:
        df (pd.DataFrame): Le DataFrame à modifier.

    Returns:
        pd.DataFrame: Le DataFrame avec la nouvelle colonne 'is_ortho_para'.
                      Retourne None si la colonne 'Ju' n'est pas trouvée.
    """
    if col not in df.columns:
        print(f"Erreur : La colonne {col} n'a pas été trouvée dans le DataFrame.")
        return None

    def determine_ortho_para(row):
        j_val = row[col]
        # Vous devrez adapter cette logique en fonction de la convention
        # que vous utilisez pour distinguer ortho et para.
        # Voici un exemple basé sur la parité de Ju :
        if isinstance(j_val, (int, float)):
            if j_val % 2 == 0:
                return 'para'
            else:
                return 'ortho'
        else:
            return 'inconnu'  # Si Ju n'est pas un nombre

    df['ortho_para_type'] = df.apply(determine_ortho_para, axis=1)
    return df



def df_mean_by_col(df1, colonnes_vu_ju=('vu', 'Ju'), col_to_mean=['int']):
    """
    Crée un nouveau DataFrame en calculant la moyenne des colonnes spécifiées dans col_to_mean
    pour chaque combinaison unique de valeurs dans les colonnes spécifiées pour 'vu' et 'Ju' de df1.

    Args:
        df1 (pd.DataFrame): Le DataFrame d'origine.
        colonnes_vu_ju (tuple): Un tuple contenant les noms des colonnes représentant 'vu' et 'Ju'. Par défaut ('vu', 'Ju').
        col_to_mean (list): Une liste des noms de colonnes dont la moyenne doit être calculée. Par défaut ['int'].

    Returns:
        pd.DataFrame: Un nouveau DataFrame avec les colonnes 'vu', 'Ju' et la moyenne de chaque colonne spécifiée dans col_to_mean
                      pour chaque combinaison unique.
        Retourne None si les colonnes spécifiées n'existent pas.
    """
    if not all(col in df1.columns for col in colonnes_vu_ju + tuple(col_to_mean)):
        print(f"Erreur : Les colonnes {colonnes_vu_ju} et {col_to_mean} doivent exister dans le DataFrame d'entrée.")
        return None

    try:
        # Grouper le DataFrame par les colonnes 'vu' et 'Ju'
        grouped = df1.groupby(list(colonnes_vu_ju))

        # Calculer la moyenne de chaque colonne spécifiée dans col_to_mean pour chaque groupe
        df_moyenne = grouped[col_to_mean].mean().reset_index()

        # Renommer les colonnes de la moyenne pour plus de clarté
        nouveaux_noms = {col: f'moyenne_{col}' for col in col_to_mean}
        df_moyenne.rename(columns=nouveaux_noms, inplace=True)

        return df_moyenne

    except Exception as e:
        print(f"Une erreur est survenue lors de la création du DataFrame de moyenne : {e}")
        return None   
    
    
    
 
    
def detect_3consecutifs(df, colonne):
    """
    Détecte si trois valeurs consécutives existent dans une colonne d'un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        colonne (str): Le nom de la colonne à vérifier.

    Returns:
        list: Une liste d'indices (int) des trois lignes consécutives
              sont trouvées, ou une liste vide si
              aucune séquence de trois valeurs consécutives identiques n'est trouvée.
              Retourne None si la colonne spécifiée n'existe pas.
    """
    if colonne not in df.columns:
        print(f"Erreur : La colonne '{colonne}' n'existe pas dans le DataFrame.")
        return None

    n = len(df)
    if n < 3:
        print("Liste pas assez longue pour trois indices consécutifs")
        return None  # Pas assez de lignes pour trois consécutifs
    
    indices_consecutifs = []

    for i in range(n - 2):
        valeur1 = df.iloc[i][colonne]
        valeur2 = df.iloc[i + 1 ][colonne]
        valeur3 = df.iloc[i + 2][colonne]
        if valeur1 +2 == valeur2 +1 == valeur3:
            indices_consecutifs.append([i, i + 1, i + 2])

    return indices_consecutifs

def detect_3consecutifs_V2(df, colonne):
    """
    Détecte si trois valeurs consécutives existent dans une colonne d'un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        colonne (str): Le nom de la colonne à vérifier.

    Returns:
        list: Une liste d'indices (int) des trois lignes consécutives
              sont trouvées, ou une liste vide si
              aucune séquence de trois valeurs consécutives identiques n'est trouvée.
              Retourne None si la colonne spécifiée n'existe pas.
    """
    if colonne not in df.columns:
        print(f"Erreur : La colonne '{colonne}' n'existe pas dans le DataFrame.")
        return None

    n = len(df)
    if n < 3:
        print("Liste pas assez longue pour trois indices consécutifs")
        return None  # Pas assez de lignes pour trois consécutifs
    
    indices_consecutifs = []

    for i in range(0,n):
        valeur2 = df.iloc[i][colonne]

        if valeur2+1 in df[colonne].values and valeur2-1 in df[colonne].values : 
            
            indicesmin = df.index[df[colonne] == valeur2-1].tolist()
            N_min = [df.index.get_loc(index_ligne) for index_ligne in indicesmin]
            indicesmax = df.index[df[colonne] == valeur2+1].tolist()
            N_max = [df.index.get_loc(index_ligne) for index_ligne in indicesmax]

            for idxmin in N_min : 
                for idxmax in N_max : 
                    indices_consecutifs.append([idxmin, i , idxmax])
                    
    if indices_consecutifs :
        return indices_consecutifs


#------------------------------------------------------------------------------
#                                   Data 
#------------------------------------------------------------------------------
def dat_to_data ( path, name, line_data, end_data = None ): 
        
    doc=open(path+name,"r")
    file=doc.read();  ligne=file.split("\n")
    if end_data ==None : 
        end_data = len(ligne)
        
    header=ligne[0:line_data-1]
    data_bloc = ligne[line_data:end_data]
    
    return (header, data_bloc)

#Nettoyage eventuel 


#------------------------------------------------------------------------------
#                                   Pickle 
#------------------------------------------------------------------------------
def read_pickle(name_pkl):
    with open(name_pkl, "rb") as fichier:
        Data = pickle.load(fichier)
    return(Data)


def write_pickle(Data,output_name):
    with open(output_name, "wb") as fichier:
        pickle.dump(Data, fichier)

        
#------------------------------------------------------------------------------
#                                   csv files   
#------------------------------------------------------------------------------ 



        
#------------------------------------------------------------------------------
#                            Line traduction  
#------------------------------------------------------------------------------   
def Traductor (name_dic,Observation,formalism) : 


    name_list = list(Observation.get_observable_names())

    new_name_dict = {}
    if formalism=='Chi_Inta' :
        
        for x in name_list:
            vup,temp= x.split('-')
            vlow,temp2=temp.split('_')
            trtype,temp3=temp2.split('(')
            Jlow=temp3.split(')')[0].split(' ')[1]
            
            if trtype == 'S' :
                Jup = str(int(Jlow)+2)
            elif trtype == 'O' :
                Jup = str(int(Jlow)-2)
            elif trtype == 'Q' :
                Jup = Jlow
                
            new_name_dict[x] = f"inta_00_h2_v{vup}_j{str(Jup)}__v{vlow}_j{Jlow}"
            
        my_translation = ism.Translation(name_dic,new_name_dict)
    
    #print(my_translation)
    #print(my_grid)
    #On sélectionne les raies de l'observations pour comparer ce qui est comparable
    quantity_names = my_translation.get_alternate_names()
    
    
    
    
#------------------------------------------------------------------------------
#                                      
#------------------------------------------------------------------------------ 

def find_value_in_str (motif,texte,exposant=False, style=float): 
    """Trouve la valeur dans un str et l'extrait 
    Exemple : 
        motif = r"_G(\d+)" pour trouver _G200_P5e5 
        
        r"_G(\d+)"
        r"_P(\d+)e(\d+)"
    """
    match = re.search(motif, texte)
    value = style(match.group(1)) if match else None
    if exposant == True : 
        expo = int(match.group(2))
        value = int(match.group(1)) * (10 ** expo)
        
    return (value)

#------------------------------------------------------------------------------
#                                .hdf5                           
#------------------------------------------------------------------------------ 

def explorer_hdf5(file_path):
    """
    Explore un fichier HDF5 et affiche sa structure et quelques informations.

    Args:
        file_path (str): Le chemin d'accès au fichier HDF5.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Structure du fichier HDF5 : {file_path}\n")

            def print_attrs(name, obj):
                print(f"  {name}: {type(obj).__name__}")
                if isinstance(obj, h5py.Dataset):
                    print(f"    Shape: {obj.shape}")
                    print(f"    Datatype: {obj.dtype}")
                    if obj.shape and len(obj.shape) <= 2:
                        print(f"    Exemple de données (premières 5 éléments ou lignes) :")
                        try:
                            if len(obj.shape) == 1 and obj.shape[0] <= 5:
                                print(obj[:])
                            elif len(obj.shape) == 1 and obj.shape[0] > 5:
                                print(obj[:5])
                            elif len(obj.shape) == 2 and obj.shape[0] <= 5:
                                print(obj[:, :5])
                            elif len(obj.shape) == 2 and obj.shape[0] > 5:
                                print(obj[:5, :5])
                        except Exception as e:
                            print(f"    Impossible d'afficher un aperçu des données : {e}")
                elif isinstance(obj, h5py.Group):
                    pass  # Les groupes sont explorés récursivement
                elif isinstance(obj, np.ndarray):
                    print(f"    Type: NumPy Array, Shape: {obj.shape}, dtype: {obj.dtype}")
                    if obj.shape and len(obj.shape) <= 2:
                        print(f"    Exemple de données (premières 5 éléments ou lignes) :")
                        try:
                            if len(obj.shape) == 1 and obj.shape[0] <= 5:
                                print(obj[:])
                            elif len(obj.shape) == 1 and obj.shape[0] > 5:
                                print(obj[:5])
                            elif len(obj.shape) == 2 and obj.shape[0] <= 5:
                                print(obj[:, :5])
                            elif len(obj.shape) == 2 and obj.shape[0] > 5:
                                print(obj[:5, :5])
                        except Exception as e:
                            print(f"    Impossible d'afficher un aperçu des données : {e}")
                elif isinstance(obj, h5py.Datatype):
                    print(f"    Datatype: {obj}")
                else:
                    print(f"    Type non reconnu : {type(obj)}")

            f.visititems(print_attrs)

            print("\nInformations générales sur le fichier :")
            print(f"Nombre de groupes : {len(f)}")
            datasets = [name for name, obj in f.items() if isinstance(obj, h5py.Dataset)]
            print(f"Nombre de datasets : {len(datasets)}")

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file_path}' n'a pas été trouvé.")
    except Exception as e:
        print(f"Une erreur est survenue lors de l'exploration du fichier : {e}")
        
        
        
#------------------------------------------------------------------------------
#                                beetroots                           
#------------------------------------------------------------------------------         
        

def read_beetroots_hdf5 (name,bins = 200 , param_axis=[1,2], t_burnin=100, pixel = 0, seuil = 1) : 
    with h5py.File(name, 'r') as f:
        list_Theta = np.array(f["list_Theta"][t_burnin:, 0, :]) 
        x = np.log10( list_Theta[:, param_axis[0]] )
        y = np.log10 ( list_Theta[:, param_axis[1]] )
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
        hist_masque = np.ma.masked_where(hist < seuil, hist)
        return (10**xedges , 10**yedges , hist_masque.T)
    
    
    



#------------------------------------------------------------------------------
#                                                           
#------------------------------------------------------------------------------   


def process_h2_data_to_pkl(file_path: str, file_name: str, column_to_consider: int, output_filename: str):
    """
    Lit un fichier .dat de données H2, transforme les noms de lignes en noms de colonnes
    spécifiques, extrait les valeurs d'une colonne donnée, et sauvegarde le résultat
    dans un fichier .pkl en tant que DataFrame pandas.

    Args:
        file_path (str): Le chemin absolu du dossier où se trouve le fichier .dat.
        file_name (str): Le nom du fichier .dat (ex: 'NGC2023_full_corrected.dat').
        column_to_consider (int): L'index de la colonne à extraire pour les données.
                                  (0 pour le nom de la ligne, 1 pour le flux, 2 pour l'erreur).
        output_filename (str): Le nom complet du fichier .pkl de sortie (ex: 'mon_output.pkl').
    """
    full_file_path = os.path.join(file_path, file_name)
    
    # --- Lecture du fichier .dat ---
    try:
        with open(full_file_path, 'r') as f:
            file_content = f.read()
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{full_file_path}' n'a pas été trouvé.")
        print("Veuillez vérifier le chemin et le nom du fichier.")
        return # Quitte la fonction si le fichier n'est pas trouvé
    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture du fichier: {e}")
        return

    lines = file_content.strip().split('\n')
    data_values = [] # Contiendra les valeurs extraites de la colonne choisie
    line_names = []  # Contiendra les noms de lignes originaux

    # Parcourir les lignes et extraire les données, en ignorant les commentaires
    for line in lines:
        if line.strip().startswith('#'):
            continue
        
        parts = [p.strip() for p in line.split(',')]
        
        # Extraire le nom de la ligne (toujours la première partie)
        line_name = parts[0]
        line_names.append(line_name)
        
        # Extraire la valeur de la colonne spécifiée
        try:
            # On prend la valeur de la colonne spécifiée par `column_to_consider`
            # En sachant que le nom est en `parts[0]`, le flux en `parts[1]`, l'erreur en `parts[2]`
            # Donc si `column_to_consider` est 1 (flux) ou 2 (erreur), on l'utilise
            value = float(parts[column_to_consider].replace('E-5', 'e-5'))
            data_values.append(value)
        except (ValueError, IndexError) as e:
            print(f"Erreur lors de l'extraction ou de la conversion pour la ligne '{line}' (colonne {column_to_consider}): {e}")
            # Si une erreur se produit pour une ligne, on peut choisir de l'ignorer
            # ou de lever une exception. Ici, on imprime un message et on saute.
            data_values.append(np.nan) # Ajouter NaN pour garder la taille du tableau
            continue

    # --- Logique de renommage des colonnes ---
    new_name_dict = {}
    for x in line_names:
        try:
            match = re.match(r'(\d+)-(\d+)_(Q|S|O)\(\s*(\d+)\)', x)
            if not match:
                print(f"Format de nom de ligne inattendu pour '{x}'. Skipping.")
                continue

            vup_str, vlow_str, trtype, jlow_str = match.groups()
            
            vup = int(vup_str)
            vlow = int(vlow_str)
            Jlow = int(jlow_str)

            Jup = None
            if trtype == 'S':
                Jup = Jlow + 2
            elif trtype == 'O':
                Jup = Jlow - 2
            elif trtype == 'Q':
                Jup = Jlow
            else:
                print(f"Type de transition inconnu '{trtype}' pour '{x}'. Skipping.")
                continue
            
            new_name = f"h2_v{vup}_j{Jup}__v{vlow}_j{Jlow}"
            new_name_dict[x] = new_name
        except Exception as e:
            print(f"Erreur lors du traitement de '{x}': {e}")
            continue

    # Créez le DataFrame pandas
    # Il est important de s'assurer que les valeurs et les noms de colonnes correspondent.
    # On filtre les noms de lignes originaux pour ne garder que ceux qui ont été renommés avec succès.
    final_column_names = []
    final_data_values = []
    
    for i, original_name in enumerate(line_names):
        if original_name in new_name_dict:
            final_column_names.append(new_name_dict[original_name])
            final_data_values.append(data_values[i]) # Assurez-vous d'utiliser la valeur correspondante

    df_output = pd.DataFrame([final_data_values], columns=final_column_names)

    # Définir 'X' et 'Y' comme MultiIndex
    df_output['X'] = 0
    df_output['Y'] = 0
    df_output = df_output.set_index(['X', 'Y'])

    # Afficher le DataFrame (pour vérification)
    print("\nDataFrame créé:")
    print(df_output)

    # --- Sauvegarder le DataFrame dans un fichier pickle ---
    try:
        with open(output_filename, 'wb') as f:
            pickle.dump(df_output, f)
        print(f"\nLe DataFrame a été sauvegardé avec succès dans '{output_filename}'")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier pickle '{output_filename}': {e}")
        
        

def read_IDAT_Nu(file_path):
    """
    Parses a text file containing molecular data, extracts relevant columns,
    converts them to appropriate types, and further separates molecular
    information (molecule name, v, J) into new columns.

    Args:
        file_path (str): The path to the input text file.

    Returns:
        pd.DataFrame: A DataFrame with columns:
                      'index' (Int64), 'value' (float), 'unit' (str),
                      'quantity_raw' (str, original quantity string),
                      'molecule' (str), 'v' (Int64), 'J' (Int64).
                      'v' and 'J' will be pd.NA if not applicable (e.g., for 'N(CO)').
                      Returns an empty DataFrame if parsing fails or no relevant data is found.
    """
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue # Skip empty lines and comment/metadata lines


            idx,val,hashtag,unit,qt = line.split(maxsplit=4)

                # Regex to extract molecule, v, and J (e.g., N(CO v=0,J=12))
            match = re.match(r'N\((\w+)\s+v=(\d+),J=(\d+)\)', qt)
            if match:
                mol = match.group(1)
                v_level = int(match.group(2))
                J_level = int(match.group(3))
            else:
                mol_only_match = re.match(r'N\((\w+)\)', qt)
                v_level=None
                if mol_only_match:
                    mol = mol_only_match.group(1)
                else:
                    mol = qt # Store the raw string if nothing else matches
            if v_level!=None :
                data.append({
                    'index': idx,
                    'value': val,
                    'unit': unit,
                    'quantity_raw': qt, # Keep the original string for reference
                    'molecule': mol,
                    'v': v_level,
                    'J': J_level
                })

    if not data:
        print("No valid data rows found in the file.")
        return pd.DataFrame(columns=['index', 'value', 'unit', 'quantity_raw', 'molecule', 'v', 'J'])

    df = pd.DataFrame(data)
    
    # Convert 'index', 'v', 'J' to nullable integers (Int64)
    # pd.NA allows NaN-like values in integer columns
    df['index'] = df['index'].astype('Int64')
    df['v'] = pd.to_numeric(df['v'], errors='coerce').astype('Int64')
    df['J'] = pd.to_numeric(df['J'], errors='coerce').astype('Int64')
    
    # Ensure 'value' is float and 'unit', 'quantity_raw', 'molecule' are strings
    df['value'] = df['value'].astype(float)
    df['unit'] = df['unit'].astype(str)
    df['quantity_raw'] = df['quantity_raw'].astype(str)
    df['molecule'] = df['molecule'].astype(str) # Convert pd.NA to 'NA' string if it was assigned

    return df



def Int_to_df(lines, intensities):
    data = []
    
    for line_str, intensity in zip(lines, intensities):
        match = re.match(r'(\w+)\s+v(\d+)-(\d+)\s+([RPQ])\((\d+)\)', line_str)
        if match:
            molecule = match.group(1)
            v_upper = int(match.group(2))
            v_lower = int(match.group(3))
            branch = match.group(4)
            J_initial = int(match.group(5))

            J_upper = None
            J_lower = None

            if branch == 'R':
                J_upper = J_initial + 1
                J_lower = J_initial
            elif branch == 'P':
                J_upper = J_initial - 1
                J_lower = J_initial
            elif branch == 'Q':
                J_upper = J_initial
                J_lower = J_initial

            data.append({
                'Lines': line_str,
                'Int': intensity,
                'Molecule': molecule,
                'vu': v_upper,
                'vl': v_lower,
                'Ju': J_upper,
                'Jl': J_lower
            })
    
    df = pd.DataFrame(data)
    df['vu'] = df['vu'].astype('Int64')
    df['vl'] = df['vl'].astype('Int64')
    df['Ju'] = df['Ju'].astype('Int64')
    df['Jl'] = df['Jl'].astype('Int64')
    
    return df



def get_j_upper(row):
    """Calcule J_upper basé sur le nom de la ligne (Notation Herzberg)
    Prend une liste str de format 1-0_P( 4)  
    retourne vu-vl , Ju, Jl à stocker dans trois listes ou colonnes d'un df"""
    try:
        name = row['Line']
        # Format attendu : "1-0_P( 4)" ou "2-1_R( 10)"
        parts = name.split('_')
        band = parts[0]     # "1-0"
        branch_part = parts[1] # "P( 4)"
        
        branch = branch_part[0] # "P" ou "R"
        # Extraction du nombre entre parenthèses
        j_lower = int(branch_part.split('(')[1].split(')')[0])
        
        # Règle spectroscopique pour trouver J_upper
        if branch == 'P':
            j_upper = j_lower - 1
        elif branch == 'R':
            j_upper = j_lower + 1
        else:
            return None, None, None
            
        return band, j_upper, branch
    except:
        return None, None, None
    
    
def parse_line_properties(line_str):
    """
    Extrait vu, Ju, vl, Jl d'une chaîne au format Herzberg (ex: '1-0_P( 4)').
    Gère les branches P, R, Q, S, O.
    
    Retourne:
        pd.Series: [vu, Ju, vl, Jl]
    """
    try:
        # Vérification basique
        if not isinstance(line_str, str):
            return pd.Series([None, None, None, None])

        # Séparation Bande / Branche (ex: "1-0" et "P( 4)")
        parts = line_str.strip().split('_')
        if len(parts) != 2:
            return pd.Series([None, None, None, None])
        
        band_part = parts[0]   # "1-0"
        branch_part = parts[1] # "P( 4)"

        # 1. Niveaux Vibrationnels (vu, vl)
        v_levels = band_part.split('-')
        vu = int(v_levels[0])
        vl = int(v_levels[1])

        # 2. Niveaux Rotationnels (Jl) et Branche
        branch = branch_part[0].upper()
        # Extraction propre du chiffre entre parenthèses
        jl_str = branch_part.split('(')[1].split(')')[0]
        Jl = int(jl_str)

        # 3. Calcul de Ju selon la règle de sélection
        if branch == 'P':     # Delta J = -1
            Ju = Jl - 1
        elif branch == 'R':   # Delta J = +1
            Ju = Jl + 1
        elif branch == 'Q':   # Delta J = 0
            Ju = Jl
        elif branch == 'S':   # Delta J = +2 (Typique H2)
            Ju = Jl + 2
        elif branch == 'O':   # Delta J = -2 (Typique H2)
            Ju = Jl - 2
        else:
            return pd.Series([None, None, None, None])

        return pd.Series([vu, Ju, vl, Jl], index=['vu', 'Ju', 'vl', 'Jl'])

    except Exception:
        return pd.Series([None, None, None, None], index=['vu', 'Ju', 'vl', 'Jl'])    