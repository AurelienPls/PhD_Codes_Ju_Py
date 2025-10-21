#!/bin/bash
#Propose un fichier simple qui exécute un ordre simple 
commande="./idat extract"

path_obs= "/Users/aurelien/Desktop/Stage_M2_LERMA/PDR/Results/Alltgz/out/HHIGRINS_P1e6_G222_BestFit_J20/"
name_obs= "HHIGRINS_P1e6_G222_BestFit_J20_s_20.hdf5"

name_extract= "Extract.txt"

name_out= "Test1.txt"
commande_args=("./idat", "extract", "$path_obs$name_obs", "$name_extract", "$name_out")
commande_complete="${commande_args[@]}" # Pour passer tous les éléments de la liste comme arguments

#commande_complete= "$commande $path_obs$name_obs $name_extract $name_out"

echo "Exécution de la commande d'extraction de $name_obs"
eval "$commande_complete"
