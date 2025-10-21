#!/bin/bash

# Nom de la commande à exécuter
commande="./idat extract"

# Nom du fichier contenant les noms de fichiers
fichier_noms="b.dat"


# Boucle pour lire chaque ligne du fichier
while IFS= read -r nom_fichier; do
  # Vérifier si la ligne n'est pas vide
  if [ -n "$nom_fichier" ]; then
    # Construire la commande complète
    commande_complete="$commande \"$nom_fichier\" output.txt"

    echo "Exécution de : $commande_complete"
    eval "$commande_complete"

    # Vérification du statut de la commande
    if [ $? -ne 0 ]; then
      echo "Erreur lors de l'exécution de la commande pour $nom_fichier"
    fi
  fi
done < "$fichier_noms"

echo "Traitement terminé."
exit 0