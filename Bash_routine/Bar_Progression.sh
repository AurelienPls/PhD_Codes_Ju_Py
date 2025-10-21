#!/bin/bash

# Nombre total d'étapes (à adapter à votre boucle)
total_steps=10
current_step=0

echo -n "Progression : [ ]" # Initialisation de la barre

while [ "$current_step" -lt "$total_steps" ]; do
  # Simuler une tâche qui prend du temps
  sleep 1

  # Incrémenter le compteur d'étapes
  ((current_step++))

  # Calculer le pourcentage de progression
  percentage=$(( (current_step * 100) / total_steps ))

  # Mettre à jour la barre de chargement
  bar_length=$(( (current_step * 20) / total_steps )) # Exemple de longueur de barre (20 caractères)
  bar=""
  for (( i=0; i<bar_length; i++ )); do
    bar="$bar#"
  done

  # Effacer la ligne précédente et afficher la nouvelle progression
echo -ne "\rProgression : [$bar$(printf '%${20-\$bar_length}s' " ")] $percentage%"
done

echo "" # Nouvelle ligne après la fin de la barre
echo "Terminé !"

exit 0