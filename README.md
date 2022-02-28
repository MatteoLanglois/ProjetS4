# Projet Semestre 4 :
## Création d'une intelligence artificielle de classification d'images

# Introduction
* IA de détection de masque (Avec, sans ou mal mis)
* Plusieurs modèles de classification : DeepLearning et une IA non basée sur le DeepLearning
* Utilisation de la librairie TensorFlow
* Dans recherches : différentes pistes pour le DL/Feature extraction
* Ainsi que un programme pour vérifier l'utilisation de la carte graphique

# Utilisation
* Utiliser le VirtualEnvironnement python (venv)
* Avoir le dataset dans la racine du projet (https://www.kaggle.com/rjouba/dataset)
* Exécuter le script python "MaskDL1.py"
* L'entraînement et l'exploitation sont actuellement fait dans le même script
* Pour ajouter d'autres images pour l'exploitation, il suffit de les ajouter dans le dossier "input" (maximum 28 images)

# A faire
- [x] Trouver un sujet et un dataset
- [x] Réaliser un premier modèle de classification via DeepLearning
- [ ] Faire des histogrammes pour chaque images pour voir les différentes classes prédites
- [ ] Réaliser un second modèle de classification via DeepLearning pour comparer avec le premier
- [ ] Réaliser un troisième modèle de classification via une IA non basée sur le DeepLearning
- [ ] Trouver une méthode pour faire une classification sans DeepLearning
- [ ] Faire un programme qui supprimes les images qui ne fonctionnent pas
- [ ] Trouver comment sauvegarder un modèle pour ensuite pour prédire des images dans un autre script
# Bibliographie
## Tutos utilisés :
* https://www.tensorflow.org/tutorials/images/classification
* https://www.tensorflow.org/tutorials/images/data_augmentation
* https://www.tensorflow.org/tutorials/quickstart/beginner
* https://github.com/chandrikadeb7/Face-Mask-Detection
* https://debuggercafe.com/image-classification-using-tensorflow-on-custom-dataset/
* https://penseeartificielle.fr/installer-facilement-tensorflow-gpu-sous-windows/
## DataSet :
* https://www.kaggle.com/rjouba/dataset