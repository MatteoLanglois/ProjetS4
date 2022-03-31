# Projet Semestre 4 :
## Création d'une intelligence artificielle de classification d'images

# Introduction
* IA de détection de masque (Avec, sans ou mal mis)
* Plusieurs modèles de classification : DeepLearning (et une IA non basée sur le DeepLearning, pas encore finie)
* Utilisation de la librairie TensorFlow
* Dans recherches : différentes pistes pour le DL/Feature extraction
* Ainsi qu'un programme pour vérifier l'utilisation de la carte graphique

# Utilisation
* Cloner le dépôt github
* via le cmd/terminal/shell aller à la racine du projet
* Créer l'environnement virtuel (Sous Windows : ```py -m venv env```, Sous une distribution Linux : virtualenv test)
* Activer l'environnement virtuel Python (Sous windows : ```.\env\Scripts\activate```, sous une distribution Linux : ```source test/bin/activate```)
* Installer les packages nécessaires (```pip install -r requirements.txt```)
* Télécharger le dataset et mettez-le à la racine du projet (https://www.kaggle.com/rjouba/dataset)
* Exécuter le programme qui vous intéresse
## Pour le Deep Learning
* Entraînez votre modèle via le programme DeepLearning\TrainModel1.py ou DeepLearning\TrainModel2.py
* Testez votre modèle via le programme DeepLearning\FaceMaskDetection.py ou DeepLearning\VideoMaskDetection.py
* Changer le programme en fonction du modèle que vous voulez tester
## Pour les modèles basés sur les K plus proches voisins
* Pas encore fait
## Pour les recherches
* Vous pouvez exécuter les différents programmes tant que l'environnement virtuel est actif


# A faire
### Principal :
- [x] Trouver un sujet et un dataset
- [x] Faire des matrices de confusion pour comparer les différents modèles
### Deep Learning
- [x] Réaliser un premier modèle de classification via DeepLearning
- [x] Faire des histogrammes pour chaque image pour voir les différentes classes prédites
- [x] Réaliser un second modèle de classification via DeepLearning pour comparer avec le premier
- [x] Trouver comment sauvegarder un modèle pour ensuite pour prédire des images dans un autre script
- [x] Améliorer le réseau neuronal
- [x] Lors de l'utilisation de la webcam, rogner les différents visages et prédire pour chacun puis afficher proba + prédiction au-dessus de chaque
### KNN
- [ ] Réaliser un troisième modèle de classification via une IA non basée sur le DeepLearning
- [x] Trouver une méthode pour faire une classification sans DeepLearning (KNN)
- [ ] Trouver comment avoir un taux de confiance pour la détection des parties du visage
- [ ] Trouver comment faire une prédiction pour une image
### Autres
- [ ] Faire un programme qui supprime les images qui ne fonctionnent pas
- [x] Créer un environnement virtuel 
- [x] Faire en sorte de pouvoir réutiliser l'environnement virtuel via la création d'un requirements.txt

# Bibliographie
## Tutos utilisés :
* https://www.tensorflow.org/tutorials/images/classification
* https://www.tensorflow.org/tutorials/images/data_augmentation
* https://www.tensorflow.org/tutorials/quickstart/beginner
* https://github.com/chandrikadeb7/Face-Mask-Detection
* https://debuggercafe.com/image-classification-using-tensorflow-on-custom-dataset/
* https://penseeartificielle.fr/installer-facilement-tensorflow-gpu-sous-windows/
* etc
## DataSet :
* https://www.kaggle.com/rjouba/dataset