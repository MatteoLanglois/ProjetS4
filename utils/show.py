"""Importation des bibliothèques nécessaires"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


def show(image_paths, predictions=None, class_names=None, type=None):
    # Création d'un dictionnaire pour la matrice de confusion
    MatriceConf = {"Y actuel": [], "Y prédiction": []}
    # Pour chaque image
    for i, image_path in enumerate(image_paths):
        # Sauvegarde de l'image
        orig_image = plt.imread(image_path)
        # Affichage de l'image
        plt.subplot(5, 10, 2 * i + 1)
        plt.imshow(orig_image)
        # Suppression des axes
        plt.axis('off')
        # Si l'on utilise du Deep Learning
        if type == "DL":
            # Affichage de la prédiction et de la valeur de la prédiction
            plt.title(f"{class_names[np.argmax(predictions[i][1])]} ({round(np.max(predictions[i][1]) * 100, 2)}%)")
            plt.subplot(5, 10, 2 * i + 2)
            # Affichage des probabilités via un camembert
            plt.pie(predictions[i][1].numpy() * 100, colors=["#37B0B3", "#4453B3", "#BF46F0"])
            # Ajout des valeurs de la matrice de confusion
            MatriceConf["Y actuel"].append(class_names.index(image_path[8:-6]))
            MatriceConf["Y prédiction"].append(np.argmax(predictions[i][1]))
    # Ajout de la légende pour les probabilités
    if type == "DL":
        plt.legend(class_names, loc="center left", bbox_to_anchor=(1, 0.5))

        plt.subplot(5, 10, 35)
        # Affichage de la matrice de confusion
        df = pd.DataFrame(MatriceConf, columns=['Y actuel', 'Y prédiction'])
        confusion_matrix = pd.crosstab(df['Y actuel'], df['Y prédiction'], rownames=['actuel'], colnames=['prédiction'],
                                       margins=True)
        sn.heatmap(confusion_matrix, annot=True)

        # Affichage de la légende pour la matrice de confusion
        plt.subplot(5, 10, 36)
        plt.axis("off")
        plt.text(0.2, 0.6, f"0 : {class_names[0]}", fontsize=12)
        plt.text(0.2, 0.3, f"1 : {class_names[1]}", fontsize=12)
        plt.text(0.2, 0, f"2 : {class_names[2]}", fontsize=12)

        # Calcul de la précision
        accuracy = sum([MatriceConf['Y actuel'][i] == MatriceConf['Y prédiction'][i] for i in range(0, len(image_paths))]) / len(image_paths) * 100
        # Calcul d'une précision pondérée
        acc_w = 0 # sum([MatriceConf['Y actuel'][i] == MatriceConf['Y prédiction'][i] for i in range(0, len(image_paths))])
        for i, image_path in enumerate(image_paths):
            acc_w += predictions[i][1].numpy()[class_names.index(image_path[8:-6])]
            #if np.argmax(predictions[i][1]) != class_names.index(image_path[8:-6]):
                #acc_w += predictions[i][1].numpy()[class_names.index(image_path[8:-6])]

        print(acc_w)
        accuracy_weighted = round(acc_w / len(image_paths), 3) * 100

        # Affichage de la précision et de la précision pondérée
        plt.subplot(5, 10, 45)
        plt.axis('off')
        plt.text(-0.8, 0.4, f"Précision globale: {round(accuracy, 3)}%", fontsize=20)
        plt.text(-0.8, 0.1, f"Précision pondérée: {round(accuracy_weighted, 3)}%", fontsize=20)

