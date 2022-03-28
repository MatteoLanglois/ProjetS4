import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


def show(image_paths, predictions=None, class_names=None, type=None):
    MatriceConf = {"Y actuel": [], "Y prédiction": []}
    for i, image_path in enumerate(image_paths):
        orig_image = plt.imread(image_path)
        plt.subplot(5, 10, 2 * i + 1)
        plt.imshow(orig_image)
        plt.axis('off')
        if type == "DL":
            plt.title(f"{class_names[np.argmax(predictions[i][1])]} ({round(np.max(predictions[i][1]) * 100, 2)}%)")
            plt.subplot(5, 10, 2 * i + 2)
            plt.pie(predictions[i][1].numpy() * 100, colors=["#37B0B3", "#4453B3", "#BF46F0"])
            MatriceConf["Y actuel"].append(class_names.index(image_path[8:-6]))
            MatriceConf["Y prédiction"].append(np.argmax(predictions[i][1]))
    if type == "DL":
        plt.legend(class_names, loc="center left", bbox_to_anchor=(1, 0.5))

        plt.subplot(5, 10, 35)
        df = pd.DataFrame(MatriceConf, columns=['Y actuel', 'Y prédiction'])
        confusion_matrix = pd.crosstab(df['Y actuel'], df['Y prédiction'], rownames=['actuel'], colnames=['prédiction'],
                                       margins=True)
        sn.heatmap(confusion_matrix, annot=True)

        plt.subplot(5, 10, 36)
        plt.axis("off")
        plt.text(0.2, 0.6, f"0 : {class_names[0]}", fontsize=12)
        plt.text(0.2, 0.3, f"1 : {class_names[1]}", fontsize=12)
        plt.text(0.2, 0, f"2 : {class_names[2]}", fontsize=12)

        accuracy = sum([confusion_matrix.iloc[i, i] for i in range(0, 3)]) / len(image_paths) * 100

        acc_w = 0
        for i, image_path in enumerate(image_paths):
            if np.argmax(predictions[i][1]) == class_names.index(image_path[8:-6]):
                acc_w += 1
            else:
                acc_w += predictions[i][1].numpy()[class_names.index(image_path[8:-6])]

        accuracy_weighted = round(acc_w / len(image_paths), 3) * 100

        plt.subplot(5, 10, 45)
        plt.axis('off')
        plt.text(-0.8, 0.4, f"Précision globale: {round(accuracy, 3)}%", fontsize=20)
        plt.text(-0.8, 0.1, f"Précision pondérée: {round(accuracy_weighted, 3)}%", fontsize=20)
