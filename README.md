# Semester 4 Project:
## Creation of Artificial Intelligence for image classification

# Introduction
* Mask detection AI (With, without or misplaced)
* Several classification models: DeepLearning (and a non DeepLearning based AI, not yet finished)
* Use of the TensorFlow library
* In research: different tracks for DL/Feature extraction
* As well as a program to check the use of the graphics card

# Usage
* Clone the GitHub repository
* via cmd/terminal/shell go to the root of the project
* Create the virtual environment (Under Windows: ```py -m venv env```, Under a Linux distribution: virtualenv test)
* Activate the Python virtual environment (On Windows: ```.\env\Scripts\activate```, on Linux distribution: ```source test/bin/activate``)
* Install the necessary packages (```pip install -r requirements.txt```)
* Download the dataset and put it in the root of the project (https://www.kaggle.com/rjouba/dataset)
* Run the program you are interested in
## For Deep Learning
* Train your model via the program DeepLearning\TrainModel1.py or DeepLearning\TrainModel2.py
* Test your model with the DeepLearning\FaceMaskDetection.py or DeepLearning\VideoMaskDetection.py program
* Change the program according to the model you want to test
## For models based on the K nearest neighbors
* Not done yet
## For searches
* You can run the different programs as long as the virtual environment is active


# To do
### Main:
- [x] Find a subject and a dataset
- [x] Make confusion matrices to compare the different models
### Deep Learning
- [x] Make a first classification model via DeepLearning
- [x] Make histograms for each image to see the different predicted classes
- [x] Make a second classification model via DeepLearning to compare with the first one
- [x] Find out how to save a model to predict images in another script
- [x] Improve the neural network
- [x] When using the webcam, crop the different faces and predict for each then display proba + prediction above each
### KNN
- [ ] Perform a third classification model via non-DeepLearning based AI
- [ ] Find a way to do a classification without DeepLearning (KNN)
- [ ] Find a way to have a confidence rate for face part detection
- [ ] Find how to make a prediction for an image
### Other
- [ ] Make a program that removes images that don't work
- [x] Create a virtual environment 
- [x] Make it possible to reuse the virtual environment via the creation of a requirements.txt

# Bibliography
## Tutorials used :
* https://www.tensorflow.org/tutorials/images/classification
* https://www.tensorflow.org/tutorials/images/data_augmentation
* https://www.tensorflow.org/tutorials/quickstart/beginner
* https://github.com/chandrikadeb7/Face-Mask-Detection
* https://debuggercafe.com/image-classification-using-tensorflow-on-custom-dataset/
* https://penseeartificielle.fr/installer-facilement-tensorflow-gpu-sous-windows/
* etc
## DataSet:
* https://www.kaggle.com/rjouba/dataset