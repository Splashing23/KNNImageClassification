import numpy as np
import os
import cv2
import random
from math import inf


class NN:

    def __init__(self) -> None:
        self.params = []

    # Create duplicate dir with images resized to 1000 x 1000
    def ttSplit(self, classes_dir="cropped_images-20240423T205638Z-001/cropped_images") -> None:
        self.classes_dir = "cropped_images-20240423T205638Z-001/cropped_images"
        self.new_train_dir = "modifiedTrain"
        self.new_test_dir = "modifiedTest"

        # If new dir doesn't already exist...
        if not os.path.exists(self.new_train_dir) and not os.path.exists(self.new_test_dir):
            os.makedirs(self.new_train_dir)
            os.makedirs(self.new_test_dir)
            
            # Iterate through class folders
            for classDir in os.listdir(self.classes_dir):

                # Copy class folders to new dir
                os.makedirs(f"{self.new_train_dir}/{classDir}")
                os.makedirs(f"{self.new_test_dir}/{classDir}")

                imgs = os.listdir(f"{self.classes_dir}/{classDir}")
                random.shuffle(imgs)

                numTrain = int(0.8 * len(imgs))

                # Iterate through individual TRAIN images within class folders and copy resized imgs
                for imgFileName in imgs[:numTrain]:
                    img = cv2.imread(f"{self.classes_dir}/{classDir}/{imgFileName}")
                    resized = cv2.resize(img, (1000, 1000))
                    cv2.imwrite(f"{self.new_train_dir}/{classDir}/{imgFileName}", resized)

                # Iterate through individual TEST images within class folders and copy resized imgs
                for imgFileName in imgs[numTrain:]:
                    img = cv2.imread(f"{self.classes_dir}/{classDir}/{imgFileName}")
                    resized = cv2.resize(img, (1000, 1000))
                    cv2.imwrite(f"{self.new_test_dir}/{classDir}/{imgFileName}", resized)
    
    # Score function
    def score(self):
        pass

    # Loss Function

# Activation functions
class ActivFunc:
    def relu():
        pass

    def sigmoid():
        pass

    def tanh():
        pass

    def softmax():
        pass



# Layer
class Layer:
    def __init__(self, iDim, oDim, activation=None):
        self.params = np.random.rand(iDim, oDim)

