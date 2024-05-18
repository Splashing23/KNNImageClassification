import numpy as np
import os
import cv2
import random
from math import inf


class KNNModel:

    # Create duplicate dir with images resized to 1000 x 1000
    def splitAndTrain(self) -> None:
        classesDir = "cropped_images-20240423T205638Z-001/cropped_images"
        newTrainDir = "modifiedTrain"
        newTestDir = "modifiedTest"

        # If new dir doesn't already exist...
        if not os.path.exists(newTrainDir) and not os.path.exists(newTestDir):
            os.makedirs(newTrainDir)
            os.makedirs(newTestDir)
            
            # Iterate through class folders
            for classDir in os.listdir(classesDir):

                # Copy class folders to new dir
                os.makedirs(f"{newTrainDir}/{classDir}")
                os.makedirs(f"{newTestDir}/{classDir}")

                imgs = os.listdir(f"{classesDir}/{classDir}")
                random.shuffle(imgs)

                numTrain = int(0.8 * len(imgs))

                # Iterate through individual TRAIN images within class folders and copy resized imgs
                for imgFileName in imgs[:numTrain]:
                    img = cv2.imread(f"{classesDir}/{classDir}/{imgFileName}")
                    resized = cv2.resize(img, (1000, 1000))
                    cv2.imwrite(f"{newTrainDir}/{classDir}/{imgFileName}", resized)

                # Iterate through individual TEST images within class folders and copy resized imgs
                for imgFileName in imgs[numTrain:]:
                    img = cv2.imread(f"{classesDir}/{classDir}/{imgFileName}")
                    resized = cv2.resize(img, (1000, 1000))
                    cv2.imwrite(f"{newTestDir}/{classDir}/{imgFileName}", resized)

    # Incrementally rotate query, calculate loss, and return lowest loss
    def SELoss(self, queryImg, XImg) -> int:
        loss = inf
        for i in range(4):
            queryImg = cv2.rotate(queryImg, cv2.ROTATE_90_CLOCKWISE)
            errArr = XImg - queryImg
            sqErrArr = errArr ** 2
            totSE = np.sum(sqErrArr)
            if totSE < loss:
                loss = totSE
        return loss
    
    # Input an image, compare it to all in dataset, then classify based on which is most similar and k value
    def classify(self, queryFile: str, k: int = 2) -> str:
        trainDir = "modifiedTrain"
        queryImg = cv2.imread(queryFile)
        queryImg = cv2.resize(queryImg, (1000, 1000))
        knn = [(str(i), inf) for i in range(k)]

        # Iterate through class folders
        for classDir in os.listdir(trainDir):

            # Iterate through individual images within class folders
            for imgFileName in os.listdir(f"{trainDir}/{classDir}"):
                XImg = cv2.imread(f"{trainDir}/{classDir}/{imgFileName}")
                loss = self.SELoss(queryImg, XImg)
                knn.append((classDir, loss))
                knn = sorted(knn, key = lambda x: x[1])
                knn.pop()
        
        # Find the frequency of the the each class in the kth nearest neighbors
        for i in range(k, -1, -1):
            freq = {}
            for i in range(i):
                key = knn[i][0]
                freq[key] =  freq.get(key, 0) + 1

            maxClass = ""
            maxFreq = 0

            # If duplicate maxClasses exist, reduce k by 1 and repeat finding kth nearest neighbors
            duplicate = False

            # Return class with highest frequency
            for item, count in freq.items():
                if count > maxFreq:
                    maxFreq = count
                    maxClass = item
                    duplicate = False
                elif count == maxFreq:
                    duplicate = True
            if not duplicate:
                break
        
        return maxClass
    
    def eval(self, k=2) -> float:
        
        testDir = "modifiedTest"
        correct = 0
        incorrect = 0

        # Iterate through test class folders
        for classDir in os.listdir(testDir):

            # Iterate through individual images within test class folders
            for imgFileName in os.listdir(f"{testDir}/{classDir}"):
                if classDir == self.classify(f"{testDir}/{classDir}/{imgFileName}", k):
                    correct += 1
                else:
                    incorrect += 1
        
        # Return the percentage of accuracy
        return (correct / (correct + incorrect)) * 100
    
    def tuneHyperParam(self) -> str:
        
        highAcc = 0
        bestk = 0

        # Check different k values and return k with best accuracy
        for i in range(1, 11):
            acc = self.eval(i)
            print(i, acc)
            if acc > highAcc:
                bestk = i
                highAcc = acc

        return f"Best k is {bestk} with accuracy of {highAcc}%"
