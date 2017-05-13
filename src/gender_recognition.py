import cv2
import time
import os
from PIL import Image
import numpy as np
from sklearn.decomposition import FastICA
from sklearn import svm
from image_preparation import crop_face
import matplotlib.pyplot as plt

def evaluateAndCountGenders(ica, classifier, inputFolder="../result_faces"):
    resultFaces = []
    femaleCount = 0
    maleCount = 0

    print("Target;Female count;Male count")

    try:
        themeDirs = os.listdir(inputFolder)

        # Go through all theme directiories
        for themeDirName in themeDirs:
            topicFemaleCount = 0
            topicMaleCount = 0
            videoNameDirPath = "%s/%s" % (inputFolder, themeDirName)
            videoDirNames = os.listdir(videoNameDirPath)

            # Go trough all video directories
            for videoDirName in videoDirNames:
                vidFemaleCount = 0
                vidMaleCount = 0
                faceImagesDirPath = "%s/%s" % (videoNameDirPath, videoDirName)
                faceImages = os.listdir(faceImagesDirPath)

                # Go through all face images
                for faceImageName in faceImages:
                    faceImagePath = "%s/%s" % (faceImagesDirPath, faceImageName)
                    faceImg = crop_face(Image.open(faceImagePath), offset_pct=(0.1,0.1))
                    flattenFaceImg = np.array(faceImg).flatten()

                    features = ica.transform([flattenFaceImg])
                    result = classifier.predict(features)
                    if (result[0] == "f"):
                        vidFemaleCount +=1
                        topicFemaleCount +=1
                        femaleCount +=1
                    else:
                        vidMaleCount +=1
                        topicMaleCount += 1
                        maleCount +=1

                print("Video " + videoDirName + ";" + str(vidFemaleCount) + ";" + str(vidMaleCount))

            print("Topic " + themeDirName + ";" + str(topicFemaleCount) + ";" + str(topicMaleCount))

        print("Total;" + str(femaleCount) + ";" + str(maleCount))

    except os.error as e:
        print(e)



def loadTrainingImages(prefix, imgs, targets):
    imgDirPath = '../train_faces/' + prefix

    try:
        imgPaths = os.listdir(imgDirPath)

        for imgPath in imgPaths:
            imgFullPath = "%s/%s" % (imgDirPath, imgPath)
            img = crop_face(Image.open(imgFullPath).convert('RGB'), offset_pct=(0.3,0.3))
            imgs.append(np.array(img).flatten())
            targets.append(prefix)

    except os.error as e:
        print(e)

def getIcaAndClassifier(imgs, targets, componentsCnt):
    # Calc ICA
    ica = FastICA(n_components=componentsCnt, whiten=True)
    ica.fit(imgs)
    features = ica.transform(imgs)

    # Fit SVM
    classifier = svm.SVC(gamma=0.001, C=componentsCnt)
    classifier.fit(features, targets)

    # Predict
    print("Prediction done with " + str(componentsCnt) + " features")

    return ica, classifier

def evaluateGenders():
    trainingImgs = []
    targets = []

    loadTrainingImages('m', trainingImgs, targets)
    loadTrainingImages('f', trainingImgs, targets)

    ica, classifier = getIcaAndClassifier(trainingImgs, targets, 10)

    evaluateAndCountGenders(ica, classifier)



if __name__ == "__main__":
    evaluateGenders()
