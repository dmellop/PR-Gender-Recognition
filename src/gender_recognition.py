import os
from PIL import Image
import numpy as np
from sklearn.decomposition import FastICA
from sklearn import svm
from image_preparation import crop_face

class GenderRegocnition:

    predictionMethod = "ica"

    def evaluateAndCountGenders(self, ica=None, classifier=None, inputFolder="../result_faces"):
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

                        if (self.predictionMethod == "ica"):
                            features = ica.transform([flattenFaceImg])
                            result = classifier.predict(features)

                        if (result[0] == "f"):
                            vidFemaleCount +=1
                            topicFemaleCount +=1
                            femaleCount +=1
                        elif (result[0] == "m"):
                            vidMaleCount +=1
                            topicMaleCount += 1
                            maleCount +=1

                    print("Video " + videoDirName + ";" + str(vidFemaleCount) + ";" + str(vidMaleCount))

                print("Topic " + themeDirName + ";" + str(topicFemaleCount) + ";" + str(topicMaleCount))

            print("Total;" + str(femaleCount) + ";" + str(maleCount))

        except os.error as e:
            print(e)


    def loadTrainingImages(self, prefix, imgs, targets):
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

    def getIcaAndClassifier(self, imgs, targets, componentsCnt):
        # Calc ICA
        ica = FastICA(n_components=componentsCnt, whiten=True)
        ica.fit(imgs)
        features = ica.transform(imgs)

        # Fit SVM
        classifier = svm.SVC(gamma=0.01, C=componentsCnt)
        classifier.fit(features, targets)

        # Predict
        print("Prediction done with " + str(componentsCnt) + " features")

        return ica, classifier

    def evaluateGenders(self):
        if (self.predictionMethod == "ica"):
            trainingImgs = []
            targets = []

            self.loadTrainingImages('m', trainingImgs, targets)
            self.loadTrainingImages('f', trainingImgs, targets)

            ica, classifier = self.getIcaAndClassifier(trainingImgs, targets, 25)

            self.evaluateAndCountGenders(ica, classifier)
        elif (self.predictionMethod == "angus"):
            self.evaluateAndCountGenders()


if __name__ == "__main__":
    genderRecognition = GenderRegocnition()
    genderRecognition.evaluateGenders()
