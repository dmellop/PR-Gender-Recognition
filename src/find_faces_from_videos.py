import face_recognition
import cv2
import os


def recognize_face_dlib(image):

    # Load the jpg file into a numpy array

    # Find all the faces in the image
    face_locations = face_recognition.face_locations(image)

    i = 0

    images = []

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]

        images.append(face_image)

    return images


# Haarcascade method for face recognition
def recognize_face_haarcascade(image):
    faces = regocnizeFaceByImg(image)
    if faces is None:
        return []
    face_images = cropFacesFromImage(image, faces)

    return face_images


def regocnizeFaceByImg(image):
    # Using default opencv provided cascade
    try:
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,  # Using grayscale image
            scaleFactor=1.1,
            # Since some faces may be closer to the camera, they would appear bigger than those faces in the back. The scale factor compensates for this.
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    except cv2.error as e:
        print(e)
        return None

    return faces


def regocnizeFaceByPath(imgPath):
    image = cv2.imread(imgPath)
    faces = regocnizeFaceByImg(image)

    return faces, image


def cropFacesFromImage(image, faces):
    faceImages = []
    for (x, y, w, h) in faces:
        croppedImg = image[y: y + h, x: x + w]  # Crop from x, y, w, h -> 100, 200, 300, 400
        faceImages.append(croppedImg)

    return faceImages


def video_to_faces_haarcascade(video_path, target_path, skip=3):

    vidcap = cv2.VideoCapture(video_path)  # TODO check if video file exists
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if count % (skip + 1) == 0:  # In default: Skip 3 frames, save only every 4th frame
            # print 'Read a new frame: ', success
            faceImages = recognize_face_haarcascade(image)

            i = 0

            for face in faceImages:
                cv2.imwrite("%s/frame%i_face_%i.jpg" % (target_path, count, i), face)  # save frame as JPEG file
                i += 1

            print("Frame nr done: %i" % count)

        count += 1

    vidcap.release()


def video_to_faces_dlib(video_path, target_path, skip=3):

    vidcap = cv2.VideoCapture(video_path)  # TODO check if video file exists
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        if count % (skip + 1) == 0:  # In default: Skip 3 frames, save only every 4th frame
            # print 'Read a new frame: ', success
            faceImages = recognize_face_dlib(image)

            i = 0

            for face in faceImages:
                cv2.imwrite("%s/frame%i_face_%i.jpg" % (target_path, count, i), face)  # save frame as JPEG file
                i += 1

            print("Frame done: %i" % count)

        count += 1

        success, image = vidcap.read()


    vidcap.release()


# Split video to frames (skipping 3 frames by default) and then look for faces on each frame and finally save these to
# target folder for further work
# input_folder (str): Folder that is checked for videos
# target_folder (str): Folder where video frames are added
# skip (int): how many frames are skipped
# method: what method are used for face detection
def videos2faces(input_folder="videos", target_folder="faces", skip=3, method="haarcascade"):
    try:
        dirs = os.listdir(input_folder)

        # Go through all subdirectories
        for dir_name in dirs:  # d - keyword inserted into youtube (to group videos)
            d_path = "%s/%s" % (input_folder, dir_name)
            videos = os.listdir(d_path)


            # Go through all the videos
            for v_name in videos:  # get all video names
                video_path = "%s/%s" % (d_path, v_name)
                v_name_new = ".".join(v_name.split(".")[:-1])  # Remove file extension from the video name
                v_name_new = v_name_new.replace("?", "")  # Remove question marks from folder path.
                target_path = "%s/%s/%s" % (target_folder, dir_name, v_name_new)

                if not os.path.exists(target_path):  # Create directory to target folder if needed
                    os.makedirs(target_path)

                print("Working on video name:" + v_name)

                # Convert video to frames and faces
                if method == "dlib":
                    video_to_faces_haarcascade(video_path, target_path, skip)
                elif method == "haarcascade":
                    video_to_faces_haarcascade(video_path, target_path, skip)
                else:
                    print("No such method")
                    return

            print("Videos are processed for directory: " + dir_name)

    except os.error as e:
        print(e)


#videos2faces(input_folder="videos", target_folder="faces", skip=3, method="haarcascade")

