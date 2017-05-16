import os
import time

import cv2
import face_recognition


from src.helpers import natural_keys


# Compares already known faces with new face image
# returns known_faces which is altered if needed (added new known face or replaced old one with new face)
# idx (int): in order to save compared face to folder
def compare_to_known_faces(new_img, known_faces):

    enc = face_recognition.face_encodings(new_img)

    if len(enc) == 0:
        # Couldn't encode face from image, return known_faces without altering
        return known_faces, -1  # -1 if no face found

    # If no elements in known faces then add first element to it, otherwise cannot compare to anything.
    if len(known_faces) == 0:
        known_faces.append(enc[0])
        return known_faces, 0  # First face index

    # Should be only one face on the image, so take first element
    new_img_enc = enc[0]
    results = face_recognition.compare_faces(known_faces, new_img_enc)  # Compare to known faces

    if not (True in results):  # This means face does not match to other faces
        known_faces.append(new_img_enc)  # Add new face to known faces
        idx = len(known_faces)-1
    else:
        idx = results.index(True)  # Must be at least one True result
        # Replace already known face with new face - because this face is probably more similar to next face
        known_faces[idx] = new_img_enc

    return known_faces, idx


# Compares all faces one by one in the input directory and saves results to target_directory.
# Faces are compared to previously found faces of each group.
def get_final_faces(input_path, target_path, sort_faces=False):

    face_dirs = []
    no_face_path = target_path + "/no_face"

    if sort_faces:
        if not os.path.exists(no_face_path):  # Create directory to target folder if needed
            os.makedirs(no_face_path)

    imgs = os.listdir(input_path)

    imgs.sort(key=natural_keys)  # Sort images

    if not os.path.exists(target_path):  # Create directory to target folder if needed
        os.makedirs(target_path)

    known_faces = []
    face_imgs = []

    i = 0

    for img in imgs:
        # Compare face to already known faces
        face = cv2.imread(input_path + "/" + img)

        known_faces, idx = compare_to_known_faces(face, known_faces)

        # Check if new face added to known faces
        if len(face_imgs) != len(known_faces):
            face_imgs.append(face)  # Add face to face imgs

        # Save face to right directory
        if sort_faces:
            if idx == -1:
                cv2.imwrite('%s/face%i.jpg' % (no_face_path, i), face)
            else:
                if idx+1 > len(face_dirs):  # Add new path where to save given face
                    face_dirs.append((target_path + "/face" + str(idx)))
                    if not os.path.exists(face_dirs[idx]):  # Create directory to target folder if needed
                        os.makedirs(face_dirs[idx])

                # Save face to correct path
                cv2.imwrite('%s/face%i.jpg' % (face_dirs[idx], i), face)

        i += 1

    # Save found faces to target path
    j = 0
    for face in face_imgs:
        cv2.imwrite('%s/face%i.jpg' % (target_path, j), face)  # save face as JPEG file
        j += 1


# Goes through all faces of all videos in input_path to remove repeating faces
# For that compares each face to previously found faces of each people
# Note - Input should contain subdirectories of categories, then subdirs of videos and then face images.
def compare_faces(input_path="faces", target_path="res", sort_faces=False):

    categories = os.listdir(input_path)

    for cat in categories:

        videos = os.listdir(input_path + "/" + cat)

        for vid in videos:

            print("Working on video: " + vid)
            # Faces path for given video
            in_path = "%s/%s/%s" % (input_path, cat, vid)
            # Results path for given video
            res_path = "%s/%s/%s" % (target_path, cat, vid)

            start = time.clock()
            get_final_faces(input_path=in_path, target_path=res_path, sort_faces=sort_faces)
            elapsed = time.clock()
            elapsed = elapsed - start
            print("Time spent: %f" % elapsed)


#compare_faces(sort_faces=True)
