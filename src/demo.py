from find_faces_from_videos import videos2faces
from compare_faces import compare_faces
from gender_recognition import GenderRegocnition


if __name__ == "__main__":

    # First let's find faces from videos
    videos2faces(input_folder="demo", target_folder="demo_faces", skip=3, method="haarcascade")
    # Second step is to cluster found faces and remove pictures with no face
    compare_faces(input_path="demo_faces", target_path="demo_res", sort_faces=True)
    # Last let find how many males and females are among found faces
    genderRecognition = GenderRegocnition()
    genderRecognition.evaluateGenders(input_folder="demo_res", train_faces_path="train_faces")
