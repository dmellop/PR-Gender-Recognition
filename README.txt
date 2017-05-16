Project: Gender Recognition and People Counting from Videos
Repo: https://github.com/markus93/PR-Gender-Recognition
Team: Martin Appo, Markus K�ngsepp

Video playlist containing all videos used: https://www.youtube.com/playlist?list=PLxbghykXtQogvtFX4N2jfXdR1UFCt6xnV


Folders:
-> SRC - contains all code used to count people and recognize gender from videos.
-> PLOTS - Contains output plots and code and data used to generate plots (data from data_with_results.xlsx)
-> DEMO - contains demo video
-> TRAIN_FACES - images used to train gender classifier

Files:
-> data_with_results.xlsx - contains actual counts and counts found by algorithms. Also plots Actual Count vs Found by Algorithm and Count of Females vs Found Females


Running a demo -> run demo.py script, which will find faces from demo video and put them to demo_faces folder and cluster faces to demo_res folder.

-> Make sure you have train_faces, haarcascade_frontalface_default.xml and demo/news/*.mp4
-> also make sure you have necessary libraries (cv2, sklearn, face_recognition, PIL, numpy)