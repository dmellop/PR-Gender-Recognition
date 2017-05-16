# PR-Gender-Recognition

<b>Project:</b> Gender Recognition and People Counting from Videos <br>
<b>Team:</b> Martin Appo, Markus Kängsepp

Video playlist containing all videos used: https://www.youtube.com/playlist?list=PLxbghykXtQogvtFX4N2jfXdR1UFCt6xnV


<b>Folders:</b>
- SRC - contains all code used to count people and recognize gender from videos.
- PLOTS - Contains output plots and code and data used to generate plots (data from data_with_results.xlsx)
- DEMO - contains demo video
- TRAIN_FACES - images used to train gender classifier

<b>Files:</b>
- data_with_results.xlsx - contains actual counts and counts found by algorithms. Also plots Actual Count vs Found by Algorithm and Count of Females vs Found Females


<b>Running a demo: </b>
- run demo.py script, which will find faces from demo video and put them to demo_faces folder and cluster faces to demo_res folder.
- Make sure you have train_faces, haarcascade_frontalface_default.xml and demo/news/*.mp4
- also make sure you have necessary libraries (cv2, sklearn, face_recognition, PIL, numpy)
