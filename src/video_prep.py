import cv2


# Converts video to frames
# video_path: Location of the input video file
# target_path: Location of the output frames
# skip: how many frames are going to be skipped


def video2frames(video_path, target_path, skip=3):

    vidcap = cv2.VideoCapture(video_path)  # TODO check if video file
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if count % (skip + 1) == 0:  # In default: Skip 3 frames, save only every 4th frame
            # print 'Read a new frame: ', success
            cv2.imwrite("%s/frame%d.jpg" % (target_path, count), image)  # save frame as JPEG file
        count += 1

    vidcap.release()
